import json
import re
import threading
import time as _time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import cv2
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from interfaces.srv import DetectSemantics, VerifyTarget
from interfaces.msg import SemanticBBox

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup



class QwenVLMNode(Node):
    def __init__(self):
        super().__init__('qwen_vlm_node')

        # --- Parameters ---
        self.declare_parameter('max_bbox_area_ratio', 0.60)
        
        self.bridge = CvBridge()

        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self._request_lock = threading.Lock()

        self.vlm_busy_publisher = self.create_publisher(Bool, "vlm_busy", 10, callback_group=self.io_cb_group)

        self.model_id = "Qwen/Qwen3-VL-4B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading model on {self.device}...")
        self._load_model()

        # --- Services ---
        self.srv            = self.create_service(DetectSemantics, 'detect_semantic_clues', self.handle_detection_request, callback_group=self.service_cb_group)
        self.verify_srv     = self.create_service(VerifyTarget, '/verify_target', self.handle_verify_target, callback_group=self.service_cb_group)

        self.get_logger().info(f"Model {self.model_id} loaded successfully on {self.device}. VLM Node is ready.")

    def _load_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Configurazione specifica per Jetson / CUDA
            if self.device == "cuda":
                kwargs = {
                    "dtype": torch.float16,
                    "device_map": {"": 0}, # Force everything on GPU (avoid bugs about Jetson unified memory)
                    "low_cpu_mem_usage": True # Avoid the RAM usage peak at startup
                }
            else:
                kwargs = {
                    "dtype": torch.float32,
                    "device_map": None
                }

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **kwargs
            )
            
            # Configure generation parameters to disable sampling and ensure deterministic output
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.do_sample = False
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
                
            # .to(self.device) is only needed if we are on CPU. If we are on CUDA, device_map={"": 0} has already moved everything correctly.
            if self.device == "cpu":
                self.model.to(self.device)

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Concurrency management                                              #
    # ------------------------------------------------------------------ #

    def _set_busy(self, is_busy: bool):
        msg = Bool()
        msg.data = is_busy
        self.vlm_busy_publisher.publish(msg)

    def _try_acquire(self) -> bool:
        acquired = self._request_lock.acquire(blocking=False)
        if acquired:
            self._set_busy(True)
        return acquired

    def _release(self):
        if self._request_lock.locked():
            self._request_lock.release()
        self._set_busy(False)

    # ------------------------------------------------------------------ #
    #  Core detection logic (shared by both services)                     #
    # ------------------------------------------------------------------ #

    def _run_detection(self, pil_image: PILImage.Image, query: str, repetition_penalty: float = 1.0) -> list:
        """Run VLM detection on a single image. Returns list of detection dicts."""
        prompt_text = (
            f"Locate every instance that belongs to the following categories: {query}. Report bbox coordinates in JSON format."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": prompt_text},
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, repetition_penalty=repetition_penalty)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_detections(output_text)

    def _parse_detections(self, output_text: str) -> list:
        """Parse VLM output into list of detection dicts. Handles markdown and truncated JSON."""
        output_text = re.sub(r'```json\s*', '', output_text)
        output_text = re.sub(r'```\s*', '', output_text)

        # Try full array parse first
        detections = []
        match = re.search(r'\[\s*\{.*?\}\s*\]', output_text, flags=re.DOTALL)
        if match:
            try:
                detections = json.loads(match.group(0))
                return detections
            except json.JSONDecodeError:
                pass

        # Fallback: recover complete objects from truncated JSON
        array_start = output_text.find('[')
        if array_start != -1:
            json_fragment = output_text[array_start:]
            for obj_match in re.finditer(r'\{[^{}]*\}', json_fragment):
                try:
                    obj = json.loads(obj_match.group(0))
                    if "bbox_2d" in obj and "label" in obj:
                        detections.append(obj)
                except json.JSONDecodeError:
                    continue

        return detections

    def _deduplicate_detections(self, detections: list, iou_threshold: float = 0.9) -> list:
        kept = []
        for det in detections:
            if "bbox_2d" not in det:
                continue
            x1, y1, x2, y2 = det["bbox_2d"]
            is_dup = False
            for prev in kept:
                # Solo deduplicazione same-label
                if prev.get("label") != det.get("label"):
                    continue
                px1, py1, px2, py2 = prev["bbox_2d"]
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter
                if union > 0 and inter / union > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        return kept

    #  /detect_semantic_clues handler 

    def handle_detection_request(self, request: DetectSemantics.Request, response: DetectSemantics.Response):
        self.get_logger().info(f"[vlm/detect] Received request for: '{request.target_query}'")
        
        # 1. Concurrency Check
        if not self._try_acquire():
            self.get_logger().warn("[vlm/detect] Request rejected. VLM is currently busy processing another image.")
            response.success = False
            return response

        try:
            # 2. Image Decoding
            try:
                cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(cv_image_rgb)
                img_h, img_w = cv_image.shape[:2]
            except Exception as e:
                self.get_logger().error(f"[vlm/detect] Image decoding failed: {e}")
                response.success = False
                return response

            # 3. Prompt Construction

            # This is the prompt template suggested by the official Qwen page
            #prompt_text = (
            #    f"Detect {target_query} in the image and return their locations in the form of coordinates. Report bbox coordinates in JSON format."
            #)

            prompt_text = (
                f"Locate every instance that belongs to the following categories: {request.target_query}. Report bbox coordinates in JSON format."
            )
            
            #prompt_text = (
            #    f"Where can we search for {target_object} in the image? Report bbox coordinates in JSON format."
            #)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 4. Model Inference
            self.get_logger().info("[vlm/detect] Running inference...")
            _t0 = _time.monotonic()
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.0)
            response.inference_time_sec = float(_time.monotonic() - _t0) # Measure inference time and include in response
            self.get_logger().info(f"[vlm/detect] Inference time: {response.inference_time_sec:.2f}s")
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 5. JSON Parsing
            detections = self._parse_detections(output_text)
            if not detections:
                self.get_logger().warn(f"[vlm/detect] No detections found.")
                response.success = False
                return response
            
            # 6. Coordinate Normalization (1000x1000 to Absolute Pixels)
            max_area_ratio = self.get_parameter('max_bbox_area_ratio').value
            img_area = img_w * img_h

            response.detections = []
            for det in detections:
                if "bbox_2d" in det and "label" in det:
                    try:
                        x1, y1, x2, y2 = map(float, det["bbox_2d"])
                        
                        x1_abs = (x1 * img_w) / 1000.0
                        y1_abs = (y1 * img_h) / 1000.0
                        x2_abs = (x2 * img_w) / 1000.0
                        y2_abs = (y2 * img_h) / 1000.0

                        bbox_area = (x2_abs - x1_abs) * (y2_abs - y1_abs)
                        bbox_ratio = bbox_area / img_area

                        if bbox_ratio > max_area_ratio:
                            self.get_logger().warn(
                                f"Bbox ignored ({det['label']}): covers {bbox_ratio*100:.1f}% of the image (limit: {max_area_ratio*100:.1f}%)"
                            )
                            continue # Skip this detection

                        bbox_msg = SemanticBBox()
                        bbox_msg.label = str(det["label"])
                        bbox_msg.x_min = x1_abs
                        bbox_msg.y_min = y1_abs
                        bbox_msg.x_max = x2_abs
                        bbox_msg.y_max = y2_abs
                        
                        response.detections.append(bbox_msg)
                        self.get_logger().info(f"Detected '{bbox_msg.label}' at [{bbox_msg.x_min:.0f}, {bbox_msg.y_min:.0f}, {bbox_msg.x_max:.0f}, {bbox_msg.y_max:.0f}]")
                    except Exception as e:
                        self.get_logger().warn(f"[vlm/detect] Ignored malformed bounding box: {det} | Error: {e}")

            response.success = True
            return response

        finally:
            # Ensures the lock is always released and the busy state is reset,
            # even if an unexpected error occurs during inference.
            self._release()

    def handle_verify_target(self, request: VerifyTarget.Request, response: VerifyTarget.Response) -> VerifyTarget.Response:

        self.get_logger().info(f"[verify] Received request for: '{request.target_query}'")

        if not self._try_acquire():
            self.get_logger().warn("[verify] VLM busy, request rejected.")
            response.confirmed = False
            response.message   = "VLM busy"
            return response

        try:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
                pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                self.get_logger().error(f"[verify] Image decode failed: {e}")
                response.confirmed = False
                response.message   = "Image decode failed"
                return response

            detections = self._run_detection(pil_image, request.target_query)

            if not detections:
                response.confirmed = False
                response.message   = "No detections"
                return response

            # Filter by area
            img_h, img_w = cv_image.shape[:2]
            img_area     = img_w * img_h
            max_ratio    = self.get_parameter('max_bbox_area_ratio').value

            valid = [
                d for d in detections
                if "bbox_2d" in d and "label" in d
                and ((d["bbox_2d"][2] - d["bbox_2d"][0]) *
                    (d["bbox_2d"][3] - d["bbox_2d"][1]) *
                    img_w * img_h / 1e6) / img_area <= max_ratio
            ]

            confirmed = len(valid) > 0
            response.confirmed = confirmed
            response.message   = (
                f"Confirmed: {[d['label'] for d in valid]}"
                if confirmed else "Not confirmed"
            )
            self.get_logger().info(f"[verify] {response.message}")

        finally:
            self._release()

        return response

# --- Main ---

def main(args=None):
    rclpy.init(args=args)
    node = QwenVLMNode()
    executor = MultiThreadedExecutor() 
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()