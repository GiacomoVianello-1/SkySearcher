#!/usr/bin/env python3

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json
import re
import threading

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
        self.declare_parameter('max_bbox_area_ratio', 0.60)
        self.bridge = CvBridge()

        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self._request_lock = threading.Lock()

        self.vlm_busy_publisher = self.create_publisher(Bool, "/vlm_busy", 10, callback_group=self.io_cb_group)

        self.model_id = "Qwen/Qwen3-VL-4B-Instruct" # "Qwen/Qwen3-VL-4B-Instruct" or "Qwen/Qwen3-VL-4B-Instruct-FP8"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading {self.model_id} on {self.device}...")
        self._load_model()

        self.srv            = self.create_service(DetectSemantics, 'detect_semantic_clues', self.handle_detection_request, callback_group=self.service_cb_group)
        self.verify_srv     = self.create_service(VerifyTarget, '/verify_target', self.handle_verify_target, callback_group=self.service_cb_group)

        self.get_logger().info("VLM Node is ready.")

    def _load_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.do_sample = False    # Deterministic sampling for reproducibility.
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
            if self.device == "cpu":
                self.model.to(self.device)
            self.get_logger().info(f"Model loaded successfully. Config:")
            self.get_logger().info(f"Do Sample: {self.model.generation_config.do_sample}; Temperature: {self.model.generation_config.temperature}; Top-P: {self.model.generation_config.top_p}; Top-K: {self.model.generation_config.top_k}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Concurrency management                                            #
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
    #  Helper methods for Image & BBox Processing                        #
    # ------------------------------------------------------------------ #

    def _convert_msg_to_pil(self, img_msg: Image):
        """Converts ROS Image message to BGR cv2 image, RGB PIL Image, and dimensions (img_w, img_h)."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            img_h, img_w = cv_image.shape[:2]
            return cv_image, pil_image, img_w, img_h
        except Exception as e:
            self.get_logger().error(f"Image decoding failed: {e}")
            return None, None, 0, 0

    def _normalize_and_filter_detections(self, detections: list, img_w: int, img_h: int, max_area_ratio: float = None) -> list:
        """Normalizes Qwen [0,1000] bboxes to absolute pixel coordinates and filters out overly large boxes."""
        if max_area_ratio is None:
            max_area_ratio = float(self.get_parameter('max_bbox_area_ratio').value)
        img_area = float(img_w * img_h)
        valid = []

        for det in detections:
            if "bbox_2d" not in det or "label" not in det:
                continue
            try:
                x1, y1, x2, y2 = map(float, det["bbox_2d"])
                
                # Convert from Qwen's normalized [0, 1000] grid to absolute pixels if necessary
                if max(x1, x2) <= 1000.0 and max(y1, y2) <= 1000.0:
                    x1_abs = (x1 * float(img_w)) / 1000.0
                    y1_abs = (y1 * float(img_h)) / 1000.0
                    x2_abs = (x2 * float(img_w)) / 1000.0
                    y2_abs = (y2 * float(img_h)) / 1000.0
                else:
                    x1_abs, y1_abs, x2_abs, y2_abs = x1, y1, x2, y2

                bbox_area = (x2_abs - x1_abs) * (y2_abs - y1_abs)
                bbox_ratio = bbox_area / img_area if img_area > 0 else 0.0

                if bbox_ratio > max_area_ratio:
                    self.get_logger().warn(f"Bbox ignored ({det['label']}) as it covers {bbox_ratio*100:.1f}% of the image (limit: {max_area_ratio*100:.1f}%)")
                    continue

                valid.append({
                    "label": str(det["label"]),
                    "bbox_2d": [x1_abs, y1_abs, x2_abs, y2_abs],
                    "bbox_ratio": bbox_ratio
                })
            except Exception as e:
                self.get_logger().warn(f"Ignored malformed bounding box: {det} | Error: {e}")

        return valid

    # ------------------------------------------------------------------ #
    #  Core detection logic (shared by all services)                     #
    # ------------------------------------------------------------------ #

    def _run_detection(self, pil_image: PILImage.Image, query: str, repetition_penalty: float = 1.0) -> list:
        """Run VLM detection on a single image. Returns list of detection dicts."""

        prompt_text = f"Locate every instance that belongs to the following categories: {query}. Report bbox coordinates in JSON format."
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

        self.get_logger().info("[vlm/detect] Generating...")
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, repetition_penalty=repetition_penalty)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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

    # ------------------------------------------------------------------ #
    #  ROS Service Handlers                                              #
    # ------------------------------------------------------------------ #

    def handle_detection_request(self, request: DetectSemantics.Request, response: DetectSemantics.Response):
        self.get_logger().info(f"[vlm/detect] Received request for: '{request.target_query}'")
        
        if not self._try_acquire():
            self.get_logger().warn("[vlm/detect] Request rejected. VLM is currently busy processing another image.")
            response.success = False
            return response

        try:
            _, pil_image, img_w, img_h = self._convert_msg_to_pil(request.image)
            if pil_image is None:
                response.success = False
                return response

            raw_detections = self._run_detection(pil_image, request.target_query)
            if not raw_detections:
                self.get_logger().warn(f"[vlm/detect] No detections found.")
                response.success = False
                return response
            
            valid_detections = self._normalize_and_filter_detections(raw_detections, img_w, img_h)

            response.detections = []
            for det in valid_detections:
                bbox_msg = SemanticBBox()
                bbox_msg.label = det["label"]
                bbox_msg.x_min, bbox_msg.y_min, bbox_msg.x_max, bbox_msg.y_max = det["bbox_2d"]
                response.detections.append(bbox_msg)

                #self.get_logger().info(f"Detected '{bbox_msg.label}'")

            self.get_logger().info(f"VLM generated {len(response.detections)} detections.")
            response.success = True
            return response

        finally:
            self._release()

    def handle_verify_target(self, request: VerifyTarget.Request, response: VerifyTarget.Response) -> VerifyTarget.Response:
        #self.get_logger().info(f"[verify] Received request for: '{request.target_query}'")

        if not self._try_acquire():
            self.get_logger().warn("[verify] VLM busy, request rejected.")
            response.confirmed = False
            response.message   = "VLM busy"
            return response

        try:
            _, pil_image, img_w, img_h = self._convert_msg_to_pil(request.image)
            if pil_image is None:
                response.confirmed = False
                response.message   = "Image decode failed"
                return response

            target_obj = request.target_query.strip()
            obj_desc = getattr(request, 'object_description', '').strip()

            #self.get_logger().info(f"[verify] Verifying Target: '{target_obj}' | Desc: '{obj_desc}'")

            # Strip qualitative framing prefix if present
            clean_desc = re.sub(r'QUALITATIVE FRAMING QUERY:?\s*', '', obj_desc, flags=re.IGNORECASE).strip()
            desc_text = f"Visual Description: {clean_desc}." if clean_desc else ""

            prompt_text = (
                f"Target Object: '{target_obj}'.\n"
                f"Check if this exact target object is present in the image, strictly matching all visual attributes (such as color, pattern, material).\n\n"
                f"Respond ONLY in JSON format: {{\"present\": true, \"reasoning\": \"<brief explanation>\"}}"
            )

            # Run VLM with decision prompt
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)

            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            confirmed = False
            reasoning = output_text
            bbox_str = "[]"

            clean_text = re.sub(r'```json\s*', '', output_text)
            clean_text = re.sub(r'```\s*', '', clean_text)
            match = re.search(r'\{.*?\}', clean_text, flags=re.DOTALL)

            if match:
                try:
                    res_json = json.loads(match.group(0))
                    present_val = res_json.get("present", False)
                    if isinstance(present_val, str):
                        present = present_val.lower() == "true"
                    else:
                        present = bool(present_val)

                    reasoning = str(res_json.get("reasoning", ""))

                    if present:
                        confirmed = True
                except Exception as e:
                    self.get_logger().warn(f"[verify] JSON parse error: {e}")
                    if '"present": true' in clean_text.lower() or '"present":true' in clean_text.lower():
                        confirmed = True
            else:
                if "true" in clean_text.lower() and "not present" not in clean_text.lower() and "false" not in clean_text.lower():
                    confirmed = True

            # Automatic bbox extraction using Qwen's native detection format if present or confirmed
            bbox_match = re.search(r'(?:bbox|box_2d|bbox_2d)[\"\']?\s*:\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', output_text, flags=re.IGNORECASE)
            if bbox_match:
                bbox_str = f"[{bbox_match.group(1)},{bbox_match.group(2)},{bbox_match.group(3)},{bbox_match.group(4)}]"
            elif confirmed:
                try:
                    dets = self._run_detection(pil_image, target_obj)
                    if dets:
                        valid_dets = self._normalize_and_filter_detections(dets, img_w, img_h, max_area_ratio=1.0)
                        if valid_dets:
                            b = valid_dets[0]["bbox_2d"]
                            ymin_s = int((b[1] / img_h) * 1000)
                            xmin_s = int((b[0] / img_w) * 1000)
                            ymax_s = int((b[3] / img_h) * 1000)
                            xmax_s = int((b[2] / img_w) * 1000)
                            bbox_str = f"[{ymin_s},{xmin_s},{ymax_s},{xmax_s}]"
                except Exception as e:
                    self.get_logger().warn(f"[verify] Automatic detection fallback error: {e}")

            response.confirmed = confirmed
            response.message = f"BBOX: {bbox_str} | Confirmed: {confirmed} | Reason: {reasoning}"
            self.get_logger().info(f"🤖 Verified: {confirmed}. {reasoning}")

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