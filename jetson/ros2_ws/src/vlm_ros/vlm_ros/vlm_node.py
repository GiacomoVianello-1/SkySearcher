import json
import re
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import cv2
from PIL import Image as PILImage

from interfaces.srv import DetectSemantics
from interfaces.msg import SemanticBBox

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

class QwenVLMNode(Node):
    def __init__(self):
        super().__init__('qwen_vlm_node')

        # We want to reject predicted bboxes that are unreasonably large (e.g. covering the 60% of the image) 
        # as they are likely to be hallucinations. They are non-informative for our purposes. 
        self.declare_parameter('max_bbox_area_ratio', 0.60)

        self.bridge = CvBridge()
        
        # --- Threading & Concurrency Setup ---
        self._request_lock = threading.Lock()
        self.cb_group = ReentrantCallbackGroup()
        
        # Publisher to broadcast the VLM's busy status to the rest of the ROS network
        self.vlm_busy_publisher = self.create_publisher(Bool, "vlm_busy", 10)

        # --- Model Setup ---
        # Note: Keeping the VL model required for bounding box extraction
        self.model_id = "Qwen/Qwen3-VL-4B-Instruct" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.get_logger().info(f"Loading {self.model_id} on {self.device}...")
        self._load_model()
        
        # --- Service Setup ---
        self.srv = self.create_service(
            DetectSemantics, 
            'detect_semantic_clues', 
            self.handle_detection_request,
            callback_group=self.cb_group
        )
        self.get_logger().info("Qwen VLM Node is ready. Listening on 'detect_semantic_clues'.")

    def _load_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, 
                dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Optional: Enforce deterministic generation without sampling
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.do_sample = False
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
                
            if self.device == "cpu":
                self.model.to(self.device)
                
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    # --- Concurrency Management ---
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

    # --- Service Callback ---
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
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 5. JSON Parsing
            match = re.search(r'\[\s*\{.*?\}\s*\]', output_text, flags=re.DOTALL)
            if not match:
                self.get_logger().warn(f"[vlm/detect] No JSON array found in output. Raw: {output_text}")
                response.success = False
                return response

            try:
                detections = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                self.get_logger().error(f"[vlm/detect] JSON parsing error: {e}")
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

def main(args=None):
    rclpy.init(args=args)
    node = QwenVLMNode()
    
    # Required to allow the ReentrantCallbackGroup to process requests concurrently
    # (though our lock ensures only one inference runs at a time, this prevents ROS from freezing)
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