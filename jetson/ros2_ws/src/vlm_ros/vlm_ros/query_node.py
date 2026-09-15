import rclpy
import cv2
import os
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from interfaces.srv import DetectSemantics

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class QueryNode(Node):
    def __init__(self):
        super().__init__('query_node')

        # --- Parameters ---
        self.declare_parameter('target_query', 'chair, drone')

        self.declare_parameter('save_images', False)
        self.declare_parameter('save_dir', 'saved_images')

        self.target_query = self.get_parameter('target_query').value

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # Publishers & Subscribers
        self.camera_info_sub  = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.camera_image_sub = self.create_subscription(Image,      '/camera/camera/color/image_raw'  , self.image_callback, 10)

        self.image_pub      = self.create_publisher(Image, "acquired_img", 10)
        self.annotated_pub  = self.create_publisher(Image, "/annotated_img", 10)

        self.latest_image = None

        # Services
        self.acquire_img_srv = self.create_service(Trigger, 'acquire_image', self.handle_acquire_image_request, callback_group=self.cb_group)
        self.vlm_client      = self.create_client(DetectSemantics, 'detect_semantic_clues', callback_group=self.cb_group)
        
        self.get_logger().info("Query Node is ready.")

    # --- Callbacks ---
    def camera_info_callback(self, msg:CameraInfo):
        self.camera_info = msg
        self.intrinsic_matrix = msg.k.reshape((3, 3))
    
    def image_callback(self, msg:Image):
        self.latest_image = msg
        
    def handle_acquire_image_request(self, request, response):

        # Snapshot the latest image for processing
        image_to_process = self.latest_image

        if self.latest_image is None:
            response.success = False
            response.message = "No image received yet. Try again."
            return response

        self.image_pub.publish(image_to_process)

        save_images = self.get_parameter('save_images').value
        save_dir = self.get_parameter('save_dir').value

        if save_images:
            try:
                os.makedirs(save_dir, exist_ok=True)
                cv_acquired = self.bridge.imgmsg_to_cv2(image_to_process, desired_encoding='bgr8')
                acquired_path = os.path.join(save_dir, f"acquired.png")
                cv2.imwrite(acquired_path, cv_acquired)
                self.get_logger().info(f"Saved acquired image to: {acquired_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save acquired image: {e}")
        
        if not self.vlm_client.wait_for_service(timeout_sec=3.0):
            response.success = False
            response.message = "VLM service 'detect_semantic_clues' not available."
            return response

        # VLM Query
        target_string = self.target_query
        
        vlm_req = DetectSemantics.Request()
        vlm_req.image = image_to_process
        vlm_req.target_query = target_string

        self.get_logger().info("Blocking call to VLM...")
        vlm_res = self.vlm_client.call(vlm_req)
        
        if vlm_res.success:
            num_detections = len(vlm_res.detections)
            response.success = True
            response.message = f"VLM processed successfully. Found {num_detections} semantic clues."
            
            try:
                # Convert ROS img into OpenCV (BGR)
                cv_image = self.bridge.imgmsg_to_cv2(image_to_process, desired_encoding='bgr8')
                
                target_items = [item.strip() for item in target_string.split(',')]
                label_dict = {item: 0.0 for item in target_items}
                
                # Create the label_dict dynamically from the target_query string
                self._publish_annotated_bboxes(cv_image, vlm_res.detections, label_dict, image_to_process.header.frame_id, save_images, save_dir)
                self.get_logger().info(f"Inference time: {vlm_res.inference_time_sec:.2f}s")
            except Exception as e:
                self.get_logger().error(f"Error during image annotation: {e}")

        else:
            response.success = False
            response.message = "VLM processing failed to detect semantics."

        return response

    # --- UTILS ---
    def _publish_annotated_bboxes(self, frame_bgr, detections, label_dict, frame_id, save_images, save_dir):
        """Publishes the annotated image with bounding boxes and labels."""
        annotated = frame_bgr.copy()
        
        for det in detections:
            x1, y1 = int(det.x_min), int(det.y_min)
            x2, y2 = int(det.x_max), int(det.y_max)
            label = det.label
            
            # Identify if the detection is target (red) or a hint (green)
            is_target = any((k.lower() in det.label.lower() and label_dict[k] == 0.0) for k in label_dict)
            color = (0, 0, 255) if is_target else (0, 255, 0)
            
            # Draw the bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Compute text dimensions
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # If bbox is in the upper part of the image, draw the text box inside
            bg_y1 = y1 - text_h - 6
            bg_y2 = y1
            txt_y = y1 - 5
            
            if bg_y1 < 0:
                bg_y1 = y1
                bg_y2 = y1 + text_h + 6
                txt_y = y1 + text_h + 1

            # Draw the background rectangle for text
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w, bg_y2), color, -1)

            # Draw the label text
            cv2.putText(annotated, label, (x1, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if save_images:
            try:
                os.makedirs(save_dir, exist_ok=True)
                annotated_path = os.path.join(save_dir, f"annotated.png")
                cv2.imwrite(annotated_path, annotated)
                self.get_logger().info(f"Saved annotated image to: {annotated_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save annotated image: {e}")

        try:
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            
            msg.header.frame_id = frame_id
            
            self.annotated_pub.publish(msg)
            self.get_logger().info("Annotated image published on /annotated_img")
        except Exception as e:
            self.get_logger().warn(f"Failed to publish annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = QueryNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()