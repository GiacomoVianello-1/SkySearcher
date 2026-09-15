#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge

# Ensure you use the correct package name where srv and msg are defined
from interfaces.srv import DetectSemantics

class VLMTestClient(Node):
    def __init__(self):
        super().__init__('vlm_test_client')
        
        # Create the service client
        self.cli = self.create_client(DetectSemantics, 'detect_semantic_clues')
        
        # Wait for the VLM/Qwen node to come online
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service 'detect_semantic_clues'...")
            
        self.req = DetectSemantics.Request()
        self.bridge = CvBridge()

    def send_request(self, image_path, target_query):
        self.get_logger().info(f"Loading image from: {image_path}")
        
        # 1. Read image with OpenCV
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            self.get_logger().error(f"Could not find image at: {image_path}")
            return None, None

        # 2. Convert OpenCV -> ROS Image Message
        self.req.image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.req.target_query = target_query

        self.get_logger().info(f"Sending request for: '{target_query}'...")
        
        # 3. Asynchronous service call
        self.future = self.cli.call_async(self.req)
        
        # Wait for the response by blocking this single test thread
        rclpy.spin_until_future_complete(self, self.future)
        
        return self.future.result(), cv_image

def main(args=None):
    rclpy.init(args=args)
    
    test_client = VLMTestClient()
    
    # --- Test Parameters ---
    image_path = "../test_images/frame_736.png" # Change path if file is in a different directory
    target_query = "lake, building, fountain, car, park path"
    output_path = "../test_images/ros_annotated_frame.png"
    
    # Execute the request
    response, original_cv_image = test_client.send_request(image_path, target_query)
    
    # Analyze Outcome
    if response is not None:
        if response.success:
            test_client.get_logger().info(f"Service completed successfully. Found {len(response.detections)} bboxes.")
            
            # Draw results on the original image
            for det in response.detections:
                x1, y1 = int(det.x_min), int(det.y_min)
                x2, y2 = int(det.x_max), int(det.y_max)
                label = det.label
                
                # Draw the Bbox rectangle (Red)
                cv2.rectangle(original_cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw text background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(original_cv_image, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 0, 255), -1)
                
                # Write text label
                cv2.putText(original_cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                test_client.get_logger().info(f" -> {label}: [{x1}, {y1}, {x2}, {y2}]")
                
            # Save final image
            cv2.imwrite(output_path, original_cv_image)
            test_client.get_logger().info(f"Annotated image saved to: {output_path}")
            
        else:
            test_client.get_logger().error("Service returned success = False. Check server logs.")
    else:
        test_client.get_logger().error("Service call failed (Network exception or server crashed).")

    test_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()