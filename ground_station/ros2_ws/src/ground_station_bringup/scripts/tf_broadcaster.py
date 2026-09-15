#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import re

class VrpnTfBroadcaster(Node):
    """Republishes mocap poses as TF. Indoor only; outdoors the pose comes from RTK."""

    def __init__(self):
        super().__init__('vrpn_tf_broadcaster')

        # Defaults mirror config/tf_broadcaster.yaml, which is loaded by the launch file.
        self.pose_topic_pattern = re.compile(self.declare_parameter(
            'pose_topic_pattern', r'^/vrpn_mocap/(.+)/pose$').value)
        discovery_period = self.declare_parameter(
            'discovery_period', 2.0).value

        self.broadcaster = TransformBroadcaster(self)
        self.subscriptions_ = []
        self.qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.timer = self.create_timer(discovery_period, self.discover_topics)

    def discover_topics(self):
        topic_list = self.get_topic_names_and_types()
        for name, types in topic_list:
            m = self.pose_topic_pattern.match(name)
            if m and name not in [s.topic for s in self.subscriptions_]:
                tracker = m.group(1)
                self.get_logger().info(f'Subscribing to {name}')
                sub = self.create_subscription(
                    PoseStamped, name,
                    lambda msg, t=tracker: self.pose_cb(msg, t),
                    self.qos)
                sub.topic = name
                self.subscriptions_.append(sub)

    def pose_cb(self, msg: PoseStamped, tracker: str):
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = tracker
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.broadcaster.sendTransform(t)

def main():

    rclpy.init()
    rclpy.spin(VrpnTfBroadcaster())

if __name__ == '__main__':
    main()
