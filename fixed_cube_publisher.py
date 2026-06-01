#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixed_cube_publisher.py
=======================
검증 단계용: 고정된 큐브 위치를 /cube_pose 토픽으로 계속 발행한다.
좌표는 로봇 base 프레임 기준 (학습 시 cube_position_w = cube - robot_base 와 동일 정의).

학습 시 큐브 스폰 범위(reset_cube_position): x 0.1~0.2, y 0.1~0.2, z 0.025
-> 그 범위 안의 한 점을 기본값으로 둔다. 필요하면 CUBE_XYZ만 바꿔 테스트.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# base 프레임 기준 고정 큐브 위치 (학습 스폰 범위 안의 한 점)
CUBE_XYZ = (0.5, 0, 0.025)
BASE_FRAME = "link0"          # 추론 노드의 BASE_LINK와 일치 (Gazebo TF 기준 로봇 베이스)
CUBE_TOPIC = "/cube_pose"
PUBLISH_HZ = 30.0


class FixedCubePublisher(Node):
    def __init__(self):
        super().__init__("fixed_cube_publisher")
        self.pub = self.create_publisher(PoseStamped, CUBE_TOPIC, 10)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_pose)
        self.get_logger().info(
            f"Publishing fixed cube at {CUBE_XYZ} (frame={BASE_FRAME}) on {CUBE_TOPIC}"
        )

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x = float(CUBE_XYZ[0])
        msg.pose.position.y = float(CUBE_XYZ[1])
        msg.pose.position.z = float(CUBE_XYZ[2])
        msg.pose.orientation.w = 1.0   # 축 정렬 보상은 큐브 회전을 보지만,
        msg.pose.orientation.x = 0.0   # 추론 노드는 cube 위치(xyz)만 관측에 쓰므로
        msg.pose.orientation.y = 0.0   # 회전은 단위 쿼터니언으로 둠.
        msg.pose.orientation.z = 0.0
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FixedCubePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
