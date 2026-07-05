"""
cube_udp_bridge.py  (3단계 브리지측 — ROS 환경에서 실행)
──────────────────────────────────────────────────────────────
UDP로 좌표(x,y)를 받아 /cube_position (PointStamped) 으로 퍼블리시.

grounded_sam2_udp.py (conda) 가 보낸 UDP 패킷을 수신 →
기존 cube_detector 와 동일한 규약으로 /cube_position 퍼블리시.
node2_sim 은 기존처럼 /cube_position 을 구독 → 아무 변경 불필요.

  - z 는 node2_sim 에서 어차피 CUBE_Z_FIXED(0.035)로 덮어쓰므로
    여기서는 임시로 0.035 넣어 보냄 (기존 규약 유지).

실행 (ROS 환경 — rclpy 있는 곳):
  source /opt/ros/jazzy/setup.bash
  # (필요시) source ~/colcon_ws/install/setup.bash
  python3 cube_udp_bridge.py
"""
import socket
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

UDP_IP = "0.0.0.0"      # 모든 인터페이스에서 수신 (localhost 포함)
UDP_PORT = 5005
CUBE_Z_FIXED = 0.035    # 규약 유지용 (실제로는 node2_sim이 재설정)
RECV_TIMEOUT = 0.1      # 소켓 수신 타임아웃(초) — 타이머 주기와 맞물림


class CubeUDPBridge(Node):
    def __init__(self):
        super().__init__('cube_udp_bridge')

        self.pub = self.create_publisher(PointStamped, '/cube_position', 10)

        # ── UDP 소켓 (논블로킹에 가깝게 타임아웃) ──
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(RECV_TIMEOUT)

        self.latest_xy = None

        # ── 30Hz 로 수신+퍼블리시 시도 ──
        self.timer = self.create_timer(1.0 / 30.0, self.loop)
        self.get_logger().info(
            f'cube_udp_bridge 시작 — UDP {UDP_PORT} 수신 → /cube_position 퍼블리시')

    def loop(self):
        # UDP 수신 (가장 최신 패킷만 쓰도록 버퍼 비우기)
        data = None
        while True:
            try:
                packet, _ = self.sock.recvfrom(1024)
                data = packet          # 계속 읽어서 마지막 것만 사용
            except socket.timeout:
                break
            except BlockingIOError:
                break

        if data is None:
            return  # 이번 주기엔 새 데이터 없음

        try:
            x_str, y_str = data.decode().split(',')
            x_user = float(x_str)
            y_user = float(y_str)
        except Exception:
            self.get_logger().warn(f'파싱 실패: {data!r}',
                                   throttle_duration_sec=2.0)
            return

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x = x_user
        msg.point.y = y_user
        msg.point.z = CUBE_Z_FIXED
        self.pub.publish(msg)

        self.get_logger().info(
            f'cube  x={x_user:+.4f}  y={y_user:+.4f}',
            throttle_duration_sec=1.0)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CubeUDPBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
