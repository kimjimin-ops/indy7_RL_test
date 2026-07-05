"""
grounded_sam2_udp.py  (3단계 검출측 — conda env_isaaclab 에서 실행)
────────────────────────────────────────────────────────────────────
자연어 → GroundingDINO → SAM2 → 마스크 중심 → base 기준 x,y 좌표
→ UDP로 좌표 송신 (127.0.0.1:5005).

ROS 없음. rclpy 불필요. 좌표를 UDP 패킷으로만 내보냄.
같은 PC에서 도는 cube_udp_bridge.py (ROS 환경) 가 이 패킷을 받아
/cube_position 으로 퍼블리시함.

실행:
  conda activate env_isaaclab
  python3 grounded_sam2_udp.py

키: q = 종료 (창 포커스 상태에서)
"""
import sys
import socket
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import threading

# ── 검출 대상 (자연어) ──
TEXT_PROMPT = "red cube."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# ── 자연어 입력 스레드 ──
prompt_lock = threading.Lock()

def input_thread():
    global TEXT_PROMPT
    while True:
        try:
            new_prompt = input()   # 터미널에서 입력 대기
        except EOFError:
            break
        new_prompt = new_prompt.strip()
        if new_prompt:
            # GroundingDINO 규약: 소문자 + 마침표로 끝
            if not new_prompt.endswith('.'):
                new_prompt += '.'
            with prompt_lock:
                TEXT_PROMPT = new_prompt.lower()
            print(f"🎯 대상 변경 → '{TEXT_PROMPT}'")

# ── 좌표 오프셋 (기존 cube_detector와 동일) ──
X_OFFSET = 0.94
Y_OFFSET = 0.27
DEPTH_WIN = 2

# ── UDP 송신 설정 ──
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ============================================================
# GroundingDINO
# ============================================================
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
print("🔄 GroundingDINO 로딩...")
gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to("cuda")
print("✅ GroundingDINO 로딩 완료")

# ============================================================
# SAM2
# ============================================================
SAM2_BASE_PATH = "/home/kimjimin/sam2"
sys.path.append(SAM2_BASE_PATH)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CKPT_PATH = SAM2_BASE_PATH + "/checkpoints/sam2.1_hiera_tiny.pt"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"

print("🔄 SAM2 로딩...")
sam2_model = build_sam2(CONFIG_NAME, CKPT_PATH, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("✅ SAM2 로딩 완료")

# ============================================================
# RealSense (color + depth)
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = (profile.get_stream(rs.stream.color)
        .as_video_stream_profile().get_intrinsics())
print(f"🎥 RealSense 시작 — 대상: '{TEXT_PROMPT}'  UDP→{UDP_IP}:{UDP_PORT} (q로 종료)")
threading.Thread(target=input_thread, daemon=True).start()
print("⌨️  검출할 물체를 입력하고 Enter (예: blue bottle)")


def median_depth(depth_frame, u, v, w, h):
    vals = []
    for dy in range(-DEPTH_WIN, DEPTH_WIN + 1):
        for dx in range(-DEPTH_WIN, DEPTH_WIN + 1):
            xx, yy = u + dx, v + dy
            if 0 <= xx < w and 0 <= yy < h:
                z = depth_frame.get_distance(xx, yy)
                if z > 0:
                    vals.append(z)
    return float(np.median(vals)) if vals else 0.0


def detect_box(image_rgb):
    with prompt_lock:
        current_prompt = TEXT_PROMPT
    inputs = gdino_processor(images=image_rgb, text=current_prompt,
                             return_tensors="pt").to("cuda")    
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    results = gdino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_rgb.shape[:2]],
    )
    boxes = results[0]["boxes"]
    if len(boxes) == 0:
        return None
    scores = results[0]["scores"]
    best = int(torch.argmax(scores))
    return boxes[best].cpu().numpy()


try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        cframe = frames.get_color_frame()
        dframe = frames.get_depth_frame()
        if not cframe or not dframe:
            continue

        image_bgr = np.asanyarray(cframe.get_data())
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        vis = image_bgr.copy()

        box = detect_box(image_rgb)
        if box is not None:
            x0, y0, x1, y1 = box.astype(int)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam2_predictor.set_image(image_rgb)
                masks, _, _ = sam2_predictor.predict(
                    box=box[None, :], multimask_output=False)
            mask = masks[0].astype(bool)
            if mask.ndim == 3:
                mask = mask[0]

            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                z = median_depth(dframe, cx, cy, w, h)
                if z > 0:
                    X_cam, Y_cam, Z_cam = rs.rs2_deproject_pixel_to_point(
                        intr, [cx, cy], z)
                    x_user = Y_cam + X_OFFSET
                    y_user = X_cam + Y_OFFSET

                    # ── UDP 송신: "x,y" 문자열 ──
                    msg = f"{x_user:.4f},{y_user:.4f}"
                    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

                    # 시각화
                    overlay = vis.copy()
                    overlay[mask] = (0, 255, 0)
                    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(vis,
                                f'x={x_user:+.3f} y={y_user:+.3f} z={Z_cam:.3f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

        cv2.imshow("Grounded-SAM2 UDP", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
