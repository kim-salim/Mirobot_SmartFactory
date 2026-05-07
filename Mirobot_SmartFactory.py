import os
import cv2
import numpy as np
import torch
import math
import time
import serial
from typing import Tuple, List
from collections import deque
from serial.tools import list_ports

# ===== 타입/상수 정의 =====
Vec3 = Tuple[float, float, float]

EOL   = "\r\n"
BAUDS = [115200, 38400]
LAST_POS: Vec3 = None  # phase2에서 사용할 마지막 위치

# ===== 기본 파일 위치 / YOLO / 보드 포즈 =====
DEFAULT_FOLDER = r"C:\Users\wjddb\Vision_robot"
STATIC_BOARD_PATH = os.path.join(DEFAULT_FOLDER, "static_board_pose.npz")

YOLO_REPO = r"C:\Users\wjddb\Downloads\yolov5-master\yolov5-master"
YOLO_WEIGHT = r"C:\Users\wjddb\Downloads\yolov5-master\yolov5-master\runs\train\exp9\weights\best.pt"

# 위에서 내려다보는 카메라(1번)용 임계값
CONF_THRES_TOP = 0.5
# 옆 카메라(2번)용 임계값
CONF_THRES_SIDE = 0.25
IOU_THRES = 0.45

# ===== Z 관련 설정 =====
Z_PICK = 55.0          # 실제로 물체를 집거나 놓을 Z
Z_OFFSET = 50.0        # 이동할 때 얼마만큼 더 위로 띄울지
Z_TRAVEL = Z_PICK + Z_OFFSET  # 이동 높이 = 105mm

# ===== 구조/적층 관련 (phase2) =====
CUBE_SIZE = 20.0      # mm, 큐브 한 변 길이
STACK_START = np.array([210.0, 0.0, 60.0])  # 구조를 쌓을 기준점

# 사용할 클래스
USE_CLASSES = {"red-cube", "green-cube", "blue-cube"}

# 색상별 공급(픽업) 위치 (mm)
COLOR_PICK_POS = {
    "red-cube":   (140.0, -90.0, Z_PICK),
    "green-cube": (140.0,   0.0, Z_PICK),
    "blue-cube":  (140.0,  90.0, Z_PICK),
}

# ==========================================
# 공통: YOLO 로드
# ==========================================
def load_yolo():
    model = torch.hub.load(YOLO_REPO, 'custom', path=YOLO_WEIGHT, source='local')
    model.iou = IOU_THRES
    return model

# ==========================================
# 공통: 시리얼/로봇 제어
# ==========================================
def ports():
    plist = list(list_ports.comports())
    pri = [p for p in plist if ("CH340" in (p.description or "") or "USB-SERIAL" in (p.description or ""))]
    return pri or plist

def open_try(port, baud):
    try:
        ser = serial.Serial(
            port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2,
            write_timeout=1.5,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        try:
            ser.setDTR(False); ser.setRTS(False); time.sleep(0.05)
            ser.setDTR(True);  ser.setRTS(True)
        except Exception:
            pass

        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)
        return ser
    except Exception as e:
        print(f"[open_fail] {port}@{baud}: {e}")
        return None

def rx_all(ser, delay=0.05):
    time.sleep(delay)
    n = ser.in_waiting
    return ser.read(n).decode(errors="ignore") if n else ""

def tx(ser, cmd):
    print("TX:", cmd)
    ser.write((cmd + EOL).encode())
    ser.flush()

def wait_ok(ser, timeout=3.0):
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        buf += rx_all(ser, delay=0.02)
        low = buf.lower()
        if "\nok" in ("\n" + low) or low.endswith("ok"):
            return True, buf
        if "error" in low or "alarm" in low or "lock" in low:
            print("RX(ERR):", buf.strip())
            return False, buf
    print("RX(TIMEOUT):", buf.strip())
    return False, buf

def pump_on(ser, pwm: int = 1000):
    cmd = f"M3S{int(pwm)}"
    tx(ser, cmd)
    ok, resp = wait_ok(ser, 2.0)
    if not ok:
        print("[WARN] pump_on 응답:", resp.strip())
    else:
        print("✅ Pump ON (", cmd, ")")

def pump_off(ser):
    cmd = "M3S0"
    tx(ser, cmd)
    ok, resp = wait_ok(ser, 2.0)
    if not ok:
        print("[WARN] pump_off 응답:", resp.strip())
    else:
        print("✅ Pump OFF")

# ==========================================
# 베지어 궤적 (phase1: 정리용, 낮은 z_min)
# ==========================================
def make_bezier_arc_xy_z_sort(
    p_start: Vec3,
    p_end: Vec3,
    *,
    h_min: float = 40.0,
    k: float = 0.3,
    z_min: float = 40.0,
    z_max: float = 300.0,
    margin: float = 10.0,
    n_points: int = 50
) -> List[Vec3]:
    x1, y1, z1 = p_start
    x2, y2, z2 = p_end

    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)

    h = max(h_min, k * d)

    z_mid_raw = max(z1, z2) + h
    z_mid_clamped_high = min(z_mid_raw, z_max - margin)
    z_mid = max(z_mid_clamped_high, z_min + margin)

    xm = (x1 + x2) / 2.0
    ym = (y1 + y2) / 2.0

    P0 = (x1, y1, z1)
    P1 = (xm, ym, z_mid)
    P2 = (x2, y2, z2)

    path: List[Vec3] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        s = 1.0 - t

        x = s * s * P0[0] + 2.0 * s * t * P1[0] + t * t * P2[0]
        y = s * s * P0[1] + 2.0 * s * t * P1[1] + t * t * P2[1]
        z = s * s * P0[2] + 2.0 * s * t * P1[2] + t * t * P2[2]

        path.append((x, y, z))

    return path

# ==========================================
# 베지어 궤적 (phase2: 구조 적층용, z_min=200)
# ==========================================
def make_bezier_arc_xy_z_build(
    p_start: Vec3,
    p_end: Vec3,
    *,
    h_min: float = 40.0,
    k: float = 0.3,
    z_min: float = 200.0,
    z_max: float = 300.0,
    margin: float = 10.0,
    n_points: int = 50
) -> List[Vec3]:
    x1, y1, z1 = p_start
    x2, y2, z2 = p_end

    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)

    h = max(h_min, k * d)
    z_mid_raw = max(z1, z2) + h

    z_mid_clamped_high = min(z_mid_raw, z_max - margin)
    z_mid = max(z_mid_clamped_high, z_min + margin)

    xm = (x1 + x2) / 2.0
    ym = (y1 + y2) / 2.0

    P0 = (x1, y1, z1)
    P1 = (xm, ym, z_mid)
    P2 = (x2, y2, z2)

    path: List[Vec3] = []

    for i in range(n_points):
        t = i / (n_points - 1)
        s = 1.0 - t

        x = s * s * P0[0] + 2.0 * s * t * P1[0] + t * t * P2[0]
        y = s * s * P0[1] + 2.0 * s * t * P1[1] + t * t * P2[1]
        z = s * s * P0[2] + 2.0 * s * t * P1[2] + t * t * P2[2]

        path.append((x, y, z))

    return path

# ==========================================
# 픽업/드랍 시퀀스 (phase1: 정리용)
# ==========================================
def move_with_pump_between_points_sort(
    p_pick: Vec3,
    p_place: Vec3,
    *,
    feed: float = 2000.0,
    n_points: int = 50
) -> bool:
    """
    p_pick: (x1, y1, z1)
    p_place: (x2, y2, z2)

    시퀀스:
      1) (x1, y1, Z_TRAVEL)로 이동
      2) 수직 하강 → (x1, y1, Z_PICK)
      3) 펌프 ON + 1.5초 대기
      4) 수직 상승 → (x1, y1, Z_TRAVEL)
      5) 베지어 궤적으로 (x2, y2, z2+Z_OFFSET)까지 이동
      6) 수직 하강 → (x2, y2, z2)
      7) 0.5초 대기 후 펌프 OFF
      8) 수직 상승 → (x2, y2, z2+Z_OFFSET)
    """
    x1, y1, _ = p_pick
    x2, y2, z2 = p_place

    p_top_start: Vec3 = (x1, y1, Z_TRAVEL)
    z_travel_place = z2 + Z_OFFSET
    p_top_end:   Vec3 = (x2, y2, z_travel_place)

    path_top = make_bezier_arc_xy_z_sort(p_top_start, p_top_end, n_points=n_points)

    for p in ports():
        for b in BAUDS:
            print(f"\n=== Connect for Pick&Place: {p.device} @ {b} ===")
            ser = open_try(p.device, b)
            if not ser:
                continue
            try:
                rx_all(ser, 0.2)

                for c in ("M21", "M20", "G90"):
                    tx(ser, c)
                    _ = wait_ok(ser, 1.2)

                tx(ser, "M50")
                _ = wait_ok(ser, 2.0)

                # 1) 픽업 포인트 위(Z_TRAVEL)로 이동
                cmd = f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{p_top_start[2]:.3f} F{feed:.1f}"
                tx(ser, cmd)
                ok, resp = wait_ok(ser, 10.0)
                if not ok:
                    print("[FAIL] 픽업 상단 이동 중 에러:", resp.strip())
                    return False

                # 2) 수직 하강 → Z_PICK
                cmd = f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{Z_PICK:.3f} F{feed:.1f}"
                tx(ser, cmd)
                ok, resp = wait_ok(ser, 10.0)
                if not ok:
                    print("[FAIL] 픽업 하강 중 에러:", resp.strip())
                    return False

                # 3) 펌프 ON + 1.5초 대기
                pump_on(ser)
                time.sleep(1.5)

                # 4) 수직 상승 → Z_TRAVEL
                cmd = f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{Z_TRAVEL:.3f} F{feed:.1f}"
                tx(ser, cmd)
                ok, resp = wait_ok(ser, 10.0)
                if not ok:
                    print("[FAIL] 픽업 상승 중 에러:", resp.strip())
                    return False

                # 5) 베지어 궤적으로 도착 상단까지 이동
                for (x, y, z) in path_top[1:]:
                    cmd = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}"
                    tx(ser, cmd)
                    ok, resp = wait_ok(ser, 8.0)
                    if not ok:
                        print("[FAIL] 베지어 이동 중 에러:", resp.strip())
                        return False
                    time.sleep(0.01)

                # 6) 수직 하강 → z2 (드랍 위치)
                cmd = f"G1 X{p_top_end[0]:.3f} Y{p_top_end[1]:.3f} Z{z2:.3f} F{feed:.1f}"
                tx(ser, cmd)
                ok, resp = wait_ok(ser, 10.0)
                if not ok:
                    print("[FAIL] 드랍 하강 중 에러:", resp.strip())
                    return False

                # 7) 잠깐 대기 후 펌프 OFF
                time.sleep(0.5)
                pump_off(ser)

                # 8) 수직 상승 → 다시 z2+Z_OFFSET
                cmd = f"G1 X{p_top_end[0]:.3f} Y{p_top_end[1]:.3f} Z{z_travel_place:.3f} F{feed:.1f}"
                tx(ser, cmd)
                ok, resp = wait_ok(ser, 10.0)
                if not ok:
                    print("[FAIL] 드랍 상승 중 에러:", resp.strip())
                    return False

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                print("✅ 픽업 + 이동 + 드랍 시퀀스 완료")
                ser.close()
                return True

            except Exception as e:
                print("[pick_place_fail]", e)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass

    print("✗ 연결 실패: 포트/보드레이트 조합이 모두 실패했습니다.")
    return False

# ==========================================
# 픽업/드랍 시퀀스 (phase2: 구조 적층용)
# ==========================================
def move_with_pump_between_points_build(
    p_start: Vec3,   # 픽업 위치 (x, y, z_pick)
    p_end: Vec3,     # 놓는 위치 (x, y, z_place)
    *,
    feed: float = 2000.0,
    n_points: int = 50
) -> bool:
    global LAST_POS

    x_pick, y_pick, z_pick = p_start
    x_place, y_place, z_place = p_end

    pick_lift_z = min(z_pick + Z_OFFSET, 300.0)
    place_lift_z = min(z_place + Z_OFFSET, 300.0)

    pick_lift = (x_pick,  y_pick,  pick_lift_z)
    place_lift = (x_place, y_place, place_lift_z)

    for p in ports():
        for b in BAUDS:
            print(f"\n=== Connect for Bezier+Pump move: {p.device} @ {b} ===")
            ser = open_try(p.device, b)
            if not ser:
                continue

            try:
                rx_all(ser, 0.2)

                for c in ("M21", "M20", "G90"):
                    tx(ser, c)
                    _ = wait_ok(ser, 1.2)

                tx(ser, "M50")
                _ = wait_ok(ser, 2.0)

                # 1) 이전 위치 -> 픽업 위치 위 (pick_lift) : 베지어 또는 직선
                if LAST_POS is None:
                    print("[INFO] 첫 작업 → 직선으로 픽업 위치 위로 이동")
                    cmd_start = f"G1 X{pick_lift[0]:.3f} Y{pick_lift[1]:.3f} Z{pick_lift[2]:.3f} F{feed:.1f}"
                    tx(ser, cmd_start)
                    ok, resp = wait_ok(ser, 10.0)
                    if not ok:
                        print("[FAIL] 첫 픽업 위치 위 이동 중 에러:", resp.strip())
                        return False
                else:
                    print("[INFO] 이전 위치 → 픽업 위치 위로 베지어 이동")
                    path_to_pick = make_bezier_arc_xy_z_build(LAST_POS, pick_lift, n_points=n_points)
                    for (x, y, z) in path_to_pick:
                        cmd = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}"
                        tx(ser, cmd)
                        ok, resp = wait_ok(ser, 8.0)
                        if not ok:
                            print("[FAIL] 픽업 위치로 가는 베지어 중 에러:", resp.strip())
                            return False
                        time.sleep(0.01)

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                # 2) 수직으로 내려가서 집기, 펌프 ON, 다시 위로
                print("[INFO] 픽업 위치로 수직 하강")
                cmd_down_pick = f"G1 X{x_pick:.3f} Y{y_pick:.3f} Z{z_pick:.3f} F{feed:.1f}"
                tx(ser, cmd_down_pick)
                ok, resp = wait_ok(ser, 8.0)
                if not ok:
                    print("[FAIL] 픽업 위치 수직 이동 중 에러:", resp.strip())
                    return False

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                pump_on(ser)
                time.sleep(1.5)

                print("[INFO] 픽업 후 위로 상승")
                cmd_up_pick = f"G1 X{x_pick:.3f} Y{y_pick:.3f} Z{pick_lift_z:.3f} F{feed:.1f}"
                tx(ser, cmd_up_pick)
                ok, resp = wait_ok(ser, 8.0)
                if not ok:
                    print("[FAIL] 픽업 후 상승 중 에러:", resp.strip())
                    return False

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                # 3) 픽업 위 → 놓는 위치 위 : 베지어
                print("[INFO] 픽업 위치 위 → 놓는 위치 위로 베지어 이동")
                path_pick_to_place = make_bezier_arc_xy_z_build(pick_lift, place_lift, n_points=n_points)
                for (x, y, z) in path_pick_to_place:
                    cmd = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}"
                    tx(ser, cmd)
                    ok, resp = wait_ok(ser, 8.0)
                    if not ok:
                        print("[FAIL] 놓는 위치로 가는 베지어 중 에러:", resp.strip())
                        return False
                    time.sleep(0.01)

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                # 4) 수직으로 내려서 놓기, 펌프 OFF, 다시 위로
                print("[INFO] 놓는 위치로 수직 하강")
                cmd_down_place = f"G1 X{x_place:.3f} Y{y_place:.3f} Z{z_place:.3f} F{feed:.1f}"
                tx(ser, cmd_down_place)
                ok, resp = wait_ok(ser, 8.0)
                if not ok:
                    print("[FAIL] 놓는 위치 수직 이동 중 에러:", resp.strip())
                    return False

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                time.sleep(0.5)
                pump_off(ser)

                print("[INFO] 놓은 후 위로 상승")
                cmd_up_place = f"G1 X{x_place:.3f} Y{y_place:.3f} Z{place_lift_z:.3f} F{feed:.1f}"
                tx(ser, cmd_up_place)
                ok, resp = wait_ok(ser, 8.0)
                if not ok:
                    print("[WARN] 놓은 후 상승 중 에러:", resp.strip())

                try:
                    tx(ser, "M400")
                    _ = wait_ok(ser, 10.0)
                except Exception:
                    pass

                LAST_POS = place_lift
                print("✅ 한 큐브 작업(잡으러 가기+놓으러 가기) 완료, LAST_POS =", LAST_POS)

                ser.close()
                return True

            except Exception as e:
                print("[bezier_pump_fail]", e)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass

    print("✗ 연결 실패: 포트/보드레이트 조합이 모두 실패했습니다.")
    return False

# ==========================================
# phase2: 옆 카메라용 YOLO 감지/구조 인식
# ==========================================
def detect_yolo_objects(frame_bgr, model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    df = results.pandas().xyxy[0]

    if len(df) > 0:
        print("\n[DEBUG] Raw YOLO detections:")
        print(df[["name", "confidence"]])
    else:
        print("\n[DEBUG] YOLO: 이 프레임에서 아무것도 못 찾음")

    detections = []
    for _, row in df.iterrows():
        conf = float(row["confidence"])
        if conf < CONF_THRES_SIDE:
            continue

        cls = row["name"]
        xmin, ymin = float(row["xmin"]), float(row["ymin"])
        xmax, ymax = float(row["xmax"]), float(row["ymax"])

        detections.append({
            "cls": cls,
            "bbox": (xmin, ymin, xmax, ymax),
        })

    return detections

def extract_structure_relations(detections):
    X_THRESH, Y_THRESH = 25.0, 25.0

    centers = {}
    for det in detections:
        cls = det["cls"]
        if cls not in USE_CLASSES:
            continue
        xmin, ymin, xmax, ymax = det["bbox"]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        centers[cls] = (cx, cy)

    labels = list(centers.keys())
    relations = set()

    # 가로 방향: 각 큐브에서 "가장 가까운 오른쪽"만 연결
    for A in labels:
        xA, yA = centers[A]
        best_B = None
        best_dx = None
        for B in labels:
            if B == A:
                continue
            xB, yB = centers[B]
            if abs(yA - yB) > Y_THRESH:
                continue
            dx = xB - xA
            if dx <= 0:
                continue
            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_B = B
        if best_B is not None:
            relations.add(f"{A}-right-{best_B}")

    # 세로 방향: 각 큐브에서 "바로 아래"만 연결
    for A in labels:
        xA, yA = centers[A]
        best_B = None
        best_dy = None
        for B in labels:
            if B == A:
                continue
            xB, yB = centers[B]
            if abs(xA - xB) > X_THRESH:
                continue
            dy = yB - yA
            if dy <= 0:
                continue
            if best_dy is None or dy < best_dy:
                best_dy = dy
                best_B = B
        if best_B is not None:
            relations.add(f"{A}-top-{best_B}")

    return list(relations), centers

def compute_positions(relations):
    if not relations:
        return {}

    nodes = set()
    edges = []
    top_nodes = set()

    for rel in relations:
        if "-top-" in rel:
            A, B = rel.split("-top-")
            direction = "top"
            top_nodes.add(A)
        elif "-right-" in rel:
            A, B = rel.split("-right-")
            direction = "right"
        elif "-left-" in rel:
            A, B = rel.split("-left-")
            direction = "left"
        elif "-bottom-" in rel:
            A, B = rel.split("-bottom-")
            direction = "bottom"
        else:
            continue

        nodes.add(A)
        nodes.add(B)
        edges.append((A, direction, B))

    if not nodes:
        return {}

    graph = {n: [] for n in nodes}
    for A, direction, B in edges:
        if direction == "top":
            graph[A].append(("bottom", B))
            graph[B].append(("top", A))
        elif direction == "bottom":
            graph[A].append(("top", B))
            graph[B].append(("bottom", A))
        elif direction == "right":
            graph[A].append(("right", B))
            graph[B].append(("left", A))
        elif direction == "left":
            graph[A].append(("left", B))
            graph[B].append(("right", A))

    bottom_candidates = nodes - top_nodes
    if bottom_candidates:
        root = sorted(bottom_candidates)[0]
    else:
        root = sorted(nodes)[0]

    positions = {}
    positions[root] = STACK_START.copy()

    q = deque([root])

    while q:
        cur = q.popleft()
        cur_pos = positions[cur]

        for direction, nb in graph[cur]:
            if nb in positions:
                continue

            if direction == "top":
                offset = np.array([0.0, 0.0, CUBE_SIZE])
            elif direction == "bottom":
                offset = np.array([0.0, 0.0, -CUBE_SIZE])
            elif direction == "right":
                offset = np.array([CUBE_SIZE, 0.0, 0.0])
            elif direction == "left":
                offset = np.array([-CUBE_SIZE, 0.0, 0.0])
            else:
                continue

            positions[nb] = cur_pos + offset
            q.append(nb)

    return positions

def print_all_cube_positions(centers, positions):
    print("\n===== ALL CUBES =====")
    if not centers and not positions:
        print("  (no cubes)")
        print("=====================\n")
        return

    all_names = set(centers.keys()) | set(positions.keys())
    for name in sorted(all_names):
        img_pos = centers.get(name, None)
        world_pos = positions.get(name, None)

        line = f"  {name}: "
        if img_pos is not None:
            line += f"img_center=(u={img_pos[0]:.1f}, v={img_pos[1]:.1f}) "
        else:
            line += "img_center=(none) "

        if world_pos is not None:
            line += f"world_pos=({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})"
        else:
            line += "world_pos=(not connected)"

        print(line)
    print("=====================\n")

def draw_detections(frame_bgr, detections):
    show = frame_bgr.copy()
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        cls = det["cls"]
        cv2.rectangle(show,
                      (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)),
                      (0, 255, 0), 2)
        cv2.putText(show, cls,
                    (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    return show

def build_with_robot(positions: dict):
    tasks = []

    for name, pos in positions.items():
        if name not in COLOR_PICK_POS:
            print(f"[WARN] '{name}' 픽업 위치 미정 → 건너뜀")
            continue

        src = COLOR_PICK_POS[name]
        dst = (float(pos[0]), float(pos[1]), float(pos[2]))
        tasks.append((name, src, dst))

    if not tasks:
        print("[INFO] 실행할 이동 작업이 없습니다.")
        return

    tasks.sort(key=lambda t: t[2][2])  # z 낮은 것부터

    for name, src, dst in tasks:
        print(f"[MOVE] {name}: {src} -> {dst}")
        ok = move_with_pump_between_points_build(src, dst, feed=2000.0, n_points=50)
        if not ok:
            print(f"[ERROR] {name} 이동 실패, 이후 작업 중단")
            break

# ==========================================
# phase1: 1번 웹캠으로 색깔별 정리
#   - w: YOLO 감지
#   - e: 가까운 순으로 집어서 색상별 위치에 정리
#   - q: 전체 프로그램 종료
# ==========================================
def run_phase1_sort(model, H_inv, T_base_board) -> bool:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] 1번 카메라를 열 수 없습니다.")
        return False

    model.conf = CONF_THRES_TOP
    detections = []
    need_detect = False

    print("[PHASE1] 시작: q=종료, w=YOLO 감지, e=자동 픽업/정리")

    phase1_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        show = frame.copy()

        if need_detect:
            results = model(frame_rgb)
            df = results.pandas().xyxy[0]

            detections = []

            for _, row in df.iterrows():
                conf = float(row["confidence"])
                if conf < CONF_THRES_TOP:
                    continue

                cls_name = row["name"]
                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                u = (xmin + xmax) * 0.5
                v = (ymin + ymax) * 0.5

                pt_img = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
                pt_board_h = H_inv @ pt_img
                pt_board_h /= pt_board_h[2, 0]
                bx = pt_board_h[0, 0]
                by = pt_board_h[1, 0]
                bz = 0.0

                P_board = np.array([bx, by, bz, 1.0], dtype=np.float64)
                P_base = T_base_board @ P_board
                Xb, Yb, Zb = P_base[:3]

                Xr = round(float(Xb))
                Yr = round(float(Yb))
                Zr = Z_PICK

                detections.append({
                    "cls": cls_name,
                    "conf": conf,
                    "X": float(Xb),
                    "Y": float(Yb),
                    "Z": float(Zb),
                    "Xr": Xr,
                    "Yr": Yr,
                    "Zr": Zr,
                    "bbox": (xmin, ymin, xmax, ymax)
                })

            if detections:
                print("\n[DETECT] 감지된 물체:")
                for i, det in enumerate(detections):
                    dist_xy = math.hypot(det["Xr"], det["Yr"])
                    print(
                        f"  [{i}] {det['cls']}  "
                        f"base(mm) 실수: X={det['X']:.2f}, Y={det['Y']:.2f}, Z={det['Z']:.2f}  "
                        f"반올림+Z고정: X={det['Xr']}, Y={det['Yr']}, Z={det['Zr']}  "
                        f"conf={det['conf']:.2f}  dist={dist_xy:.1f}mm"
                    )
                print("→ e 키를 누르면 색/거리 기준으로 자동 픽업·정리합니다.")
            else:
                print("[DETECT] 조건을 만족하는 물체 없음")

            need_detect = False

        for det in detections:
            xmin, ymin, xmax, ymax = det["bbox"]
            cv2.rectangle(show,
                          (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)),
                          (255, 255, 255), 2)
            cv2.putText(show, f"{det['cls']} {det['conf']:.2f}",
                        (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        cv2.imshow("phase1_topcam_sort", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False  # 전체 종료

        elif key == ord('w'):
            print("[PHASE1] 현재 프레임으로 YOLO 감지 시도...")
            need_detect = True

        elif key == ord('e'):
            if not detections:
                print("[PHASE1] 아직 감지 결과 없음. 먼저 w 키로 감지하세요.")
            else:
                drop_xy_map = {
                    "red-cube": (140.0, -90.0),
                    "green-cube": (140.0, 0.0),
                    "blue-cube": (140.0, 90.0),
                }

                stack_counts = {}
                ref_x, ref_y = 0.0, 0.0
                remaining = detections.copy()
                total = len(remaining)
                print(f"\n[PHASE1] 총 {total}개 물체를 순차적으로 처리합니다.")

                move_idx = 1

                while remaining:
                    best_i = None
                    best_d = float("inf")
                    for i, det in enumerate(remaining):
                        dx = det["Xr"] - ref_x
                        dy = det["Yr"] - ref_y
                        d = math.hypot(dx, dy)
                        if d < best_d:
                            best_d = d
                            best_i = i

                    target = remaining.pop(best_i)
                    cls_name = target["cls"]

                    if cls_name not in drop_xy_map:
                        print(f"[WARN] 지원하지 않는 클래스 '{cls_name}' → 스킵")
                        continue

                    base_x, base_y = drop_xy_map[cls_name]
                    count_same = stack_counts.get(cls_name, 0)

                    STACK_DZ = 28.0
                    z_place = Z_PICK + STACK_DZ * count_same
                    stack_counts[cls_name] = count_same + 1

                    print(
                        f"\n[MOVE] {move_idx}/{total}번째 물체: {cls_name} "
                        f"(픽업: Xr={target['Xr']}, Yr={target['Yr']} → "
                        f"드랍: ({base_x}, {base_y}, {z_place}), "
                        f"기준점에서 거리={best_d:.1f}mm)"
                    )

                    x1 = target["Xr"]
                    y1 = target["Yr"] + 15  # 기존 Y 보정 유지
                    p_pick = (x1, y1, Z_PICK)
                    p_place = (base_x, base_y, z_place)

                    ok = move_with_pump_between_points_sort(
                        p_pick,
                        p_place,
                        feed=2000.0,
                        n_points=50
                    )

                    if not ok:
                        print("[WARN] 이동 중 에러 발생 → 나머지 물체 중단")
                        break

                    ref_x, ref_y = base_x, base_y
                    move_idx += 1

                detections = []
                phase1_done = True
                print("[PHASE1] 정리 완료. 이제 2번 카메라로 구조 인식 단계로 넘어갑니다.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return phase1_done

# ==========================================
# phase2: 2번 웹캠으로 구조 인식 후 모양대로 적층
#   - s: YOLO 감지
#   - d: 마지막 감지 결과로 로봇 동작
#   - q: 종료
# ==========================================
def run_phase2_build(model):
    global LAST_POS
    LAST_POS = None  # 새 구조 쌓기 시작할 때 초기화

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("[ERROR] 2번 카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detections = []
    relations = []
    positions = {}
    centers = {}
    need_detect = False

    model.conf = CONF_THRES_SIDE

    print("[PHASE2] q: 종료, s: YOLO 감지, d: 마지막 감지 결과로 로봇 동작")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if need_detect:
            print("[PHASE2] YOLO 감지 실행")
            detections = detect_yolo_objects(frame, model)
            print(f"[INFO] 감지된 개수(필터 통과 후): {len(detections)}")

            relations, centers = extract_structure_relations(detections)
            print("Relations:", relations)

            positions = compute_positions(relations)
            print("Positions dict(keys):", positions.keys())

            print_all_cube_positions(centers, positions)

            need_detect = False

        if detections:
            show = draw_detections(frame, detections)
        else:
            show = frame

        cv2.imshow("phase2_sidecam_structure_detect", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            print("[PHASE2] YOLO 감지 요청 (s)")
            need_detect = True
        elif key == ord("d"):
            if positions:
                print("[PHASE2] d 입력: 마지막 구조로 로봇 동작 시작")
                build_with_robot(positions)
                print("[PHASE2] 로봇 동작 완료, 계속해서 s로 재감지 가능")
            else:
                print("[PHASE2] 아직 유효한 감지 결과가 없습니다. 먼저 s로 감지하세요.")

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 메인: phase1 → phase2 순서로 실행
# ==========================================
def main():
    if not os.path.exists(STATIC_BOARD_PATH):
        print("[ERROR] static_board_pose.npz 가 없습니다.")
        print("        먼저 save_static_board_pose.py 를 실행해서 생성하세요.")
        return

    pose = np.load(STATIC_BOARD_PATH)
    H_inv = pose["H_inv"]
    T_base_board = pose["T_base_board"]

    print("[INFO] YOLO 모델 로드 중...")
    model = load_yolo()
    print("[INFO] YOLO 로드 완료.")

    # 1단계: 위에서 내려다보는 카메라로 색깔별 정리
    phase1_ok = run_phase1_sort(model, H_inv, T_base_board)
    if not phase1_ok:
        print("[MAIN] phase1 이 완료되지 않아 프로그램을 종료합니다.")
        return

    # 2단계: 옆 카메라로 구조 인식 후 모양대로 적층
    run_phase2_build(model)

if __name__ == "__main__":
    main()
