import threading
import time as time_module

import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify

import camera
from camera import start_camera, get_color_and_depth
from face_db import load_known_faces, find_match
from attendance import mark_attendance, get_today_status, get_today_records
from jp_text import put_text, text_size
from face_model import create_face_app

app = Flask(__name__)

# ---------------- 設定 ----------------
RECOGNIZE_EVERY_N_FRAMES = 3   # Nフレームに1回だけ顔認識用に画像を渡す
CONFIRM_COUNT = 4              # 出席を記録するために必要な連続認識回数

# 深度による「生体顔」判定（RealSense使用時のみ有効）。
# 写真や画面によるなりすまし対策です。まず False で試し、
# その後 True にして自分の顔が通ることを確認してください。
LIVENESS_ENABLED = False
LIVENESS_MIN_RANGE_MM = 12     # 顔の凹凸（鼻と頬の差）が最低これだけ必要
LIVENESS_MIN_DIST_MM = 250
LIVENESS_MAX_DIST_MM = 1500
# --------------------------------------------

CAMERA_READY = start_camera()

print("顔認識モデルを読み込み中...（初回は少し時間がかかる場合があります）")
face_app = create_face_app()
print("モデルの準備ができました。")

KNOWN_FACES = load_known_faces()
print(f"データベースに {len(KNOWN_FACES)} 名の学生が登録されています: {list(KNOWN_FACES.keys())}")


def make_placeholder_frame(text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    put_text(img, text, (20, 225), size=22, color=(0, 0, 255))
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def is_live_face(depth_image, box):
    """顔の中心部の深度の凹凸で判定します。深度がなければ True を返します。"""
    if not LIVENESS_ENABLED or depth_image is None:
        return True
    h, w = depth_image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 + bw * 0.25))
    cx2 = int(min(w, x2 - bw * 0.25))
    cy1 = int(max(0, y1 + bh * 0.25))
    cy2 = int(min(h, y2 - bh * 0.25))
    if cx2 <= cx1 or cy2 <= cy1:
        return False
    region = depth_image[cy1:cy2, cx1:cx2].astype(np.float32)
    valid = region[region > 0]
    if valid.size < 0.3 * region.size:
        return False
    p10, p90 = np.percentile(valid, [10, 90])
    median = float(np.median(valid))
    if not (LIVENESS_MIN_DIST_MM <= median <= LIVENESS_MAX_DIST_MM):
        return False
    return (p90 - p10) >= LIVENESS_MIN_RANGE_MM


def draw_face_info(image, box, label, status, ok=True):
    x1, y1, x2, y2 = box
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text = label if not status else f"{label} - {status}"
    tw, th = text_size(text, 18)
    top = max(y1 - th - 14, 0)
    cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + 10), color, -1)
    put_text(image, text, (x1 + 4, top + 5), size=18, color=(255, 255, 255))


def _jpeg_part(frame_bytes):
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---- 顔認識は別スレッドで実行（映像がカクつかないようにする）----
_state_lock = threading.Lock()
_pending = {"color": None, "depth": None}   # 認識待ちの最新フレーム
_last_results = []                          # [(box, label, status, ok)]


def recognition_worker():
    global _last_results
    streaks = {}        # {student_id: 連続認識回数}
    while True:
        with _state_lock:
            color_image = _pending["color"]
            depth_image = _pending["depth"]
            _pending["color"] = None
        if color_image is None:
            time_module.sleep(0.01)
            continue

        try:
            faces = face_app.get(color_image)
            results = []
            seen_now = set()

            for face in faces:
                box = face.bbox.astype(int)
                sid, score = find_match(face.embedding, KNOWN_FACES)

                if sid is None:
                    results.append((box, "未登録", None, False))
                    continue

                if not is_live_face(depth_image, box):
                    results.append((box, "偽物? (写真/画面)", None, False))
                    continue

                seen_now.add(sid)
                streaks[sid] = streaks.get(sid, 0) + 1

                status = get_today_status(sid)
                if status is None and streaks[sid] >= CONFIRM_COUNT:
                    status = mark_attendance(sid, KNOWN_FACES[sid]["name"])
                results.append((box, sid, status or "...", True))

            for sid in list(streaks):
                if sid not in seen_now:
                    del streaks[sid]

            with _state_lock:
                _last_results = results
        except Exception as e:
            print(f"顔認識エラー: {e}")
            time_module.sleep(0.1)


if CAMERA_READY:
    threading.Thread(target=recognition_worker, daemon=True).start()


def generate_frames():
    if not CAMERA_READY:
        frame = make_placeholder_frame("カメラが接続されていないか、起動できませんでした")
        while True:
            yield _jpeg_part(frame)
            time_module.sleep(0.5)

    frame_counter = 0

    while True:
        color_image, depth_image = get_color_and_depth()
        if color_image is None:
            yield _jpeg_part(make_placeholder_frame("フレームを取得できません。カメラを確認してください"))
            time_module.sleep(0.1)
            continue

        frame_counter += 1
        display_image = color_image.copy()

        if frame_counter % RECOGNIZE_EVERY_N_FRAMES == 0:
            with _state_lock:
                if _pending["color"] is None:   # ワーカーが空いているときだけ渡す
                    _pending["color"] = color_image.copy()
                    _pending["depth"] = None if depth_image is None else depth_image.copy()

        with _state_lock:
            results = list(_last_results)

        for box, label, status, ok in results:
            draw_face_info(display_image, box, label, status, ok)

        if depth_image is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((display_image, depth_colormap))
        else:
            images = display_image

        _, buffer = cv2.imencode('.jpg', images)
        yield _jpeg_part(buffer.tobytes())


@app.route('/')
def index():
    return render_template('index.html',
                           camera_ready=CAMERA_READY,
                           camera_mode=camera.MODE or "なし",
                           student_count=len(KNOWN_FACES))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/today')
def api_today():
    return jsonify(get_today_records())


if __name__ == "__main__":
    print("Webサーバーを起動しました: http://127.0.0.1:5000")
    # debug=False: セキュリティのため（Werkzeugデバッガは外部からのコード実行を許す恐れがあります）
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
