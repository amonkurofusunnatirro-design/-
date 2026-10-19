"""
学生の顔を登録します。

実行方法:  python register_face.py

カメラウィンドウで:
    's' - フレームを保存（少し顔の向きを変えながら5回）
    'q' - 終了
注意: キーを有効にするには、カメラの「ウィンドウ」をクリックして選択しておく必要があります。
"""
import cv2
import numpy as np

from camera import start_camera, get_color_and_depth
from face_db import add_face
from jp_text import put_text
from face_model import create_face_app

SAMPLES_NEEDED = 5

print("顔認識モデルを読み込み中...（初回は少し時間がかかる場合があります）")
face_app = create_face_app()
print("モデルの準備ができました。")

if not start_camera():
    print("カメラを起動できませんでした。プログラムを終了します。")
    raise SystemExit(1)

student_id = input("学生ID（学生番号）: ").strip()
name = input("氏名: ").strip()
if not student_id or not name:
    print("IDと氏名は空にできません。")
    raise SystemExit(1)

print(f"\n顔を枠の中に入れてください。's' を{SAMPLES_NEEDED}回押してください"
      "（毎回少し顔の向きを変えて）。'q' でキャンセル。\n")

samples = []
while len(samples) < SAMPLES_NEEDED:
    color_image, _ = get_color_and_depth()
    if color_image is None:
        cv2.waitKey(10)
        continue

    faces = face_app.get(color_image)
    display = color_image.copy()
    for face in faces:
        b = face.bbox.astype(int)
        cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    info = f"ID: {student_id} | 顔: {len(faces)} | 保存済み: {len(samples)}/{SAMPLES_NEEDED} | s=保存 q=終了"
    put_text(display, info, (10, 8), size=16, color=(0, 255, 255))
    cv2.imshow("登録", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(faces) == 1:
            emb = faces[0].embedding
            samples.append(emb / (np.linalg.norm(emb) + 1e-6))
            print(f"サンプル {len(samples)}/{SAMPLES_NEEDED} を保存しました。")
        elif len(faces) == 0:
            print("顔が見つかりません。カメラを正しく向けてください。")
        else:
            print("複数の顔が映っています。枠内には1人だけにしてください。")
    elif key == ord('q'):
        print("キャンセルしました。")
        break

cv2.destroyAllWindows()

if len(samples) == SAMPLES_NEEDED:
    mean_emb = np.mean(samples, axis=0)
    add_face(student_id, name, mean_emb)
    print(f"'{name}' (ID: {student_id}) の登録が完了しました！")
