"""
顔認識モデルの設定と作成。app.py と register_face.py で共通に使います。
NVIDIA GPU（onnxruntime-gpu）が使える場合は自動で GPU を使い、なければ CPU を使います。

重要: 登録時と認識時で必ず同じ MODEL_NAME を使ってください。
      MODEL_NAME を変えた場合は faces_db.json を削除し、顔を登録し直してください。
"""
from insightface.app import FaceAnalysis

MODEL_NAME = 'buffalo_l'   # InsightFace が自動でダウンロードする最強パック


def create_face_app():
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        available = []

    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id, det_size, device = 0, (640, 640), "GPU (CUDA)"
    else:
        providers = ['CPUExecutionProvider']
        ctx_id, det_size, device = -1, (320, 320), "CPU"

    print(f"顔認識モデル: {MODEL_NAME} / 実行環境: {device} / 検出サイズ: {det_size}")
    face_app = FaceAnalysis(name=MODEL_NAME, providers=providers)
    face_app.prepare(ctx_id=ctx_id, det_size=det_size)
    return face_app
