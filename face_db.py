"""
顔データベース (faces_db.json)。
形式: {"student_id": {"name": "氏名", "embedding": [...]}}
"""
import json
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faces_db.json')


def load_known_faces():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {sid: {"name": it["name"],
                  "embedding": np.array(it["embedding"], dtype=np.float32)}
            for sid, it in data.items()}


def save_known_faces(known):
    data = {sid: {"name": it["name"], "embedding": it["embedding"].tolist()}
            for sid, it in known.items()}
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_face(student_id, name, embedding):
    known = load_known_faces()
    known[student_id] = {"name": name,
                         "embedding": np.array(embedding, dtype=np.float32)}
    save_known_faces(known)


def find_match(embedding, known_faces, threshold=0.45):
    """コサイン類似度で最も近い学生を探します: (student_id, score)。"""
    if not known_faces:
        return None, 0.0
    emb = np.array(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    best_id, best_score = None, -1.0
    for sid, it in known_faces.items():
        k = it["embedding"] / (np.linalg.norm(it["embedding"]) + 1e-6)
        score = float(np.dot(emb, k))
        if score > best_score:
            best_id, best_score = sid, score
    if best_score >= threshold:
        return best_id, best_score
    return None, best_score
