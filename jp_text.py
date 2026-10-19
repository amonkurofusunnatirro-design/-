"""
日本語テキスト描画ヘルパー。
cv2.putText は日本語を描画できないため、Pillow を使って画像に文字を描きます。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/yugothic.ttf",
    # macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]

_font_cache = {}


def _get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if font is None:
        print("警告: 日本語フォントが見つかりません。jp_text.py の _FONT_CANDIDATES にフォントを追加してください。")
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def text_size(text, size=18):
    """テキストの (幅, 高さ) を返します。"""
    bbox = _get_font(size).getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def put_text(image, text, org, size=18, color=(255, 255, 255)):
    """
    image (BGR, numpy) に日本語テキストを描きます。
    org = 左上の座標 (x, y)、color = BGR。image を直接書き換えて返します。
    （速度のため、テキスト周辺の小さな領域だけを変換します）
    """
    font = _get_font(size)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 2
    h = bbox[3] - bbox[1] + 2
    x, y = int(org[0]), int(org[1])
    H, W = image.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return image
    patch = image[y0:y1, x0:x1]
    pil_img = Image.fromarray(np.ascontiguousarray(patch[:, :, ::-1]))  # BGR -> RGB
    draw = ImageDraw.Draw(pil_img)
    draw.text((x - x0 - bbox[0], y - y0 - bbox[1]), text, font=font,
              fill=(color[2], color[1], color[0]))
    patch[:] = np.asarray(pil_img)[:, :, ::-1]
    return image
