"""
カメラモジュール。
まず Intel RealSense（カラー + 深度）を試します。
見つからない場合は、通常のWebカメラ（cv2.VideoCapture）に切り替えます。
"""
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

pipeline = None
align = None
webcam = None
MODE = None  # "realsense" または "webcam"


def start_camera():
    global pipeline, align, webcam, MODE

    if REALSENSE_AVAILABLE:
        try:
            p = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            p.start(config)
            pipeline = p
            align = rs.align(rs.stream.color)  # 深度をカラーフレームに合わせる
            MODE = "realsense"
            print("RealSenseカメラを起動しました。")
            return True
        except Exception as e:
            print(f"RealSenseを起動できませんでした ({e})。Webカメラを試します...")
            pipeline = None
    else:
        print("pyrealsense2 が見つかりません。Webカメラを試します...")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        webcam = cap
        MODE = "webcam"
        print("Webカメラを起動しました（深度なし）。")
        return True

    cap.release()
    print("エラー: カメラが見つかりません。")
    return False


def get_color_and_depth():
    """(color, depth) を返します。Webカメラの場合 depth = None。エラー時は (None, None)。"""
    if MODE == "realsense" and pipeline is not None:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None
            return (np.asanyarray(color_frame.get_data()),
                    np.asanyarray(depth_frame.get_data()))
        except Exception as e:
            print(f"フレーム取得エラー: {e}")
            return None, None

    if MODE == "webcam" and webcam is not None:
        ok, frame = webcam.read()
        if not ok:
            return None, None
        return frame, None

    return None, None
