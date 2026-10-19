# 顔認証出席システム

## インストール
    pip install -r requirements.txt
(Windowsで insightface のインストール時にエラーが出る場合: 「Microsoft C++ Build Tools」をインストールしてから、もう一度試してください。)

## 使い方
1. 学生の登録（学生ごとに1回）:
       python register_face.py
   IDと氏名を入力し、's' を5回押してください。
2. システムの起動:
       python app.py
   ブラウザで: http://127.0.0.1:5000

## ファイル
- app.py           - Webサーバー、顔認識と出席記録
- register_face.py - 顔の登録
- face_db.py       - 顔データベース (faces_db.json)
- attendance.py    - 出席記録 (出席.csv)
- camera.py        - RealSense またはWebカメラ
- jp_text.py       - 映像上に日本語を描画するヘルパー
- face_model.py    - 顔認識モデルの設定（GPUがあれば自動で使用）

## 設定 (app.py の先頭)
- CONFIRM_COUNT - 出席記録に必要な連続認識回数
- LIVENESS_ENABLED - RealSenseの深度による写真/画面なりすまし対策
- attendance.py: DEADLINE - 遅刻の基準時刻（現在は9:00）

## 注意事項
- 古い faces_db.json や 出席.csv / davomat.csv とは形式が異なります。古いファイルは削除して、顔を登録し直してください。
- 出席.csv がExcelで開かれていると書き込めません。閉じてください。
- faces_db.json は生体情報です。他人に渡さないでください。
