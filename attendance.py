"""
出席モジュール。各学生は1日1回だけ記録されます。
本日の結果はメモリに保存されます（フレームごとにCSVを読み込みません）。
"""
import csv
import os
import threading
from datetime import datetime, time

ATTENDANCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '出席.csv')

# この時刻より後に来た場合は「遅刻」になります
DEADLINE = time(9, 0)

_lock = threading.Lock()
_cache_date = None
_cache = {}  # {student_id: {"name":..., "time":..., "status":...}}


def _ensure_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['ID', '氏名', '日付', '時刻', '状態'])


def _refresh_cache():
    """日付が変わったとき（または初回）に、本日の記録をCSVから読み込みます。"""
    global _cache_date, _cache
    today = datetime.now().strftime('%Y-%m-%d')
    if _cache_date == today:
        return
    _ensure_file()
    _cache = {}
    try:
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('日付') == today and row.get('ID'):
                    _cache[row['ID']] = {"name": row.get('氏名', ''),
                                         "time": row.get('時刻', ''),
                                         "status": row.get('状態', '')}
    except OSError as e:
        print(f"CSVを読み込めません: {e}")
    _cache_date = today


def get_today_status(student_id):
    with _lock:
        _refresh_cache()
        rec = _cache.get(student_id)
        return rec["status"] if rec else None


def get_today_records():
    with _lock:
        _refresh_cache()
        rows = [{"id": sid, **rec} for sid, rec in _cache.items()]
    rows.sort(key=lambda r: r["time"])
    return rows


def mark_attendance(student_id, name):
    """新規に記録した場合は「出席」/「遅刻」を返し、すでに記録済みなら None を返します。"""
    with _lock:
        _refresh_cache()
        if student_id in _cache:
            return None

        now = datetime.now()
        status = "出席" if now.time() <= DEADLINE else "遅刻"
        try:
            with open(ATTENDANCE_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([student_id, name,
                                        now.strftime('%Y-%m-%d'),
                                        now.strftime('%H:%M:%S'), status])
        except PermissionError:
            print("警告: 出席.csv が他のプログラム（Excelなど）で開かれています。"
                  "閉じてください。次回の試行で書き込まれます。")
            return None

        _cache[student_id] = {"name": name,
                              "time": now.strftime('%H:%M:%S'),
                              "status": status}
        return status
