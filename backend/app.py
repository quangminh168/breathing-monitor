import os

from flask import Flask, render_template, jsonify, request
from sqlalchemy import desc

from database import SessionLocal
from models import BreathingRecord

# backend/app.py -> lùi 1 cấp để ra thư mục gốc breathing_monitor/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "dashboard", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "dashboard", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)


def record_to_dict(record):
    return {
        "id": record.id,
        "bpm": record.bpm,
        "source": record.source,
        "note": record.note,
        "recorded_at": record.created_at.isoformat(),
    }


@app.route("/")
def dashboard():
    """Trang dashboard chính (dashboard/templates/index.html)."""
    return render_template("index.html")


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Trả về bản ghi nhịp thở gần nhất."""
    db = SessionLocal()
    try:
        record = (
            db.query(BreathingRecord)
            .order_by(desc(BreathingRecord.created_at))
            .first()
        )
        if record is None:
            return jsonify({"message": "Chưa có dữ liệu"}), 404
        return jsonify(record_to_dict(record)), 200
    finally:
        db.close()


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Trả về danh sách lịch sử nhịp thở, mới nhất trước.
    Query params (tuỳ chọn): limit (mặc định 100).
    """
    limit = request.args.get("limit", default=100, type=int)
    db = SessionLocal()
    try:
        records = (
            db.query(BreathingRecord)
            .order_by(desc(BreathingRecord.created_at))
            .limit(limit)
            .all()
        )
        return jsonify([record_to_dict(r) for r in records]), 200
    finally:
        db.close()


@app.route("/api/breathing", methods=["POST"])
def ingest_breathing():
    """
    Endpoint để AI worker (đo nhịp thở bằng camera, chạy độc lập)
    PUSH kết quả BPM mới lên.

    Body JSON:
    {
        "bpm": 14.7,
        "source": "camera_ai",
        "note": "tuỳ chọn"
    }
    """
    data = request.get_json(silent=True)
    if not data or "bpm" not in data:
        return jsonify({"error": "Thiếu field 'bpm' trong body JSON"}), 400

    try:
        bpm_value = float(data["bpm"])
    except (TypeError, ValueError):
        return jsonify({"error": "'bpm' phải là số"}), 400

    db = SessionLocal()
    try:
        record = BreathingRecord(
            bpm=bpm_value,
            source=data.get("source", "camera_ai"),
            note=data.get("note"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return jsonify(record_to_dict(record)), 201
    finally:
        db.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
