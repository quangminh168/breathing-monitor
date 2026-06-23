from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime
)

from sqlalchemy.orm import declarative_base

from datetime import datetime

Base = declarative_base()


class BreathingRecord(Base):
    __tablename__ = "breathing_records"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    bpm = Column(
        Float,
        nullable=False
    )

    # Nguồn dữ liệu: "camera_ai" (AI worker tự gửi) hoặc giá trị khác
    # nếu sau này bạn thêm nguồn đo khác.
    source = Column(
        String(50),
        default="camera_ai"
    )

    # Ghi chú tuỳ chọn, ví dụ trạng thái ROI/tín hiệu lúc đo.
    note = Column(
        String(255),
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )
