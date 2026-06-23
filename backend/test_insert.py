from database import SessionLocal
from models import BreathingRecord

db = SessionLocal()

record = BreathingRecord(
    bpm=18.5
)

db.add(record)

db.commit()

print("Inserted")