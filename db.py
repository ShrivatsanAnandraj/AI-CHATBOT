import os
from datetime import datetime
from pathlib import Path

# Load .env if present (DATABASE_URL, ADMIN_PASSWORD, ...)
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                        create_engine, select)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or "sqlite:///data/local.db"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


class Sample(Base):
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    domain = Column(String(8), nullable=False)      # cv | dl | nlp | ml | ds
    payload = Column(Text, nullable=False)          # JSON
    label = Column(String(64), nullable=True)       # None => unlabeled
    target = Column(Float, nullable=True)           # regression target (ds)
    created_at = Column(DateTime, default=datetime.utcnow)
    labeled_at = Column(DateTime, nullable=True)


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True)
    role = Column(String(8), nullable=False)        # user | bot
    message = Column(Text, nullable=False)
    intent = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)


def init_db():
    from werkzeug.security import generate_password_hash
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(engine)
    with SessionLocal() as s:
        if s.scalar(select(Admin).limit(1)) is None:
            username = os.environ.get("ADMIN_USERNAME", "admin")
            password = os.environ.get("ADMIN_PASSWORD", "openitopen")
            s.add(Admin(username=username,
                        password_hash=generate_password_hash(password)))
            s.commit()


def verify_admin(username, password):
    from werkzeug.security import check_password_hash
    with SessionLocal() as s:
        admin = s.scalar(select(Admin).where(Admin.username == username))
        return bool(admin and check_password_hash(admin.password_hash, password))


def fetch_samples(domain, labeled=True):
    with SessionLocal() as s:
        stmt = select(Sample).where(Sample.domain == domain)
        stmt = stmt.where(Sample.label.is_(None) if not labeled
                          else Sample.label.is_not(None))
        stmt = stmt.order_by(Sample.id)
        return list(s.scalars(stmt))


def add_sample(domain, payload, label=None, target=None):
    import json
    with SessionLocal() as s:
        row = Sample(domain=domain, payload=json.dumps(payload),
                     label=None if label is None else str(label),
                     target=target,
                     labeled_at=datetime.utcnow() if label is not None else None)
        s.add(row)
        s.commit()
        return row.id


def set_label(sample_id, label=None, target=None):
    with SessionLocal() as s:
        row = s.get(Sample, sample_id)
        if row is None:
            return False
        row.label = None if label is None else str(label)
        row.target = target
        row.labeled_at = datetime.utcnow() if label is not None else None
        s.commit()
        return True


def delete_sample(sample_id):
    with SessionLocal() as s:
        row = s.get(Sample, sample_id)
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def sample_counts():
    counts = {}
    with SessionLocal() as s:
        for domain in ("cv", "dl", "nlp", "ml", "ds"):
            rows = list(s.scalars(select(Sample).where(Sample.domain == domain)))
            counts[domain] = {
                "labeled": sum(1 for r in rows if r.label is not None),
                "unlabeled": sum(1 for r in rows if r.label is None),
            }
    return counts


def add_chat(role, message, intent=None):
    with SessionLocal() as s:
        s.add(ChatLog(role=role, message=message, intent=intent))
        s.commit()


def recent_chats(limit=50):
    from sqlalchemy import desc
    with SessionLocal() as s:
        rows = list(s.scalars(select(ChatLog).order_by(desc(ChatLog.id)).limit(limit)))
        return [{"role": r.role, "message": r.message, "intent": r.intent,
                 "created_at": r.created_at.isoformat()} for r in reversed(rows)]
