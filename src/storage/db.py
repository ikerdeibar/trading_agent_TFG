from __future__ import annotations
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.storage.models import Base
import os

load_dotenv()


@lru_cache(maxsize=1)
def get_engine():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL is not set. Check your .env file.")
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def create_tables() -> None:
    Base.metadata.create_all(get_engine())


def health_check() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
