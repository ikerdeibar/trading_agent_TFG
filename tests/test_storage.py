from src.storage.db import health_check, create_tables, get_session_factory
from src.storage.repository import log_agent

def test_health():
    assert health_check() is True

def test_log_agent():
    create_tables()
    session = get_session_factory()()
    entry = log_agent(session, agent="test", message="smoke test", level="INFO")
    assert entry.id is not None
    session.close()
