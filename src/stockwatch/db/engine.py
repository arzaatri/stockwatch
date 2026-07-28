"""SQLAlchemy engine/session factory, built from Settings."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from stockwatch.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().postgres_dsn)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
