"""
Shared database connection pool for CallSense.
Uses psycopg2 ThreadedConnectionPool — all modules import get_conn() from here
instead of creating their own connections.
"""

import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_pool = None


def _get_pool():
    """Lazily initialize the connection pool (1–10 connections)."""
    global _pool
    if _pool is None:
        url = os.getenv("NEONDB_URL")
        if not url:
            raise RuntimeError("NEONDB_URL environment variable is not set")
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=url,
            sslmode="require",
            connect_timeout=10,
        )
        logger.info("Database connection pool initialized (1–10 connections)")
    return _pool


@contextmanager
def get_conn():
    """
    Context manager that checks out a connection from the pool and returns it
    when done. Auto-rollbacks on exception, auto-returns to pool in finally.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def close_pool():
    """Shut down the pool — call on app shutdown."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed")
