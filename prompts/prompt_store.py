"""
PromptStore — PostgreSQL-backed prompt version management for CallSense.
Provides CRUD operations, version activation, rollback, and a 60-second
in-memory cache with LRU eviction to avoid hitting NeonDB on every graph node invocation.
"""

import time
import logging
from collections import OrderedDict

from db.pool import get_conn

logger = logging.getLogger(__name__)

_CACHE_TTL = 60
_CACHE_MAX_SIZE = 50


class PromptStore:
    def __init__(self):
        # OrderedDict for LRU eviction: prompt_key -> (prompt_text, timestamp)
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def _evict_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if (now - ts) >= _CACHE_TTL]
        for k in expired:
            del self._cache[k]

    def _cache_put(self, key: str, value: str):
        """Insert into cache with LRU eviction."""
        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

    # ── Read operations ──────────────────────────────────────────────────

    def get_active_prompt(self, prompt_key: str) -> str:
        """Return the active prompt text for a key, using a 60s memory cache."""
        now = time.time()
        cached = self._cache.get(prompt_key)
        if cached and (now - cached[1]) < _CACHE_TTL:
            self._cache.move_to_end(prompt_key)
            return cached[0]

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT prompt_text FROM prompt_versions WHERE prompt_key = %s AND is_active = TRUE",
                    (prompt_key,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"No active prompt found for key: {prompt_key}")
                text = row[0]
                self._cache_put(prompt_key, text)
                return text

    def list_prompt_versions(self, prompt_key: str) -> list[dict]:
        """Return all versions for a key, newest first."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, prompt_key, version, prompt_text, created_at, created_by, is_active, rollback_of
                    FROM prompt_versions
                    WHERE prompt_key = %s
                    ORDER BY version DESC
                    """,
                    (prompt_key,),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Write operations ─────────────────────────────────────────────────

    def create_prompt_version(self, prompt_key: str, prompt_text: str, created_by: str) -> dict:
        """Insert a new inactive version. Version number auto-increments per key."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM prompt_versions WHERE prompt_key = %s",
                    (prompt_key,),
                )
                next_version = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO prompt_versions (prompt_key, version, prompt_text, created_by, is_active)
                    VALUES (%s, %s, %s, %s, FALSE)
                    RETURNING id, prompt_key, version, prompt_text, created_at, created_by, is_active, rollback_of
                    """,
                    (prompt_key, next_version, prompt_text, created_by),
                )
                cols = [d[0] for d in cur.description]
                row = dict(zip(cols, cur.fetchone()))
            conn.commit()
            return row

    def activate_prompt_version(self, prompt_key: str, version: int) -> dict:
        """Activate a specific version for a key in a single transaction."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE prompt_versions SET is_active = FALSE WHERE prompt_key = %s AND is_active = TRUE",
                    (prompt_key,),
                )
                cur.execute(
                    """
                    UPDATE prompt_versions SET is_active = TRUE
                    WHERE prompt_key = %s AND version = %s
                    RETURNING id, prompt_key, version, prompt_text, created_at, created_by, is_active, rollback_of
                    """,
                    (prompt_key, version),
                )
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Version {version} not found for prompt_key: {prompt_key}")
                cols = [d[0] for d in cur.description]
                row = dict(zip(cols, result))
            conn.commit()
            self._cache.pop(prompt_key, None)
            logger.info("Activated prompt_key=%s version=%d", prompt_key, version)
            return row

    def rollback_prompt(self, prompt_key: str) -> dict:
        """Roll back to the previously active version."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, version FROM prompt_versions WHERE prompt_key = %s AND is_active = TRUE",
                    (prompt_key,),
                )
                current = cur.fetchone()
                if not current:
                    raise ValueError(f"No active prompt to rollback for key: {prompt_key}")
                current_id = current[0]

                cur.execute(
                    """
                    SELECT version FROM prompt_versions
                    WHERE prompt_key = %s AND is_active = FALSE
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (prompt_key,),
                )
                prev = cur.fetchone()
                if not prev:
                    raise ValueError(f"No previous version to rollback to for key: {prompt_key}")
                prev_version = prev[0]

                cur.execute(
                    "UPDATE prompt_versions SET is_active = FALSE WHERE prompt_key = %s AND is_active = TRUE",
                    (prompt_key,),
                )
                cur.execute(
                    """
                    UPDATE prompt_versions SET is_active = TRUE, rollback_of = %s
                    WHERE prompt_key = %s AND version = %s
                    RETURNING id, prompt_key, version, prompt_text, created_at, created_by, is_active, rollback_of
                    """,
                    (current_id, prompt_key, prev_version),
                )
                cols = [d[0] for d in cur.description]
                row = dict(zip(cols, cur.fetchone()))
            conn.commit()
            self._cache.pop(prompt_key, None)
            logger.info("Rolled back prompt_key=%s to version=%d", prompt_key, prev_version)
            return row
