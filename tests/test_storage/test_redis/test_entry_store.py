"""RedisEntryStore over the in-process fake: expiry is filtered like every other backend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from anchor.storage.redis._entry_store import RedisEntryStore
from tests.conftest import make_memory_entry as _make_entry
from tests.test_storage.test_review_fixes import _FakeConnManager


def test_expired_entries_are_hidden_but_kept_unfiltered() -> None:
    store = RedisEntryStore(_FakeConnManager())
    past = datetime.now(UTC) - timedelta(hours=1)
    store.add(_make_entry(entry_id="old", content="memory old", expires_at=past))
    store.add(_make_entry(entry_id="live", content="memory live"))

    assert [e.id for e in store.search("memory")] == ["live"]
    assert [e.id for e in store.list_all()] == ["live"]
    assert {e.id for e in store.list_all_unfiltered()} == {"old", "live"}
