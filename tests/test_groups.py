"""Group ergonomics: resolving a group by name/object, get_or_create idempotency.

These lock in the ``group=`` convenience added so callers can work with human
group names instead of opaque backend ids.
"""

import pytest

from polyvia._client import _resolve_group_id, _resolve_group_id_async
from polyvia._models import Group


class FakeTransport:
    """Minimal duck-typed transport that serves canned groups and records POSTs."""

    def __init__(self, groups):
        self._groups = [dict(g) for g in groups]  # [{"id", "name"}]
        self.posted = []

    def get(self, path, params=None):
        assert path == "/api/v1/groups"
        return {"groups": self._groups}

    def post(self, path, json=None):
        assert path == "/api/v1/groups"
        self.posted.append(json)
        gid = f"g_new_{len(self.posted)}"
        self._groups.append({"id": gid, "name": json["name"]})
        return {"group_id": gid, "name": json["name"]}


class AsyncFakeTransport(FakeTransport):
    async def get(self, path, params=None):  # type: ignore[override]
        return super().get(path, params)

    async def post(self, path, json=None):  # type: ignore[override]
        return super().post(path, json)


def test_explicit_group_id_wins_without_lookup():
    t = FakeTransport([{"id": "g_1", "name": "Earnings"}])
    assert _resolve_group_id(t, "Earnings", "g_explicit", create=True) == "g_explicit"
    assert t.posted == []  # group_id short-circuits — no list/create


def test_group_object_uses_its_id():
    t = FakeTransport([])
    g = Group(id="g_42", name="Earnings")
    assert _resolve_group_id(t, g, None, create=True) == "g_42"
    assert t.posted == []


def test_name_matches_existing_without_creating():
    t = FakeTransport([{"id": "g_1", "name": "Q4 Earnings"}])
    assert _resolve_group_id(t, "Q4 Earnings", None, create=True) == "g_1"
    assert t.posted == []  # found by name, so no duplicate created


def test_name_missing_is_created_when_create_true():
    t = FakeTransport([{"id": "g_1", "name": "Other"}])
    assert _resolve_group_id(t, "New Group", None, create=True) == "g_new_1"
    assert t.posted == [{"name": "New Group"}]


def test_name_missing_raises_when_create_false():
    t = FakeTransport([{"id": "g_1", "name": "Other"}])
    with pytest.raises(ValueError, match="No group named 'Missing'"):
        _resolve_group_id(t, "Missing", None, create=False)


def test_none_group_returns_none():
    assert _resolve_group_id(FakeTransport([]), None, None, create=True) is None


@pytest.mark.asyncio
async def test_async_resolver_creates_missing_name():
    t = AsyncFakeTransport([{"id": "g_1", "name": "Other"}])
    assert await _resolve_group_id_async(t, "New Group", None, create=True) == "g_new_1"
    assert t.posted == [{"name": "New Group"}]


@pytest.mark.asyncio
async def test_async_resolver_raises_on_unknown_name_for_read():
    t = AsyncFakeTransport([])
    with pytest.raises(ValueError, match="No group named 'Nope'"):
        await _resolve_group_id_async(t, "Nope", None, create=False)
