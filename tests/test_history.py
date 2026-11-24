"""History tests: SQLite round-trip, session isolation, N-turn cap."""
from app.history import HistoryStore


def test_round_trip_order(history_store):
    history_store.add_turn("s1", "user", "hello")
    history_store.add_turn("s1", "assistant", "hi there")
    history = history_store.get_history("s1")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_sessions_are_isolated(history_store):
    history_store.add_turn("s1", "user", "one")
    history_store.add_turn("s2", "user", "two")
    assert [m["content"] for m in history_store.get_history("s1")] == ["one"]
    assert [m["content"] for m in history_store.get_history("s2")] == ["two"]


def test_unknown_session_is_empty(history_store):
    assert history_store.get_history("nope") == []


def test_turn_cap_keeps_only_last_n_turns(tmp_path):
    store = HistoryStore(str(tmp_path / "h.db"), max_turns=2)
    # Three turns = six messages; cap is two turns = four messages.
    for i in range(3):
        store.add_turn("s1", "user", f"q{i}")
        store.add_turn("s1", "assistant", f"a{i}")
    history = store.get_history("s1")
    assert len(history) == 4
    assert [m["content"] for m in history] == ["q1", "a1", "q2", "a2"]


def test_get_history_limit_overrides_cap(history_store):
    for i in range(3):
        history_store.add_turn("s1", "user", f"q{i}")
        history_store.add_turn("s1", "assistant", f"a{i}")
    history = history_store.get_history("s1", limit=1)
    assert [m["content"] for m in history] == ["q2", "a2"]


def test_clear_removes_only_that_session(history_store):
    history_store.add_turn("s1", "user", "one")
    history_store.add_turn("s2", "user", "two")
    history_store.clear("s1")
    assert history_store.get_history("s1") == []
    assert len(history_store.get_history("s2")) == 1
