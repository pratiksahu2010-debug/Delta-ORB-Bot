from app.signal_journal import SignalJournal


def test_training_feedback_report_and_undo(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    candle_time = "2026-11-18T09:20:00+00:00"
    journal.record_evaluation(
        "MEX", "BTCUSD", candle_time, "setup",
        {"reason": "momentum_confirmed", "values": {"rvol": 1.8}},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0},
    )
    evaluation_id = journal.evaluation_id("MEX", "BTCUSD", candle_time)

    assert journal.add_feedback(evaluation_id, "good", "Clean early entry")
    assert journal.learning_report()["feedback_counts"] == {"good": 1}
    assert journal.learning_report()["live_strategy_modified"] is False

    assert journal.undo_feedback(evaluation_id)
    assert journal.learning_report()["human_reviewed"] == 0
    assert journal.latest_record(decision="setup")["evaluation_id"] == evaluation_id


def test_training_notes_are_research_only_and_persist(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    journal.add_training_note("Avoid reviewing entries after an already extended impulse")
    notes = journal.list_training_notes()
    assert len(notes) == 1
    assert "extended impulse" in notes[0]["note"]
    assert journal.learning_report()["training_notes"] == 1


def test_feedback_token_maps_to_the_exact_signal(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    candle_time = "2026-11-18T09:20:00+00:00"
    journal.record_evaluation(
        "MEX", "BTCUSD", candle_time, "setup",
        {"reason": "momentum_confirmed"},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0},
    )
    evaluation_id = journal.evaluation_id("MEX", "BTCUSD", candle_time)
    token = journal.create_feedback_token(evaluation_id)
    assert token
    assert len(token) <= 16
    assert journal.resolve_feedback_token(token) == evaluation_id
    assert journal.resolve_feedback_token("not-a-token") is None
