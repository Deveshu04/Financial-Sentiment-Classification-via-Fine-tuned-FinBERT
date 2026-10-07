import pytest

from errors import ValidationError


def test_query_all_sorted(market):
    rows = market.query()
    assert len(rows) == 60
    assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)


def test_query_range_is_inclusive(market):
    dates = [r["date"] for r in market.query()]
    assert [r["date"] for r in market.query(dates[10], dates[19])] == dates[10:20]
    assert [r["date"] for r in market.query(start=dates[55])] == dates[55:]
    assert [r["date"] for r in market.query(end=dates[2])] == dates[:3]
    assert market.query("", "") == market.query()


@pytest.mark.parametrize("value", ["2020/01/02", "2020-13-01", "2020-02-30", "20200102", "2020-W01-1", "2020-1-2", "yesterday", " 2020-01-02"])
def test_bad_dates_rejected(market, value):
    with pytest.raises(ValidationError):
        market.query(value, None)
    with pytest.raises(ValidationError):
        market.query(None, value)


def test_start_after_end_rejected(market):
    with pytest.raises(ValidationError, match="after"):
        market.query("2020-03-01", "2020-02-01")


def test_test_sessions_carry_predictions(market):
    tested = [r for r in market.query() if r["in_test"]]
    assert len(tested) == 40
    assert all(r["pred_up"] is not None and r["correct"] == (r["pred_up"] == r["actual_up"]) for r in tested)
    assert all(r["pred_up"] is None and r["correct"] is None for r in market.query() if not r["in_test"])
