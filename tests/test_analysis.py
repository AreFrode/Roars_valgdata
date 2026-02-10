from valgdata.analysis import summarize


def test_summarize_basic():
    df = [{"a": 1}, {"a": 2}, {"a": 3}]
    s = summarize(df)
    assert s["rows"] == 3
    assert s["columns"] == 1
