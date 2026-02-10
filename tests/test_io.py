from pathlib import Path
from valgdata import load_csv


def test_load_csv_exists():
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv"
    df = load_csv(p)
    # Support both pandas DataFrame and list-of-dicts fallback
    if hasattr(df, "shape"):
        assert df.shape[0] > 0
    else:
        assert len(df) > 0
