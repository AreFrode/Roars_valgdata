from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def load_csv(path):
    """Load a CSV into a pandas DataFrame when pandas is available; otherwise return a list of dicts.
    Path may contain spaces or non-ASCII characters.
    """
    p = Path(path)
    if pd is not None:
        return pd.read_csv(p, engine="python")
    # fallback: use stdlib csv to produce list[dict]
    import csv

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]
