"""Example script showing how to use the extracted package."""
from valgdata import load_csv, summarize
from pathlib import Path
import json

if __name__ == '__main__':
    p = Path('Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv')
    df = load_csv(p)
    s = summarize(df)
    print(json.dumps(s, indent=2, ensure_ascii=False))
