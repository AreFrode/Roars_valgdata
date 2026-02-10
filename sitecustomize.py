# sitecustomize hook to monkeypatch pandas.read_csv for notebook execution
import os

TARGET = '/home/arefrode/RoarValg/Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv'
REPO_CSV = os.path.join(os.path.dirname(__file__), 'Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv')

try:
    import pandas as _pd
    _orig_read_csv = _pd.read_csv

    def _patched_read_csv(filepath_or_buffer, *args, **kwargs):
        try_path = filepath_or_buffer
        # If notebook uses the absolute path from a different machine, redirect to local copy
        if isinstance(filepath_or_buffer, str) and os.path.normpath(filepath_or_buffer) == os.path.normpath(TARGET):
            if os.path.exists(REPO_CSV):
                try_path = REPO_CSV
        return _orig_read_csv(try_path, *args, **kwargs)

    _pd.read_csv = _patched_read_csv
except Exception:
    # If pandas isn't available or monkeypatching fails, silently continue
    pass
