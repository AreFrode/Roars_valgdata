def summarize(df):
    """Return a small summary dict for programmatic use/tests.

    Supports both pandas.DataFrame and a list of dicts (fallback when pandas is not installed).
    """
    # Try pandas-style handling
    try:
        import pandas as pd

        is_pd = pd is not None and hasattr(df, "shape") and hasattr(df, "columns")
    except Exception:
        is_pd = False

    if is_pd:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "columns_list": list(df.columns),
            "describe": df.describe(include="all").to_dict(),
        }

    # Fallback for list of dicts
    if isinstance(df, list):
        rows = len(df)
        cols = list(df[0].keys()) if rows > 0 else []
        desc = {}
        for c in cols:
            vals = [r.get(c) for r in df]
            non_null = [v for v in vals if v not in (None, "")]
            unique = len(set(non_null))
            sample = non_null[:5]
            desc[c] = {"count": len(non_null), "unique": unique, "sample": sample}
        return {
            "rows": rows,
            "columns": len(cols),
            "columns_list": cols,
            "describe": desc,
        }

    return {"rows": 0, "columns": 0, "columns_list": [], "describe": {}}
