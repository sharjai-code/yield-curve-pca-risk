"""Public GSW zero-coupon data, expressed in percentage points."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
DEFAULT_MATURITIES = (1, 2, 3, 5, 7, 10, 20, 30)


def download_gsw(destination: str | Path, timeout: int = 30) -> Path:
    """Download the current vintage and retain retrieval time and content hash."""
    path = Path(destination)
    request = Request(GSW_URL, headers={"User-Agent": "yield-curve-pca-risk/0.1"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    # Validate before replacing a previously downloaded file.
    parse_gsw(raw.decode("utf-8-sig"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    metadata = {"url": GSW_URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def parse_gsw(text: str) -> pd.DataFrame:
    """Locate the Date header instead of assuming a fixed preamble length."""
    lines = text.splitlines()
    header = next((i for i, s in enumerate(lines)
                   if s.strip().startswith("Date,") and "SVENY01" in s), None)
    if header is None:
        raise ValueError("No GSW Date/SVENY header found; expected the Fed CSV.")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), na_values=["NA", "N/A"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any():
        raise ValueError("Duplicate curve dates are not allowed.")
    return frame.set_index("Date").sort_index()


def load_yields(path: str | Path, maturities=DEFAULT_MATURITIES,
                start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Select complete curves. No filling, interpolation across dates, or scaling."""
    mats = list(maturities)
    if len(mats) < 2 or len(set(mats)) != len(mats) or any(int(m) != m or m < 1 or m > 30 for m in mats):
        raise ValueError("Use distinct integer maturities between 1 and 30 years.")
    raw = parse_gsw(Path(path).read_text(encoding="utf-8-sig"))
    columns = [f"SVENY{int(m):02d}" for m in sorted(mats)]
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing zero-coupon series: {sorted(missing)}")
    frame = raw.loc[start:end, columns].apply(pd.to_numeric, errors="raise")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame.columns = pd.Index(sorted(mats), name="maturity_years")
    if len(frame) < 10:
        raise ValueError("At least ten complete daily curves are required.")
    return frame


def yield_changes_bp(yields_pct: pd.DataFrame) -> pd.DataFrame:
    """1 percentage point = 100 basis points; difference complete observations."""
    return yields_pct.diff().iloc[1:] * 100.0
