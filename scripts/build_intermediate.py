"""Step 1 of the pipeline: raw CSV -> validated, typed, segmented Parquet.

Run:  .venv/bin/python scripts/build_intermediate.py
"""
import time
from predictops.data.ingestion import build_intermediate, load_intermediate

t0 = time.time()
out = build_intermediate()
df = load_intermediate()
print(f"Wrote {out} in {time.time()-t0:.1f}s")
print(f"rows={len(df):,}  segments={df.segment_id.nunique()}  memory={df.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"parquet size={out.stat().st_size/1e6:.1f} MB")
