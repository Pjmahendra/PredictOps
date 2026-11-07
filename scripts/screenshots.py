"""Capture dashboard screenshots (for the report / README). Needs API + Streamlit running and Google Chrome.
Run: .venv/bin/python scripts/screenshots.py"""
from playwright.sync_api import sync_playwright

from predictops.config import project_path

OUT = project_path("reports/screenshots"); OUT.mkdir(parents=True, exist_ok=True)
PAGES = {"overview": "", "sensors": "sensors", "replay": "replay", "prediction": "prediction", "anomaly": "anomaly",
         "explain": "explain", "health": "health", "maintenance": "maintenance", "performance": "performance", "monitoring": "monitoring"}
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    for name, path in PAGES.items():
        pg.goto(f"http://localhost:8501/{path}")
        pg.wait_for_selector(".po-title", timeout=60_000)
        pg.wait_for_timeout(6000)                         # let charts render
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print("saved", name)
    b.close()
