"""Screenshots of the running app (drives the installed Microsoft Edge, headless)."""
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOW = Path(__file__).resolve().parents[1] / "portfolio"
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
    page.goto("http://localhost:8502", wait_until="networkidle")
    page.get_by_text("Likely range").first.wait_for(timeout=120_000)
    page.wait_for_timeout(2500)
    page.screenshot(path=str(SHOW / "app_product.png"))
    page.get_by_role("tab", name="Stock plan").click()
    page.wait_for_timeout(2500)
    page.screenshot(path=str(SHOW / "app_stock_plan.png"))
    browser.close()
print("saved")
