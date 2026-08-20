# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright"]
# ///
import json
import pathlib

from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent
data = json.loads((here / "data.json").read_text())
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(device_scale_factor=3)
    page.add_init_script(f"window.figData = {json.dumps(data)}")
    page.goto((here / "figure_organism.html").as_uri())
    page.wait_for_function("window.__ready === true", timeout=15000)
    page.locator(".figure").screenshot(path=str(here / "figure_organism.png"))
    browser.close()
print("wrote", here / "figure_organism.png")
