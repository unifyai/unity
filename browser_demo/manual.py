import os
import sys, pathlib
import time

# Ensure repository root is on PYTHONPATH so `import unity` works when this
# script is executed directly from inside the "sandboxes" folder.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from playwright.sync_api import sync_playwright
from unity.controller.session import create_session, get_live_view_urls, get_recording
from dotenv import load_dotenv

load_dotenv()


session = create_session()

print(session.connect_url)
print(session.id)

with sync_playwright() as playwright:
    browser = playwright.chromium.connect_over_cdp(session.connect_url)
    ctx = browser.contexts[0]
    page = ctx.pages[0]
    page.goto("https://www.google.com")
    print(get_live_view_urls(session.id))
    time.sleep(30)
