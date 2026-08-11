"""
capture_screenshots.py — drives the real, running TOP frontend (Vite dev server) against
the real, running backend (uvicorn) via Playwright to capture two genuine screenshots for
the paper:
  1. Batting role, scorecard built up ball-by-ball through the end of over 6.
  2. Bowling role, scorecard built up ball-by-ball through over 16.
Neither uses the app's built-in "Load sample innings" shortcut -- every delivery is logged
through the real UI controls, so the screenshots show the actual working system rather than
a canned demo state.

Prerequisites: backend running on :8080, frontend dev server running (URL passed as arg).
Usage: python capture_screenshots.py http://localhost:5174
"""

import re
import sys
import time

from playwright.sync_api import sync_playwright

FRONTEND_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5174"
OUT_DIR = "figures"

RUN_PATTERNS = [
    [1, 0, 4, 1, 1, 6],
    [0, 1, 0, 0, 4, 2],
    [1, 1, 0, 6, 0, 1],
    [0, 4, 1, 0, 1, 2],
    [1, 0, 0, 4, 1, 1],
    [2, 1, 0, 0, 6, 1],
    [1, 4, 0, 1, 0, 2],
    [0, 1, 6, 0, 1, 1],
]

MI_BOWLERS_ROTATION = ["JJ Bumrah", "TA Boult", "Harbhajan Singh", "SL Malinga"]
RCB_NEXT_BATTERS = ["RM Patidar", "PD Salt", "D Padikkal", "F du Plessis", "JM Sharma"]


def click_run(page, val):
    btn = page.locator(".tactile-grid button", has_text=re.compile(rf"^{val}$"))
    btn.first.click()
    page.wait_for_timeout(60)


def select_bowler(page, name):
    page.select_option("#current_bowler", label=name)
    page.wait_for_timeout(60)


def log_wicket(page, next_batter, runs_completed=0):
    page.get_by_role("button", name="WICKET").click()
    page.wait_for_timeout(150)
    selects = page.locator(".modal-box select")
    selects.nth(2).select_option(label=next_batter)  # Next Batter In
    page.wait_for_timeout(80)
    page.get_by_role("button", name="CONFIRM").click()
    page.wait_for_timeout(150)


def build_overs(page, n_overs, bowler_rotation, wicket_overs=None, next_batter_pool=None):
    wicket_overs = wicket_overs or {}
    next_batter_iter = iter(next_batter_pool or [])
    for over in range(n_overs):
        bowler = bowler_rotation[over % len(bowler_rotation)]
        select_bowler(page, bowler)
        pattern = RUN_PATTERNS[over % len(RUN_PATTERNS)]
        for ball_idx, runs in enumerate(pattern):
            if over in wicket_overs and ball_idx == wicket_overs[over]:
                try:
                    nb = next(next_batter_iter)
                except StopIteration:
                    nb = "JM Sharma"
                log_wicket(page, nb, runs_completed=0)
            else:
                click_run(page, runs)
        print(f"  over {over + 1} done (bowler={bowler})")


def wait_backend_live(page):
    page.wait_for_timeout(1500)


def capture_scenario(page, role, n_overs, bowler_rotation, out_file, wicket_overs=None, next_batter_pool=None):
    print(f"=== Scenario: role={role}, overs={n_overs} ===")
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")
    wait_backend_live(page)

    page.locator(f'[data-role="{role}"]').click()
    page.wait_for_timeout(200)

    build_overs(page, n_overs, bowler_rotation, wicket_overs=wicket_overs, next_batter_pool=next_batter_pool)

    page.wait_for_timeout(300)
    print("  requesting recommendation...")
    rec_button = page.get_by_role("button", name=re.compile("STRATEGIC TIMEOUT|Get Recommendation"))
    rec_button.click()
    try:
        page.wait_for_selector(".rec-headline", timeout=20000)
        print("  recommendation received")
    except Exception as e:
        print(f"  WARNING: recommendation did not render in time ({e}); screenshotting current state anyway")
    page.wait_for_timeout(600)

    path = f"{OUT_DIR}/{out_file}"
    page.screenshot(path=path, full_page=True)
    print(f"  saved {path}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1100})

        # Scenario 1: batting role, up through end of over 6
        capture_scenario(
            page, role="batting", n_overs=6,
            bowler_rotation=MI_BOWLERS_ROTATION,
            out_file="fig12_frontend_batting_over6.png",
        )

        # Scenario 2: bowling role, up through over 16, with 3 wickets for realism
        capture_scenario(
            page, role="bowling", n_overs=16,
            bowler_rotation=MI_BOWLERS_ROTATION,
            wicket_overs={2: 2, 8: 4, 12: 1},  # over index -> ball index (0-based) that is a wicket
            next_batter_pool=RCB_NEXT_BATTERS,
            out_file="fig13_frontend_bowling_over16.png",
        )

        browser.close()


if __name__ == "__main__":
    main()
