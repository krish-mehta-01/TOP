"""
capture_screenshots_bxb.py — drives the real running frontend (pointed at the
backend serving the ball-by-ball-retrained bowling suite via
TOP_BOWLING_MODEL_DIR=bowling_ballbyball) to capture two screenshots at
genuinely OFF-snapshot, mid-over points -- overs/balls paper 1's timeout-scoped
models were never trained or evaluated on (only overs 5,8,12,15 end-of-over,
zero-indexed). This demonstrates the actual new capability, not a relabeled
version of paper 1's scenarios.

Usage: python capture_screenshots_bxb.py http://localhost:5175
"""

import re
import sys

from playwright.sync_api import sync_playwright

FRONTEND_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5175"
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
    [1, 1, 0, 4, 2, 0],
    [0, 0, 1, 1, 4, 1],
    [1, 6, 0, 1, 0, 2],
]

MI_BOWLERS_ROTATION = ["JJ Bumrah", "TA Boult", "Harbhajan Singh", "SL Malinga"]


def click_run(page, val):
    btn = page.locator(".tactile-grid button", has_text=re.compile(rf"^{val}$"))
    btn.first.click()
    page.wait_for_timeout(60)


def select_bowler(page, name):
    page.select_option("#current_bowler", label=name)
    page.wait_for_timeout(60)


def build_partial_overs(page, full_overs, partial_balls, bowler_rotation):
    """Builds `full_overs` complete overs, then `partial_balls` further legal
    deliveries into the NEXT over -- stopping mid-over at a point paper 1's
    snapshot-only models never saw."""
    for over in range(full_overs):
        bowler = bowler_rotation[over % len(bowler_rotation)]
        select_bowler(page, bowler)
        pattern = RUN_PATTERNS[over % len(RUN_PATTERNS)]
        for runs in pattern:
            click_run(page, runs)
        print(f"  full over {over + 1} done (bowler={bowler})")

    if partial_balls > 0:
        bowler = bowler_rotation[full_overs % len(bowler_rotation)]
        select_bowler(page, bowler)
        pattern = RUN_PATTERNS[full_overs % len(RUN_PATTERNS)]
        for runs in pattern[:partial_balls]:
            click_run(page, runs)
        print(f"  partial over {full_overs + 1}: {partial_balls} balls logged (bowler={bowler})")


def capture_scenario(page, full_overs, partial_balls, out_file):
    print(f"=== Scenario: bowling role, {full_overs} full overs + {partial_balls} balls "
          f"(stopping at over {full_overs + 1}, ball {partial_balls}) ===")
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(1500)

    page.locator('[data-role="bowling"]').click()
    page.wait_for_timeout(200)

    build_partial_overs(page, full_overs, partial_balls, MI_BOWLERS_ROTATION)

    page.wait_for_timeout(300)
    print("  requesting recommendation (mid-over, off-snapshot)...")
    rec_button = page.get_by_role("button", name=re.compile("STRATEGIC TIMEOUT|Get Recommendation"))
    rec_button.click()
    try:
        page.wait_for_selector(".rec-headline", timeout=20000)
        print("  recommendation received")
    except Exception as e:
        print(f"  WARNING: recommendation did not render in time ({e})")
    page.wait_for_timeout(600)

    path = f"{OUT_DIR}/{out_file}"
    page.screenshot(path=path, full_page=True)
    print(f"  saved {path}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1100})

        # Scenario 1: 2 full overs + 4 balls into over 3 (over index 2, ball 4) -
        # never a valid snapshot point for paper 1 (only overs 6/9/13/16 end-of-over).
        capture_scenario(page, full_overs=2, partial_balls=4,
                          out_file="fig_bxb_scenario1_over3.png")

        # Scenario 2: 10 full overs + 3 balls into over 11 (over index 10, ball 3) -
        # deeper into the innings, still off-snapshot and mid-over.
        capture_scenario(page, full_overs=10, partial_balls=3,
                          out_file="fig_bxb_scenario2_over11.png")

        browser.close()


if __name__ == "__main__":
    main()
