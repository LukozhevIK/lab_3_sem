"""Capture real MLflow 2.16 UI screenshots; no screenshot compositing."""
import json
import os
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(DIRECTORY / ".local" / "browsers"))
from playwright.sync_api import sync_playwright
from mlflow.tracking import MlflowClient

metadata = json.loads((DIRECTORY / "production.json").read_text())
url = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").rstrip("/")
experiment = MlflowClient(tracking_uri=url).get_experiment_by_name(metadata["experiment_name"])

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 2200, "height": 1750})
    page.goto(f"{url}/#/experiments/{experiment.experiment_id}")
    page.get_by_text("matching runs", exact=False).wait_for()
    chart_radio = page.locator('input[name="runs-view-mode"][value="CHART"]')
    if not chart_radio.is_checked():
        chart_radio.locator("xpath=ancestor::label").click()
    page.get_by_text("Model metrics", exact=False).first.wait_for()
    parent = page.get_by_text("optuna_search", exact=True).locator('xpath=ancestor::*[@role="row"]')
    if not page.get_by_text("optuna_trial_00", exact=True).count():
        parent.get_by_role("button").click()
    page.get_by_text("optuna_trial_00", exact=True).wait_for()
    # Keep every run visible in the list, but avoid twelve CV-only trials
    # cluttering the comparison of the four evaluated models.
    for i in range(12):
        row = page.get_by_text(f"optuna_trial_{i:02d}", exact=True).locator('xpath=ancestor::*[@role="row"]')
        checkbox = row.locator('input.is-visibility-toggle-checkbox')
        if checkbox.is_checked():
            checkbox.locator("xpath=ancestor::label").click()
            page.wait_for_timeout(150)
    for name in ["baseline", "sklearn_features", "expert_selection", "optuna_best",
                 "optuna_search", "production_full_data"]:
        row = page.get_by_text(name, exact=True).locator('xpath=ancestor::*[@role="row"]')
        checkbox = row.locator('input.is-visibility-toggle-checkbox')
        if not checkbox.is_checked():
            checkbox.locator("xpath=ancestor::label").click()
            page.wait_for_timeout(150)
    # Remove only supplementary chart cards, never experiment metrics or runs.
    for title in ["best_cv_f1", "completed_trials", "training_seconds"]:
        heading = page.get_by_role("heading", name=title, exact=True)
        if heading.count():
            heading.locator("xpath=../..").get_by_test_id("experiment-view-compare-runs-card-menu").click()
            page.get_by_role("menuitem", name="Delete", exact=True).click()
    page.mouse.move(1950, 90)
    page.wait_for_timeout(800)  # Finish chart animations before capture.
    page.screenshot(path=str(DIRECTORY / "model_runs.png"), full_page=True)
    page.set_viewport_size({"width": 1800, "height": 950})
    page.goto(f"{url}/#/models/{metadata['model_name']}")
    page.get_by_text(f"Version {metadata['production_version']}", exact=True).wait_for()
    page.wait_for_timeout(500)
    close = page.get_by_role("button", name="Close", exact=True)
    if close.count():
        close.click()
    page.get_by_text("Production", exact=True).first.wait_for()
    page.wait_for_timeout(400)
    page.screenshot(path=str(DIRECTORY / "model_versions.png"), full_page=True)
    browser.close()
print("Saved actual MLflow UI: model_runs.png, model_versions.png")
