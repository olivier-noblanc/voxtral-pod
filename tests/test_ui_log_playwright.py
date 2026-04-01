import json
from typing import Any
from playwright.sync_api import Page, expect

def test_log_modal_flow_with_mock_api(context: Any, page: Page, live_server_url: str) -> None:
    """
    Test the Log Modal UI using mocked API responses for stability.
    """
    job_id = "mock-job-123"
    
    # 1. Mock the /status/ endpoint
    def handle_status(route: Any) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "job_id": job_id,
                "status": "processing:Diarisation",
                "progress": 45,
                "created_at": "2026-04-01T12:00:00"
            })
        )
    
    # 2. Mock the /job_log/ endpoint
    log_lines = ["Starting diarisation...", "Processing cluster 1..."]
    def handle_log(route: Any) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(log_lines)
        )

    page.route(f"**/status/{job_id}", handle_status)
    page.route(f"**/job_log/{job_id}", handle_log)

    # 3. Navigate and inject pending job
    page.goto(live_server_url)
    page.evaluate(f"localStorage.setItem('pending_job', '{job_id}');")
    page.reload()

    # 4. Verify "Log" button appears in the status area
    batch_status = page.locator("#batchStatus")
    expect(batch_status).to_contain_text("Diarisation", timeout=10000)
    
    log_button = batch_status.locator("button:has-text('Log')")
    expect(log_button).to_be_visible()

    # 5. Click "Log" and verify modal opens
    log_button.click()
    modal = page.locator("#fr-modal-log")
    # DSFR modals use 'opened' attribute for visibility in Playwright context
    expect(modal).to_have_attribute("opened", "true", timeout=5000)
    
    # 6. Verify log content is loaded
    log_content = page.locator("#logContent")
    expect(log_content).to_contain_text("Starting diarisation...")
    expect(log_content).to_contain_text("Processing cluster 1...")

    # 7. Test Refresh functionality
    log_lines.append("New log line after refresh.")
    page.locator("#refreshLogBtn").click()
    expect(log_content).to_contain_text("New log line after refresh...")

    # 8. Close modal
    page.locator("#fr-modal-log button:has-text('Fermer')").first.click()
    expect(modal).not_to_have_attribute("opened", "true")

def test_error_log_button_with_mock_api(context: Any, page: Page, live_server_url: str) -> None:
    """
    Test that the "Voir le log" button appears when a job is in error state.
    """
    job_id = "mock-error-456"
    
    def handle_status_error(route: Any) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "job_id": job_id,
                "status": "erreur",
                "error_details": "Critical Failure",
                "created_at": "2026-04-01T12:05:00"
            })
        )
    
    def handle_log_error(route: Any) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(["Initial step OK", "ERROR: Out of memory"])
        )

    page.route(f"**/status/{job_id}", handle_status_error)
    page.route(f"**/job_log/{job_id}", handle_log_error)

    # Navigate
    page.goto(live_server_url)
    page.evaluate(f"localStorage.setItem('pending_job', '{job_id}');")
    page.reload()

    # Verify "Voir le log" button
    batch_status = page.locator("#batchStatus")
    expect(batch_status).to_contain_text("Erreur", timeout=10000)
    
    err_button = batch_status.locator("button:has-text('Voir le log')")
    expect(err_button).to_be_visible()

    # Open and verify
    err_button.click()
    expect(page.locator("#logContent")).to_contain_text("ERROR: Out of memory")
