import io
import time
from PIL import Image
from fastapi.testclient import TestClient


def create_dummy_image_bytes(filename="test.png", color="blue") -> tuple[str, bytes]:
    img = Image.new("RGB", (256, 256), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return filename, buf.getvalue()


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "satquery-backend"


def test_list_models_endpoint(client: TestClient):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    models = resp.json()
    model_names = [m["name"] for m in models]
    assert "GeoChat" in model_names
    assert "ChangeChat" in model_names
    assert "Prithvi" in model_names
    assert "SAR-ML-Fusion" in model_names


def test_single_image_vqa_flow(client: TestClient):
    fn, content = create_dummy_image_bytes("optical.png")
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What is shown in this image?"},
        files=[("images", (fn, content, "image/png"))]
    )
    assert resp.status_code == 200
    res_data = resp.json()
    assert "job_id" in res_data
    job_id = res_data["job_id"]

    # Poll status until completed
    for _ in range(10):
        job_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_data = job_resp.json()
        if job_data["status"] == "completed":
            break
        time.sleep(0.1)

    assert job_data["status"] == "completed"
    assert job_data["result"]["job_id"] == job_id
    assert "GeoChat" in job_data["result"]["execution_summary"]["models"]

    # Verify observable trace
    trace_resp = client.get(f"/api/v1/jobs/{job_id}/trace")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert len(trace_data["trace"]) > 0


def test_bitemporal_change_detection_flow(client: TestClient):
    fn1, content1 = create_dummy_image_bytes("t1.png", "red")
    fn2, content2 = create_dummy_image_bytes("t2.png", "green")

    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What changed between these two images?"},
        files=[
            ("images", (fn1, content1, "image/png")),
            ("images", (fn2, content2, "image/png"))
        ]
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(10):
        job_resp = client.get(f"/api/v1/jobs/{job_id}")
        job_data = job_resp.json()
        if job_data["status"] == "completed":
            break
        time.sleep(0.1)

    assert job_data["status"] == "completed"
    assert "ChangeChat" in job_data["result"]["execution_summary"]["models"]
