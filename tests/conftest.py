import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.init_registry import register_default_models


@pytest.fixture(autouse=True)
def init_models():
    register_default_models()


@pytest.fixture
def client():
    return TestClient(app)
