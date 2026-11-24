"""CORS behaviour.

A dev server that drifts off port 5173 must still be able to talk to the API:
Vite silently picks the next free port, and a rejected preflight surfaces as a
generic upload failure rather than as a CORS problem.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app()) as test_client:
        yield test_client


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/analyses",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


@pytest.mark.parametrize("port", [5173, 5174, 5175, 3000])
def test_localhost_ports_are_allowed_in_development(client: TestClient, port: int) -> None:
    origin = f"http://localhost:{port}"
    response = _preflight(client, origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_127_0_0_1_is_allowed(client: TestClient) -> None:
    origin = "http://127.0.0.1:5173"
    response = _preflight(client, origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin", ["http://evil.com", "http://localhost.evil.com", "https://localhost"]
)
def test_other_origins_are_refused(client: TestClient, origin: str) -> None:
    response = _preflight(client, origin)
    assert "access-control-allow-origin" not in response.headers


def test_cors_origins_env_accepts_comma_separated_production_urls(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173, https://foshollens.vercel.app/")
    monkeypatch.setenv("APP_ENV", "production")
    parsed = Settings(_env_file=None)
    assert parsed.cors_origins == ["http://localhost:5173", "https://foshollens.vercel.app"]

    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "cors_origin", parsed.cors_origin)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(create_app()) as prod_client:
        allowed = _preflight(prod_client, "https://foshollens.vercel.app")
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "https://foshollens.vercel.app"

        blocked = _preflight(prod_client, "http://localhost:5174")
        assert "access-control-allow-origin" not in blocked.headers


def test_wildcard_cors_disables_allow_credentials(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "cors_origin", "*")
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(create_app()) as wildcard_client:
        response = _preflight(wildcard_client, "https://example.com")
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert response.headers.get("access-control-allow-credentials") != "true"
