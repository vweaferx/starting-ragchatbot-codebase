import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Fallback: ensure backend modules are importable even without pyproject.toml pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_config():
    """Minimal config mock for unit tests that construct RAGSystem or its components."""
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "test-api-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "/tmp/test-chroma"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return cfg


@pytest.fixture(scope="session")
def _patched_app():
    """
    Import app.py exactly once per session with two module-level side-effects patched:
      1. RAGSystem() instantiation — returns a shared MagicMock instead of touching ChromaDB.
      2. StaticFiles.__init__ — skips the frontend directory existence check.

    Returns (app_module, rag_mock) so both the TestClient and per-test mock configuration
    can reference the same mock object.
    """
    sys.modules.pop("app", None)

    rag_mock = MagicMock()
    rag_mock.session_manager.create_session.return_value = "test-session-id"

    with patch("rag_system.RAGSystem", return_value=rag_mock), \
         patch("starlette.staticfiles.StaticFiles.__init__", return_value=None):
        import app as _app_module

    return _app_module, rag_mock


@pytest.fixture(scope="session")
def api_client(_patched_app):
    """Session-scoped TestClient — one HTTP client instance for all API tests."""
    from fastapi.testclient import TestClient
    app_module, _ = _patched_app
    return TestClient(app_module.app)


@pytest.fixture
def mock_rag(_patched_app):
    """
    Function-scoped fixture that yields the shared RAGSystem mock with a clean slate:
    call counts and side_effects are reset before each test so assertions stay isolated.
    Return values are intentionally NOT reset — each test must set the values it needs.
    """
    _, rag = _patched_app
    rag.reset_mock(side_effect=True)
    # Re-apply the one default that tests depend on implicitly
    rag.session_manager.create_session.return_value = "test-session-id"
    return rag
