"""
Tests for FastAPI endpoint request/response handling.

All tests use a session-scoped TestClient backed by a mocked RAGSystem so no
real Anthropic API calls, ChromaDB, or frontend static files are required.
"""
import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_returns_200_with_valid_query(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Here is the answer.", ["Course A - Lesson 1"])
        response = api_client.post("/api/query", json={"query": "What is RAG?"})
        assert response.status_code == 200

    def test_response_schema(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Detailed answer.", ["Source 1"])
        data = api_client.post("/api/query", json={"query": "What is RAG?"}).json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_response_contains_answer_and_sources(self, api_client, mock_rag):
        mock_rag.query.return_value = ("The answer.", ["Course A - Lesson 1", "Course B - Lesson 2"])
        data = api_client.post("/api/query", json={"query": "RAG"}).json()
        assert data["answer"] == "The answer."
        assert data["sources"] == ["Course A - Lesson 1", "Course B - Lesson 2"]

    def test_auto_generates_session_id_when_not_provided(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        data = api_client.post("/api/query", json={"query": "hi"}).json()
        assert data["session_id"] == "test-session-id"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        data = api_client.post(
            "/api/query", json={"query": "hi", "session_id": "existing-session"}
        ).json()
        assert data["session_id"] == "existing-session"
        mock_rag.session_manager.create_session.assert_not_called()

    def test_query_forwarded_to_rag_system(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        api_client.post("/api/query", json={"query": "Explain embeddings", "session_id": "s1"})
        mock_rag.query.assert_called_once()
        positional_args = mock_rag.query.call_args[0]
        assert "Explain embeddings" in positional_args[0]

    def test_returns_500_on_rag_error(self, api_client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("db unavailable")
        response = api_client.post("/api/query", json={"query": "question"})
        assert response.status_code == 500

    def test_missing_query_field_returns_422(self, api_client, mock_rag):
        response = api_client.post("/api/query", json={"session_id": "s1"})
        assert response.status_code == 422

    def test_empty_sources_list_is_valid(self, api_client, mock_rag):
        mock_rag.query.return_value = ("General answer.", [])
        data = api_client.post("/api/query", json={"query": "hello"}).json()
        assert data["sources"] == []


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_200(self, api_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["RAG", "Agents", "MCP"],
        }
        assert api_client.get("/api/courses").status_code == 200

    def test_response_contains_total_and_titles(self, api_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }
        data = api_client.get("/api/courses").json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_returns_empty_catalog(self, api_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        data = api_client.get("/api/courses").json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_returns_500_on_analytics_error(self, api_client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("db error")
        response = api_client.get("/api/courses")
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    def test_returns_200(self, api_client, mock_rag):
        assert api_client.delete("/api/session/my-session").status_code == 200

    def test_returns_ok_status(self, api_client, mock_rag):
        data = api_client.delete("/api/session/my-session").json()
        assert data == {"status": "ok"}

    def test_delegates_to_session_manager(self, api_client, mock_rag):
        api_client.delete("/api/session/session-to-delete")
        mock_rag.session_manager.clear_session.assert_called_once_with("session-to-delete")
