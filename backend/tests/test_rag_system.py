"""
Tests for RAGSystem.query() — the main content-query pipeline.
Verifies that AI generation, tool retrieval, source collection,
session management, and cleanup are all orchestrated correctly.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def rag(tmp_path):
    """
    Build a RAGSystem with all external dependencies mocked so no
    real ChromaDB, embedding model, or Anthropic API is needed.
    """
    with patch("rag_system.VectorStore"), \
         patch("rag_system.DocumentProcessor"), \
         patch("rag_system.AIGenerator"), \
         patch("rag_system.SessionManager"):

        from rag_system import RAGSystem

        cfg = MagicMock()
        cfg.ANTHROPIC_API_KEY = "test_key"
        cfg.ANTHROPIC_MODEL = "claude-test"
        cfg.CHUNK_SIZE = 800
        cfg.CHUNK_OVERLAP = 100
        cfg.MAX_RESULTS = 5
        cfg.MAX_HISTORY = 2
        cfg.CHROMA_PATH = str(tmp_path)
        cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"

        system = RAGSystem(cfg)

    # Replace the real ToolManager with a mock so we can control it per test
    system.tool_manager = MagicMock()
    system.tool_manager.get_tool_definitions.return_value = [
        {"name": "search_course_content"},
        {"name": "get_course_outline"},
    ]
    system.tool_manager.get_last_sources.return_value = []

    return system


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

class TestQueryReturnShape:
    def test_returns_two_element_tuple(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"
        result = rag.query("What is RAG?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_answer_string(self, rag):
        rag.ai_generator.generate_response.return_value = "The answer"
        answer, _ = rag.query("What is RAG?")
        assert answer == "The answer"

    def test_second_element_is_sources_list(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"
        rag.tool_manager.get_last_sources.return_value = ["Course A - Lesson 1"]
        _, sources = rag.query("What is RAG?")
        assert sources == ["Course A - Lesson 1"]

    def test_sources_empty_when_no_tool_called(self, rag):
        rag.ai_generator.generate_response.return_value = "General answer"
        rag.tool_manager.get_last_sources.return_value = []
        _, sources = rag.query("What is AI?")
        assert sources == []


# ---------------------------------------------------------------------------
# Tool invocation wiring
# ---------------------------------------------------------------------------

class TestToolWiring:
    def test_tool_definitions_passed_to_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"
        tool_defs = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        rag.tool_manager.get_tool_definitions.return_value = tool_defs

        rag.query("Question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tools"] == tool_defs

    def test_tool_manager_passed_to_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("Question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tool_manager"] is rag.tool_manager

    def test_sources_reset_after_every_query(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("Question")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_sources_fetched_before_reset(self, rag):
        """get_last_sources must be called before reset_sources."""
        call_order = []
        rag.tool_manager.get_last_sources.side_effect = lambda: call_order.append("get") or []
        rag.tool_manager.reset_sources.side_effect = lambda: call_order.append("reset")
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("Question")

        assert call_order == ["get", "reset"]


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_history_passed_to_generator_when_session_exists(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"
        rag.session_manager.get_conversation_history.return_value = (
            "User: hi\nAssistant: hello"
        )

        rag.query("New question", session_id="session_1")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_history_is_none_when_no_session(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("Question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] is None

    def test_history_is_none_when_session_has_no_history(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("Question", session_id="session_1")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] is None


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_session_updated_after_response(self, rag):
        rag.ai_generator.generate_response.return_value = "The answer"

        rag.query("My question", session_id="session_1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "session_1", "My question", "The answer"
        )

    def test_session_not_updated_when_no_session_id(self, rag):
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("Question")

        rag.session_manager.add_exchange.assert_not_called()
