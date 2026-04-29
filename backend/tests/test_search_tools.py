"""
Tests for CourseSearchTool.execute() and its formatting logic.
"""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.3] * len(docs),
    )


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# ---------------------------------------------------------------------------
# execute() — routing and filter passing
# ---------------------------------------------------------------------------

class TestExecuteRouting:
    def test_returns_formatted_content_on_success(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["RAG combines retrieval with generation."],
            [{"course_title": "RAG Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool.execute(query="What is RAG?")

        assert "RAG Course" in result
        assert "Lesson 1" in result
        assert "RAG combines retrieval" in result

    def test_returns_search_error_message_on_error(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty(
            "No course found matching 'Unknown'"
        )

        result = tool.execute(query="topic", course_name="Unknown")

        assert "No course found matching" in result

    def test_returns_no_results_message_when_empty(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_no_results_message_includes_course_filter(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="topic", course_name="MCP Course")

        assert "MCP Course" in result

    def test_no_results_message_includes_lesson_filter(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="topic", lesson_number=5)

        assert "lesson 5" in result

    def test_passes_all_filters_to_search(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content"], [{"course_title": "C", "lesson_number": 2}]
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="q", course_name="MCP", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="q", course_name="MCP", lesson_number=2
        )

    def test_passes_none_defaults_when_filters_omitted(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content"], [{"course_title": "C", "lesson_number": 1}]
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="q")

        mock_store.search.assert_called_once_with(
            query="q", course_name=None, lesson_number=None
        )

    def test_multiple_results_separated_by_double_newline(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Doc A", "Doc B"],
            [
                {"course_title": "Course", "lesson_number": 1},
                {"course_title": "Course", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool.execute(query="test")

        assert "\n\n" in result
        assert "Doc A" in result
        assert "Doc B" in result


# ---------------------------------------------------------------------------
# execute() — source tracking (last_sources)
# ---------------------------------------------------------------------------

class TestSourceTracking:
    def test_last_sources_set_after_successful_search(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "MCP Course", "lesson_number": 3}],
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        assert "MCP Course" in tool.last_sources[0]
        assert "Lesson 3" in tool.last_sources[0]

    def test_last_sources_wraps_in_anchor_when_lesson_link_exists(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool.execute(query="test")

        src = tool.last_sources[0]
        assert '<a href="https://example.com/lesson"' in src
        assert 'target="_blank"' in src

    def test_last_sources_plain_text_when_no_lesson_link(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content"],
            [{"course_title": "Course", "lesson_number": 2}],
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="test")

        assert tool.last_sources[0] == "Course - Lesson 2"

    def test_last_sources_tracks_multiple_results(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Content A", "Content B"],
            [
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="test")

        assert len(tool.last_sources) == 2

    def test_last_sources_not_updated_on_empty_results(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])
        tool.last_sources = ["stale source"]

        tool.execute(query="test")

        # _format_results is not called when results are empty,
        # so last_sources retains its previous value
        assert tool.last_sources == ["stale source"]

    def test_last_sources_not_updated_on_error(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty("error")
        tool.last_sources = ["stale source"]

        tool.execute(query="test")

        assert tool.last_sources == ["stale source"]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class TestToolDefinition:
    def test_tool_name(self, tool):
        assert tool.get_tool_definition()["name"] == "search_course_content"

    def test_query_is_required(self, tool):
        schema = tool.get_tool_definition()["input_schema"]
        assert "query" in schema["required"]

    def test_optional_filters_present(self, tool):
        props = tool.get_tool_definition()["input_schema"]["properties"]
        assert "course_name" in props
        assert "lesson_number" in props
