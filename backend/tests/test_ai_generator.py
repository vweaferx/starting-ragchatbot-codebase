"""
Tests verifying that AIGenerator correctly invokes CourseSearchTool
via the tool-use flow and synthesizes the final response.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def make_text_response(text="The answer"):
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    return response


def make_tool_use_response(tool_name="search_course_content", tool_input=None, tool_id="tool_001"):
    tool_input = tool_input or {"query": "What is RAG?"}
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    response.content = [block]
    return response


@pytest.fixture
def generator():
    with patch("anthropic.Anthropic"):
        return AIGenerator(api_key="test_key", model="claude-test")


@pytest.fixture
def mock_tool_manager():
    mgr = MagicMock()
    mgr.execute_tool.return_value = "Search result: RAG stands for Retrieval-Augmented Generation."
    return mgr


# ---------------------------------------------------------------------------
# generate_response — basic routing
# ---------------------------------------------------------------------------

class TestGenerateResponseRouting:
    def test_returns_text_directly_on_end_turn(self, generator):
        generator.client.messages.create.return_value = make_text_response("Direct answer")

        result = generator.generate_response(query="What is AI?")

        assert result == "Direct answer"

    def test_passes_query_in_messages(self, generator):
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="Test query")

        kwargs = generator.client.messages.create.call_args[1]
        assert kwargs["messages"][0]["content"] == "Test query"

    def test_includes_tools_and_auto_choice_when_provided(self, generator):
        generator.client.messages.create.return_value = make_text_response()
        tools = [{"name": "search_course_content"}]

        generator.generate_response(query="q", tools=tools)

        kwargs = generator.client.messages.create.call_args[1]
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self, generator):
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="q")

        kwargs = generator.client.messages.create.call_args[1]
        assert "tools" not in kwargs

    def test_prepends_history_in_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(
            query="q", conversation_history="User: hi\nAssistant: hello"
        )

        kwargs = generator.client.messages.create.call_args[1]
        assert "Previous conversation:" in kwargs["system"]
        assert "User: hi" in kwargs["system"]

    def test_uses_base_system_prompt_when_no_history(self, generator):
        generator.client.messages.create.return_value = make_text_response()

        generator.generate_response(query="q")

        kwargs = generator.client.messages.create.call_args[1]
        assert kwargs["system"] == generator.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# generate_response — tool-use branch
# ---------------------------------------------------------------------------

class TestToolUseInvocation:
    def test_executes_tool_when_stop_reason_is_tool_use(self, generator, mock_tool_manager):
        generator.client.messages.create.side_effect = [
            make_tool_use_response(tool_input={"query": "What is RAG?"}),
            make_text_response("Synthesized answer"),
        ]

        result = generator.generate_response(
            query="What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="What is RAG?"
        )
        assert result == "Synthesized answer"

    def test_tool_result_included_in_followup_call(self, generator, mock_tool_manager):
        mock_tool_manager.execute_tool.return_value = "Found: RAG is a retrieval pattern"
        generator.client.messages.create.side_effect = [
            make_tool_use_response(tool_input={"query": "RAG"}, tool_id="id_001"),
            make_text_response("Final answer"),
        ]

        generator.generate_response(
            query="Explain RAG",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        # The last user message must carry the tool_result
        user_messages = [m for m in messages if m.get("role") == "user"]
        last_user = user_messages[-1]
        assert isinstance(last_user["content"], list)
        tool_result_block = last_user["content"][0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "id_001"
        assert tool_result_block["content"] == "Found: RAG is a retrieval pattern"

    def test_intermediate_call_includes_tools(self, generator, mock_tool_manager):
        tools = [{"name": "search_course_content"}]
        generator.client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response(),
        ]

        generator.generate_response(query="q", tools=tools, tool_manager=mock_tool_manager)

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert second_kwargs["tools"] == tools
        assert second_kwargs["tool_choice"] == {"type": "auto"}

    def test_synthesis_call_excludes_tools(self, generator, mock_tool_manager):
        generator.client.messages.create.side_effect = [
            make_tool_use_response(tool_id="t1"),
            make_tool_use_response(tool_name="get_course_outline", tool_input={"course_name": "RAG"}, tool_id="t2"),
            make_text_response("Final synthesis"),
        ]

        result = generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        third_kwargs = generator.client.messages.create.call_args_list[2][1]
        assert "tools" not in third_kwargs
        assert result == "Final synthesis"

    def test_system_prompt_preserved_in_followup_call(self, generator, mock_tool_manager):
        generator.client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response(),
        ]

        generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert second_kwargs["system"] == generator.SYSTEM_PROMPT

    def test_assistant_tool_use_message_added_before_result(self, generator, mock_tool_manager):
        tool_response = make_tool_use_response()
        generator.client.messages.create.side_effect = [
            tool_response,
            make_text_response(),
        ]

        generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        roles = [m["role"] for m in messages]
        # Conversation must follow: user → assistant → user (tool result)
        assert roles == ["user", "assistant", "user"]

    def test_two_round_tool_chain(self, generator, mock_tool_manager):
        mock_tool_manager.execute_tool.side_effect = ["Result A", "Result B"]
        generator.client.messages.create.side_effect = [
            make_tool_use_response(tool_name="search_course_content", tool_id="t1"),
            make_tool_use_response(tool_name="get_course_outline", tool_input={"course_name": "RAG"}, tool_id="t2"),
            make_text_response("Chained answer"),
        ]

        result = generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_tool_manager.execute_tool.call_count == 2
        assert generator.client.messages.create.call_count == 3
        assert result == "Chained answer"

    def test_early_exit_after_first_tool(self, generator, mock_tool_manager):
        generator.client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("Early exit answer"),
        ]

        result = generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Early exit answer"
        assert generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_tool_error_triggers_synthesis(self, generator, mock_tool_manager):
        mock_tool_manager.execute_tool.side_effect = RuntimeError("db down")
        generator.client.messages.create.side_effect = [
            make_tool_use_response(tool_id="err_001"),
            make_text_response("Fallback answer"),
        ]

        result = generator.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Fallback answer"
        # Synthesis call must not include tools
        last_kwargs = generator.client.messages.create.call_args_list[-1][1]
        assert "tools" not in last_kwargs
        # Tool result block must carry is_error=True
        synthesis_messages = last_kwargs["messages"]
        user_msgs = [m for m in synthesis_messages if m.get("role") == "user"]
        tool_result_block = user_msgs[-1]["content"][0]
        assert tool_result_block.get("is_error") is True
        assert tool_result_block["tool_use_id"] == "err_001"


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------

class TestExtractResponseText:
    def test_extracts_text_from_text_block(self, generator):
        result = generator._extract_response_text(make_text_response("Hello world"))
        assert result == "Hello world"

    def test_returns_fallback_on_empty_content_list(self, generator):
        response = MagicMock()
        response.content = []
        assert "No response" in generator._extract_response_text(response)

    def test_returns_fallback_when_content_is_none(self, generator):
        response = MagicMock()
        response.content = None
        assert "No response" in generator._extract_response_text(response)
