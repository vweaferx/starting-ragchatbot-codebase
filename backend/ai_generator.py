import anthropic
from typing import List, Optional

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information.

Tool Usage:
- Use `get_course_outline` for any question about a course outline, structure, lesson list, or what lessons a course contains. Return the course title, course link, and every lesson number and title.
- Use `search_course_content` for questions about specific course content or detailed educational materials
- **Up to 2 sequential tool calls per query** — use a second tool only when the first result reveals the need for additional lookup
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline or structure questions**: Use `get_course_outline`, then present the full lesson list
- **Course-specific content questions**: Use `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, messages, system_content, tools, tool_manager)

        return self._extract_response_text(response)
    
    def _extract_response_text(self, response) -> str:
        """Safely extract the text from a Claude response object."""
        content = getattr(response, "content", None)
        if not content:
            return "No response content returned by AI."

        # Some SDK responses wrap the assistant text as the first content block.
        try:
            first_block = content[0]
        except (IndexError, TypeError):
            return "No response content returned by AI."

        if hasattr(first_block, "text") and first_block.text is not None:
            return first_block.text
        if isinstance(first_block, str):
            return first_block
        return str(first_block)

    def _run_tool_loop(self, initial_response, messages, system_content, tools, tool_manager, max_rounds=2):
        messages = list(messages)
        current_response = initial_response

        for round_num in range(max_rounds):
            messages.append({"role": "assistant", "content": current_response.content})

            tool_results = []
            tool_error = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool error: {str(e)}",
                            "is_error": True,
                        })
                        tool_error = True
                        break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if tool_error:
                break

            if round_num < max_rounds - 1:
                next_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content,
                    "tools": tools,
                    "tool_choice": {"type": "auto"},
                }
                next_response = self.client.messages.create(**next_params)
                if next_response.stop_reason == "end_turn":
                    return self._extract_response_text(next_response)
                if next_response.stop_reason == "tool_use":
                    current_response = next_response
                    continue
                break

        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_response_text(final_response)