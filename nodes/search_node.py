"""
Search node for handling information lookup and web search requests.

Simple, focused implementation that:
- Uses Tavily API for web search
- Uses LLM for tool calling and response summarization
- Lets LLM handle all escalation decisions
- Trusts the router for proper routing
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .base_node import BaseNode, AgentState

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SearchNode(BaseNode):
    """Simple search node with LLM function calling and summarization."""
    
    def __init__(self):
        """Initialize Search node - fail fast if no search tools available."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            # Require Tavily API key - fail fast if not available
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY environment variable is required for SearchNode")
                
            # Initialize search tool
            search_tool = TavilySearchResults(max_results=5)
            search_tools = [search_tool]
            
            super().__init__(search_tools, "Search")
            
            # Initialize LLM with tools bound for function calling
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=False,
                max_tokens=1024,
                timeout=10
            ).bind_tools(search_tools)
            
            self.logger.info(f"Initialized Search node with Tavily API and function calling")
            
        except Exception as e:
            # Can't use self.logger before super().__init__ is called
            logger.error(f"Failed to initialize SearchNode: {e}")
            raise

    def should_handle_request(self, message: str) -> bool:
        """Determine if this node should handle the search/information request."""
        # Simple check - router should handle this properly
        message_lower = message.lower()
        search_keywords = [
            "search", "find", "look up", "what is", "who is", "when is", "where is",
            "weather", "news", "information", "tell me", "how to"
        ]
        return any(keyword in message_lower for keyword in search_keywords)

    def _detect_caller(self, messages: List) -> str:
        """Detect who called us based on message patterns."""
        # Look for agent instruction patterns in recent messages
        for message in reversed(messages[-3:]):
            if isinstance(message, HumanMessage) and message.content:
                content = message.content
                if content.startswith("Search for:") or "Search:" in content:
                    return "agent"
        return "router"

    async def handle_request(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle search requests - simple dispatch to LLM function calling."""
        user_message = self.get_last_user_message(state)
        if not user_message:
            return self.create_response([AIMessage(content="I didn't receive a clear search request.")])
        
        # Detect caller and delegate to LLM
        messages = state.get("messages", [])
        called_by = self._detect_caller(messages)
        
        return await self._handle_with_function_calling(user_message.content, state, called_by)

    async def _handle_with_function_calling(self, user_request: str, state: AgentState, called_by: str) -> Dict[str, Any]:
        """Handle search using LLM function calling with smart escalation."""
        import time
        
        search_start_time = time.time()
        self.logger.info(f"🔍 Starting search: {user_request[:50]}... (called by: {called_by})")
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Let LLM decide everything: tool calls, escalation, etc.
        system_prompt = f"""You are a search specialist. Today is {current_date}.

User request: "{user_request}"
Called by: {called_by}

Your options:
1. SEARCH: Use tavily_search_results_json tool for information lookup
2. ESCALATE: If this involves multiple domains (search + music/house control), respond with "ESCALATE: [reason]"

For escalation examples:
- "Search for music and play it" → ESCALATE: needs both search and music control
- "Find information about smart lights and turn them on" → ESCALATE: needs both search and house control

For pure search requests, always use the tool. Be intelligent about search queries."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_request)
        ]
        
        try:
            # Let LLM decide what to do
            llm_start_time = time.time()
            response = await self.llm.ainvoke(messages)
            llm_duration = time.time() - llm_start_time
            
            # Check if LLM decided to escalate
            if response.content and response.content.startswith("ESCALATE:"):
                reason = response.content.replace("ESCALATE:", "").strip()
                self.logger.info(f"🔄 LLM decided to escalate: {reason}")
                return self.create_escalation_response(
                    reason=reason,
                    domains=["search"],
                    original_request=user_request
                )
            
            # Check if LLM made tool calls
            if response.tool_calls:
                self.logger.info(f"🔧 LLM made {len(response.tool_calls)} tool calls")
                
                # Execute tools
                tool_start_time = time.time()
                tool_results = await self._execute_tool_calls(response.tool_calls, state)
                tool_duration = time.time() - tool_start_time
                
                # Handle response based on caller
                processing_start_time = time.time()
                if called_by == "agent":
                    # Agent needs raw results for further processing
                    raw_results = self._format_raw_results(tool_results, user_request)
                    result = self._return_raw_results_to_agent(raw_results, state)
                else:
                    # Router/Direct needs user-friendly summary
                    summary = await self._create_tts_friendly_summary(tool_results, user_request, user_request)
                    result = self.create_response([AIMessage(content=summary)])
                
                processing_duration = time.time() - processing_start_time
                total_duration = time.time() - search_start_time
                self.logger.info(f"📊 Search timing - LLM: {llm_duration:.3f}s, Tools: {tool_duration:.3f}s, Processing: {processing_duration:.3f}s, Total: {total_duration:.3f}s")
                return result
            else:
                # No tool calls - return LLM response
                total_duration = time.time() - search_start_time
                self.logger.info(f"⚡ Search completed without tools in {total_duration:.3f}s")
                content = response.content or "I wasn't sure how to search for that information."
                return self.create_response([AIMessage(content=content)])
                
        except Exception as e:
            total_duration = time.time() - search_start_time
            self.logger.error(f"❌ Search error after {total_duration:.3f}s: {e}")
            return self.create_response([AIMessage(content="I encountered an error while searching.")])

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], state: AgentState) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            # Find and execute tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                results.append({
                    "error": f"Tool {tool_name} not found",
                    "content": f"Error: Tool {tool_name} is not available"
                })
                continue
            
            try:
                result = await tool.ainvoke(tool_args) if hasattr(tool, 'ainvoke') else tool.invoke(tool_args)
                results.append({"tool_name": tool_name, "content": result})
            except Exception as e:
                self.logger.error(f"Error executing {tool_name}: {e}")
                results.append({
                    "error": str(e),
                    "content": f"Error executing {tool_name}: {str(e)}"
                })
        
        return results

    def _format_raw_results(self, tool_results: List[Dict[str, Any]], search_query: str) -> str:
        """Format raw results for agent processing."""
        if not tool_results:
            return f"Search query: {search_query}\nNo results found."
        
        formatted = f"Search query: {search_query}\n\nResults:\n"
        
        for i, result in enumerate(tool_results, 1):
            content = result.get("content", "")
            error = result.get("error")
            
            if error:
                formatted += f"\n{i}. Error: {error}\n"
                continue
                
            # Simple formatting - just include the content
            if isinstance(content, str) and content:
                formatted += f"\n{i}. {content[:500]}...\n" if len(content) > 500 else f"\n{i}. {content}\n"
        
        return formatted

    def _return_raw_results_to_agent(self, raw_results: str, state: AgentState) -> Dict[str, Any]:
        """Return raw results to agent for further processing."""
        new_state = dict(state)
        new_state["messages"] = state.get("messages", []) + [AIMessage(content=raw_results)]
        new_state["next_action"] = "continue"
        new_state["target_domain"] = "agent"
        return new_state

    async def _create_tts_friendly_summary(self, tool_results: List[Dict[str, Any]], user_request: str, search_query: str) -> str:
        """Create user-friendly summary using LLM."""
        if not tool_results:
            return "I couldn't find any information about that."
        
        # Get raw results - let LLM handle ALL parsing and cleaning
        raw_results = []
        for result in tool_results:
            content = result.get("content", "")
            error = result.get("error")
            
            if error:
                raw_results.append(f"Error: {error}")
            elif content:
                raw_results.append(str(content))
        
        if not raw_results:
            return f"I searched for information about {search_query} but couldn't find clear results."
        
        # Let LLM handle everything: JSON parsing, cleaning, summarization
        combined_raw = "\n\n".join(raw_results)
        
        system_prompt = f"""You are a voice assistant summarizing search results.

User asked: "{user_request}"

Raw search tool results (may contain JSON, URLs, markdown, etc.):
{combined_raw[:2000]}

Your job:
1. Parse any JSON data to extract meaningful information
2. Clean out URLs, markdown, encoding issues, and web artifacts
3. Create a natural, conversational response that directly answers the user's question
4. Keep it under 2-3 sentences, perfect for text-to-speech
5. Focus on the most relevant information (dates, locations, names, numbers)

Response:"""

        try:
            summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False, max_tokens=150)
            messages = [SystemMessage(content=system_prompt)]
            response = await summary_llm.ainvoke(messages)
            
            return response.content.strip() if response.content else "I found some information but couldn't extract a clear answer."
                
        except Exception as e:
            self.logger.error(f"Error in LLM summarization: {e}")
            return "I encountered an error while processing the search results." 