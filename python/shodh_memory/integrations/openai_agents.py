"""
OpenAI Agents SDK Integration for Shodh-Memory

Provides FunctionTool-based tools and Session persistence for the OpenAI Agents SDK.
Gives agents persistent memory, semantic recall, and GTD task management — no LLM calls
for memory ops, sub-millisecond retrieval, Hebbian learning.

Usage (Tools):
    from shodh_memory.integrations.openai_agents import ShodhTools
    from agents import Agent, Runner

    tools = ShodhTools(user_id="agent-1", api_key="your-key")
    agent = Agent(
        name="memory-agent",
        instructions="You have persistent memory. Use shodh_remember to store and shodh_recall to retrieve.",
        tools=tools.as_list(),
    )
    result = Runner.run_sync(agent, "Remember that I prefer Python over JavaScript")

Usage (Session):
    from shodh_memory.integrations.openai_agents import ShodhSession
    from agents import Agent, Runner

    session = ShodhSession(user_id="agent-1", api_key="your-key")
    agent = Agent(name="my-agent", instructions="You are a helpful assistant.")
    result = await Runner.run(agent, "My name is Alice", session=session)
    result = await Runner.run(agent, "What's my name?", session=session)
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

try:
    from agents import FunctionTool
    from agents.tool import ToolContext
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK is required for this integration. "
        "Install with: pip install shodh-memory[openai-agents] "
        "or: pip install openai-agents pydantic"
    )

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Pydantic argument models (generate JSON Schema for FunctionTool)
# ---------------------------------------------------------------------------

class RememberArgs(BaseModel):
    content: str = Field(description="The content to remember")
    memory_type: str = Field(
        default="Observation",
        description="Memory type: Observation, Decision, Learning, Error, Discovery, Pattern, Context, Task, Conversation",
    )
    tags: List[str] = Field(default_factory=list, description="Optional tags for categorization")


class RecallArgs(BaseModel):
    query: str = Field(description="Natural language search query")
    limit: int = Field(default=5, ge=1, le=50, description="Max results to return")


class ForgetArgs(BaseModel):
    memory_id: str = Field(description="The memory ID to delete (UUID or short prefix)")


class ContextSummaryArgs(BaseModel):
    max_items: int = Field(default=5, ge=1, le=20, description="Max items per category")
    include_decisions: bool = Field(default=True, description="Include recent decisions")
    include_learnings: bool = Field(default=True, description="Include recent learnings")
    include_context: bool = Field(default=True, description="Include project context")


class ProactiveContextArgs(BaseModel):
    context: str = Field(description="Current conversation context or topic")
    max_results: int = Field(default=5, ge=1, le=20, description="Max memories to surface")
    semantic_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Min similarity threshold")


class AddTodoArgs(BaseModel):
    content: str = Field(description="What needs to be done")
    project: str = Field(default="", description="Project name (created if doesn't exist)")
    priority: str = Field(default="medium", description="Priority: urgent, high, medium, low, none")
    contexts: List[str] = Field(default_factory=list, description="Contexts like @computer, @phone")
    due_date: str = Field(default="", description="Due date: ISO format, 'today', 'tomorrow', etc.")


class ListTodosArgs(BaseModel):
    status: List[str] = Field(
        default_factory=lambda: ["todo", "in_progress"],
        description="Filter by status: backlog, todo, in_progress, blocked, done, cancelled",
    )
    project: str = Field(default="", description="Filter by project name")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")


class CompleteTodoArgs(BaseModel):
    todo_id: str = Field(description="Todo ID or short prefix (e.g., 'SHO-1a2b')")


# ---------------------------------------------------------------------------
# HTTP client mixin
# ---------------------------------------------------------------------------

class _ShodhHTTPClient:
    """Shared HTTP client setup for tools and session."""

    def __init__(
        self,
        server_url: str = "http://localhost:3030",
        user_id: str = "default",
        api_key: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.user_id = user_id
        self.api_key = api_key or os.environ.get("SHODH_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set SHODH_API_KEY env var."
            )

        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    def _post(self, path: str, payload: dict, timeout: int = 10) -> dict:
        try:
            resp = self._session.post(
                f"{self.server_url}{path}",
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def _get(self, path: str, timeout: int = 10) -> dict:
        try:
            resp = self._session.get(
                f"{self.server_url}{path}",
                headers=self._headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def _delete(self, path: str, timeout: int = 10) -> dict:
        try:
            resp = self._session.delete(
                f"{self.server_url}{path}",
                headers=self._headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            if resp.content:
                return resp.json()
            return {"status": "deleted"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# ShodhTools — FunctionTool provider for OpenAI Agents SDK
# ---------------------------------------------------------------------------

class ShodhTools(_ShodhHTTPClient):
    """OpenAI Agents SDK tools backed by Shodh-Memory.

    Provides 8 tools for persistent memory and task management:
    - shodh_remember: Store memories with types and tags
    - shodh_recall: Semantic/hybrid search across memories
    - shodh_forget: Delete a specific memory
    - shodh_context_summary: Get condensed overview of recent learnings/decisions
    - shodh_proactive_context: Auto-surface relevant memories for current context
    - shodh_add_todo: Create tasks with projects, priorities, contexts
    - shodh_list_todos: List tasks filtered by status/project
    - shodh_complete_todo: Mark a task as complete

    Usage:
        from shodh_memory.integrations.openai_agents import ShodhTools
        from agents import Agent, Runner

        tools = ShodhTools(user_id="agent-1", api_key="your-key")
        agent = Agent(name="my-agent", tools=tools.as_list())
        result = Runner.run_sync(agent, "What do you remember about me?")
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3030",
        user_id: str = "default",
        api_key: Optional[str] = None,
        max_memories: int = 5,
        retrieval_mode: str = "hybrid",
    ):
        super().__init__(server_url=server_url, user_id=user_id, api_key=api_key)
        self.max_memories = max_memories
        self.retrieval_mode = retrieval_mode

    def as_list(self) -> List[FunctionTool]:
        """Return all tools as a list for Agent(tools=...)."""
        return [
            self._make_remember_tool(),
            self._make_recall_tool(),
            self._make_forget_tool(),
            self._make_context_summary_tool(),
            self._make_proactive_context_tool(),
            self._make_add_todo_tool(),
            self._make_list_todos_tool(),
            self._make_complete_todo_tool(),
        ]

    # -- Tool factories --

    def _make_remember_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = RememberArgs.model_validate_json(args)
            result = parent._post("/api/remember", {
                "user_id": parent.user_id,
                "content": parsed.content,
                "memory_type": parsed.memory_type,
                "tags": parsed.tags,
            })
            return json.dumps(result)

        return FunctionTool(
            name="shodh_remember",
            description="Store a memory that persists across sessions. Use for important facts, decisions, user preferences, or learnings.",
            params_json_schema=RememberArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_recall_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = RecallArgs.model_validate_json(args)
            result = parent._post("/api/recall", {
                "user_id": parent.user_id,
                "query": parsed.query,
                "limit": parsed.limit,
                "mode": parent.retrieval_mode,
            })
            return json.dumps(result)

        return FunctionTool(
            name="shodh_recall",
            description="Search persistent memory using natural language. Returns semantically similar memories ranked by relevance.",
            params_json_schema=RecallArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_forget_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = ForgetArgs.model_validate_json(args)
            result = parent._delete(f"/api/memory/{parsed.memory_id}")
            return json.dumps(result)

        return FunctionTool(
            name="shodh_forget",
            description="Delete a specific memory by its ID. Use when information is outdated or incorrect.",
            params_json_schema=ForgetArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_context_summary_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = ContextSummaryArgs.model_validate_json(args)
            result = parent._post("/api/context_summary", {
                "user_id": parent.user_id,
                "max_items": parsed.max_items,
                "include_decisions": parsed.include_decisions,
                "include_learnings": parsed.include_learnings,
                "include_context": parsed.include_context,
            })
            return json.dumps(result)

        return FunctionTool(
            name="shodh_context_summary",
            description="Get a condensed summary of recent decisions, learnings, and project context from memory.",
            params_json_schema=ContextSummaryArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_proactive_context_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = ProactiveContextArgs.model_validate_json(args)
            result = parent._post("/api/relevant", {
                "user_id": parent.user_id,
                "context": parsed.context,
                "max_results": parsed.max_results,
                "semantic_threshold": parsed.semantic_threshold,
            })
            return json.dumps(result)

        return FunctionTool(
            name="shodh_proactive_context",
            description="Surface memories relevant to the current conversation context. Uses entity matching and semantic similarity.",
            params_json_schema=ProactiveContextArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_add_todo_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = AddTodoArgs.model_validate_json(args)
            payload: dict[str, Any] = {
                "user_id": parent.user_id,
                "content": parsed.content,
                "priority": parsed.priority,
            }
            if parsed.project:
                payload["project"] = parsed.project
            if parsed.contexts:
                payload["contexts"] = parsed.contexts
            if parsed.due_date:
                payload["due_date"] = parsed.due_date
            result = parent._post("/api/todos/add", payload)
            return json.dumps(result)

        return FunctionTool(
            name="shodh_add_todo",
            description="Create a task/todo with optional project, priority, contexts, and due date. Follows GTD methodology.",
            params_json_schema=AddTodoArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_list_todos_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = ListTodosArgs.model_validate_json(args)
            payload: dict[str, Any] = {
                "user_id": parent.user_id,
                "status": parsed.status,
                "limit": parsed.limit,
            }
            if parsed.project:
                payload["project"] = parsed.project
            result = parent._post("/api/todos", payload)
            return json.dumps(result)

        return FunctionTool(
            name="shodh_list_todos",
            description="List tasks/todos filtered by status and project. Returns tasks with IDs, priorities, and status.",
            params_json_schema=ListTodosArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )

    def _make_complete_todo_tool(self) -> FunctionTool:
        parent = self

        async def invoke(ctx: ToolContext, args: str) -> str:
            parsed = CompleteTodoArgs.model_validate_json(args)
            result = parent._post("/api/todos/complete", {
                "user_id": parent.user_id,
                "todo_id": parsed.todo_id,
            })
            return json.dumps(result)

        return FunctionTool(
            name="shodh_complete_todo",
            description="Mark a task as complete. For recurring tasks, automatically creates the next occurrence.",
            params_json_schema=CompleteTodoArgs.model_json_schema(),
            on_invoke_tool=invoke,
        )


# ---------------------------------------------------------------------------
# ShodhSession — Session protocol implementation
# ---------------------------------------------------------------------------

class ShodhSession(_ShodhHTTPClient):
    """OpenAI Agents SDK Session backed by Shodh-Memory.

    Persists conversation history through shodh-memory's cognitive storage.
    Memories benefit from Hebbian learning, activation decay, and semantic
    consolidation — not just raw storage.

    Usage:
        from shodh_memory.integrations.openai_agents import ShodhSession
        from agents import Agent, Runner

        session = ShodhSession(user_id="agent-1", api_key="your-key")
        result = await Runner.run(agent, "My name is Alice", session=session)
        # Later, in a new run:
        result = await Runner.run(agent, "What's my name?", session=session)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3030",
        user_id: str = "default",
        api_key: Optional[str] = None,
        session_tag: str = "openai-agents-session",
    ):
        super().__init__(server_url=server_url, user_id=user_id, api_key=api_key)
        self.session_tag = session_tag

    async def get_items(self, limit: int = 100) -> list:
        """Retrieve recent conversation items from memory."""
        result = self._post("/api/recall/tags", {
            "user_id": self.user_id,
            "tags": [self.session_tag],
            "limit": limit,
        })
        if "error" in result:
            return []

        items = []
        memories = result if isinstance(result, list) else result.get("memories", [])
        for mem in memories:
            content = mem.get("content", "")
            items.append({
                "role": mem.get("metadata", {}).get("role", "user"),
                "content": content,
            })
        return items

    async def add_items(self, items: list) -> None:
        """Store conversation items as memories."""
        memories = []
        for item in items:
            role = "assistant"
            content = ""

            if isinstance(item, dict):
                role = item.get("role", "assistant")
                content = item.get("content", "")
            elif isinstance(item, str):
                content = item
            else:
                content = str(item)

            if not content:
                continue

            memories.append({
                "content": content,
                "memory_type": "Conversation",
                "tags": [self.session_tag, f"role:{role}"],
                "metadata": {"role": role},
            })

        if memories:
            self._post("/api/remember/batch", {
                "user_id": self.user_id,
                "memories": memories,
            }, timeout=20)

    async def pop_item(self) -> Any:
        """Remove and return the most recent conversation item."""
        result = self._post("/api/recall/tags", {
            "user_id": self.user_id,
            "tags": [self.session_tag],
            "limit": 1,
        })

        if "error" in result:
            return None

        memories = result if isinstance(result, list) else result.get("memories", [])
        if not memories:
            return None

        mem = memories[0]
        memory_id = mem.get("id", mem.get("memory_id", ""))
        if memory_id:
            self._delete(f"/api/memory/{memory_id}")

        return {
            "role": mem.get("metadata", {}).get("role", "user"),
            "content": mem.get("content", ""),
        }

    async def clear_session(self) -> None:
        """Clear all conversation memories for this session."""
        result = self._post("/api/recall/tags", {
            "user_id": self.user_id,
            "tags": [self.session_tag],
            "limit": 1000,
        })

        if "error" in result:
            return

        memories = result if isinstance(result, list) else result.get("memories", [])
        for mem in memories:
            memory_id = mem.get("id", mem.get("memory_id", ""))
            if memory_id:
                self._delete(f"/api/memory/{memory_id}")
