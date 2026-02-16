"""
Shodh-Memory Integrations

Optional adapters for popular LLM frameworks:
- LangChain: ShodhMemory class for ConversationChain, agents
- LlamaIndex: ShodhLlamaMemory for chat engines
- OpenAI Agents SDK: ShodhTools + ShodhSession for agents

Install extras:
    pip install shodh-memory[langchain]
    pip install shodh-memory[llamaindex]
    pip install shodh-memory[openai-agents]
    pip install shodh-memory[all]
"""

# Lazy imports to avoid requiring dependencies
def get_langchain_memory():
    """Get LangChain ShodhMemory class (requires langchain installed)"""
    from .langchain import ShodhMemory
    return ShodhMemory

def get_llamaindex_memory():
    """Get LlamaIndex ShodhLlamaMemory class (requires llama-index installed)"""
    from .llamaindex import ShodhLlamaMemory
    return ShodhLlamaMemory

def get_openai_agents_tools():
    """Get OpenAI Agents SDK ShodhTools class (requires openai-agents installed)"""
    from .openai_agents import ShodhTools
    return ShodhTools

def get_openai_agents_session():
    """Get OpenAI Agents SDK ShodhSession class (requires openai-agents installed)"""
    from .openai_agents import ShodhSession
    return ShodhSession

__all__ = [
    "get_langchain_memory",
    "get_llamaindex_memory",
    "get_openai_agents_tools",
    "get_openai_agents_session",
]
