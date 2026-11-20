"""The agentic layer: Claude chooses where to look, then a second model checks the answer."""

from rag_assistant.agent.check import SelfChecker
from rag_assistant.agent.loop import AgentConfig, ClaudeAgent, get_agent

__all__ = ("AgentConfig", "ClaudeAgent", "SelfChecker", "get_agent")
