"""
LogWizard Agent — the core agentic AI that analyses logs, finds root causes,
proposes solutions, and learns from every investigation.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from logwizard.config import settings
from logwizard.knowledge import KnowledgeBase
from logwizard.sources.base import LogSource
from logwizard.sources.file_source import FileSource
from logwizard.tools import ALL_TOOLS, init_tools

SYSTEM_PROMPT = """\
You are **LogWizard**, an expert Site Reliability Engineer AI agent.

Your mission is to help engineers investigate production incidents by analysing
application logs, identifying root causes, and proposing actionable solutions.

## Capabilities
You have access to tools that let you:
- Query logs from connected sources (Splunk, log files) by time range and keywords.
- Get statistical overviews of log data (counts by level, top sources).
- Search a knowledge base of previously-seen error patterns.
- Search past incident analyses for similar issues.
- Store newly learned error patterns and incident analyses to improve future investigations.

## Investigation Workflow
When a user asks "what happened?" or reports an issue, follow this process:

1. **Scope** — Clarify the time window and affected service/component if not provided.
2. **Gather** — Use `query_logs` and `query_recent_errors` to pull relevant log entries.
3. **Statistics** — Use `get_log_statistics` to understand the volume and distribution.
4. **Knowledge Lookup** — Use `search_known_errors` and `search_past_incidents` to check
   if this error pattern or incident type has been seen before.
5. **Analyse** — Examine the log entries for:
   - Error cascades (first error → downstream failures)
   - Timing correlations
   - Resource exhaustion patterns (OOM, connection pool, disk)
   - Deployment or config change indicators
   - External dependency failures
6. **Root Cause** — State the most likely root cause with supporting evidence from the logs.
7. **Solution** — Propose a concrete fix or, if uncertain, provide a clear starting point
   for further investigation with specific next steps.
8. **Learn** — Store new patterns and this incident analysis using `store_learned_pattern`
   and `store_incident_analysis` so future investigations are faster.

## Response Format
Structure your final answer as:

### Incident Summary
Brief description of what happened.

### Timeline
Key events in chronological order.

### Root Cause
Most likely cause with evidence.

### Recommended Action
Concrete steps to resolve or investigate further.

### Confidence Level
How confident you are (High / Medium / Low) and what additional data could improve the analysis.

## Rules
- Always ground your analysis in actual log data — never fabricate log entries.
- If logs are insufficient, say so and suggest what additional data is needed.
- Distinguish between correlation and causation.
- When storing learned patterns, be precise about the error signature.
- Do NOT store patterns you are unsure of as "actionable" — mark them accordingly.
"""


class LogWizardAgent:
    def __init__(
        self,
        log_source: LogSource | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self._kb = knowledge_base or KnowledgeBase()
        self._source = log_source or FileSource()

        # Wire up the tools module
        init_tools(self._source, self._kb)

        # Build LLM client — FAB (Azure OpenAI)
        self._llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
            temperature=settings.llm_temperature,
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )

    def chat(self, user_message: str) -> str:
        """Send a message and get the agent's full response."""
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=user_message)]}
        )
        # The last AI message is the final response
        ai_messages = [
            m for m in result["messages"] if hasattr(m, "content") and m.content
        ]
        return ai_messages[-1].content if ai_messages else "No response generated."

    def stream_chat(self, user_message: str):
        """Yield streamed token chunks for real-time output."""
        for chunk in self._agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            stream_mode="messages",
        ):
            msg, metadata = chunk
            if msg.content and metadata.get("langgraph_node") == "agent":
                yield msg.content

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._kb

    @property
    def log_source(self) -> LogSource:
        return self._source
