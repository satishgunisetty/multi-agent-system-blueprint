import os
import logging
import mlflow

from langchain_core.messages import AIMessage, SystemMessage

from src.multiagent.util.constants import LLM_ENDPOINT_NAME
from src.multiagent.service.web_tools import web_search_tools
from src.multiagent.util.agent_utils import safe_node_wrapper
from src.multiagent.agent.state import State
from databricks_langchain.chat_models import ChatDatabricks


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# Web Search Agent
web_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)
web_llm_with_tools = web_llm.bind_tools(tools=web_search_tools)


@safe_node_wrapper
@mlflow.trace(name="web_agent")
def web_agent_node(state: State):
    """Web Search Agent Node."""
    log.info("Entering web_agent_node")
    system_message = SystemMessage(
        content="""You are a WebSearch Agent.
Your purpose:
- Search the public web for up-to-date information
- Retrieve facts, explanations, definitions, and external knowledge
- Summarize and synthesize web results into clear and accurate answers

Guidelines:
- Always use the provided web-search tools to find information.
- If the user asks for knowledge that is not in your local context, perform a search.
- Provide concise, factual, and well-structured responses.
- If no reliable information is found, say so clearly rather than guessing.

Do NOT:
- Invent facts or fabricate URLs.
- Attempt to answer questions without searching when the answer depends on external information.

You are a specialized agent for web discovery and external research. Use your tools to gather evidence and deliver accurate findings.
"""
    )

    messages = [system_message] + state["messages"]
    log.info(f"web_agent calling LLM with {len(messages)} messages")

    try:
        response = web_llm_with_tools.invoke(messages)
        log.info(f"web_agent got response type: {type(response)}")
        log.info(
            f"web_agent response content type: {type(getattr(response, 'content', None))}"
        )
    except Exception as e:
        log.info(f"web_agent LLM invoke failed: {e}")
        raise

    # ENHANCED: More robust defensive conversion
    if isinstance(response, AIMessage):
        log.info("web_agent response is already AIMessage")
        # Already correct type
        return {"messages": [response]}
    elif isinstance(response, str):
        log.info("web_agent converting string to AIMessage")
        response = AIMessage(content=response)
    else:
        # Handle any other response type more carefully
        log.info(f"web_agent converting {type(response)} to AIMessage")
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        tool_calls = getattr(response, "tool_calls", [])
        # Ensure tool_calls is a list
        if not isinstance(tool_calls, list):
            tool_calls = []

        response = AIMessage(content=content, tool_calls=tool_calls)

    result = {"messages": [response]}
    log.info(
        f"web_agent returning: {type(result)} with message type: {type(result['messages'][0])}"
    )
    return result
