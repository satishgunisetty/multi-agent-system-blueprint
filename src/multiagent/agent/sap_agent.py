import os
import logging
import mlflow

from langchain_core.messages import AIMessage, SystemMessage

from ..util.constants import LLM_ENDPOINT_NAME
from ..service.tools import sap_tools
from ..util.agent_utils import safe_node_wrapper
from state import State
from databricks_langchain.chat_models import ChatDatabricks


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# SAP Agent
sap_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)
sap_llm_with_tools = sap_llm.bind_tools(tools=sap_tools)


@safe_node_wrapper
@mlflow.trace(name="sap_agent")
def sap_agent_node(state: State):
    """SAP specialist agent for PO and invoice operations."""
    log.info("Entering sap_agent_node")
    system_message = SystemMessage(
        content="""You are a SAP Integration Specialist for Source-to-Pay processes.

Your expertise:
- Purchase Order status inquiries and updates
- Invoice processing and rejection analysis  
- SAP system data retrieval

Use the available tools to get real-time SAP data. Provide clear, detailed responses with specific information."""
    )

    messages = [system_message] + state["messages"]
    log.info(f"sap_agent calling LLM with {len(messages)} messages")

    try:
        response = sap_llm_with_tools.invoke(messages)
        log.info(f"sap_agent got response type: {type(response)}")
        log.info(
            f"sap_agent response content type: {type(getattr(response, 'content', None))}"
        )
    except Exception as e:
        log.info(f"sap_agent LLM invoke failed: {e}")
        raise

    # ENHANCED: More robust defensive conversion
    if isinstance(response, AIMessage):
        log.info("sap_agent response is already AIMessage")
        # Already correct type
        return {"messages": [response]}
    elif isinstance(response, str):
        log.info("sap_agent converting string to AIMessage")
        response = AIMessage(content=response)
    else:
        # Handle any other response type more carefully
        log.info(f"sap_agent converting {type(response)} to AIMessage")
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
        f"sap_agent returning: {type(result)} with message type: {type(result['messages'][0])}"
    )
    return result
