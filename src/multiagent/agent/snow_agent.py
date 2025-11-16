import os
import logging
import mlflow

from langchain_core.messages import AIMessage, SystemMessage

from ..util.constants import LLM_ENDPOINT_NAME
from ..service.tools import snow_tools
from ..util.agent_utils import safe_node_wrapper
from state import State
from databricks_langchain.chat_models import ChatDatabricks


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# ServiceNow Agent
snow_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)
snow_llm_with_tools = snow_llm.bind_tools(snow_tools)


@safe_node_wrapper
@mlflow.trace(name="servicenow_agent")
def servicenow_agent_node(state: State):
    """ServiceNow specialist agent for ticket management."""
    log.info("Entering servicenow_agent_node")
    system_message = SystemMessage(
        content="""You are a ServiceNow Integration Specialist for IT Service Management.

    Responsibilities:
    - Create and manage support tickets for all types of issues.
    - Handle incident management and service request processing.
    - Ensure each ticket has a clear title, detailed description, and appropriate priority.

    Always respond concisely, focusing on guiding the user through ticket creation and providing updates on existing tickets."""
    )

    messages = [system_message] + state["messages"]
    log.info(f" snow_agent calling LLM with {len(messages)} messages")
    try:
        response = snow_llm_with_tools.invoke(messages)
        log.info(f"snow_agent got response type: {type(response)}")
        log.info(
            f"snow_agent response content type: {type(getattr(response, 'content', None))}"
        )
    except Exception as e:
        log.error(f"sap_agent LLM invoke failed: {e}")
        raise

    # ENHANCED: Same robust defensive conversion
    if isinstance(response, AIMessage):
        log.info("snow_agent response is already AIMessage")
        return {"messages": [response]}
    elif isinstance(response, str):
        log.info("snow_agent converting string to AIMessage")
        response = AIMessage(content=response)
    else:
        log.info(f"snow_agent converting {type(response)} to AIMessage")
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        tool_calls = getattr(response, "tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        response = AIMessage(content=content, tool_calls=tool_calls)

    result = {"messages": [response]}
    log.info(
        f"snow_agent returning: {type(result)} with message type: {type(result['messages'][0])}"
    )
    return result
