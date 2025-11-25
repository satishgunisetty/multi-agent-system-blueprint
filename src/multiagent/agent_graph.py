import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.multiagent.service.postgre_connection_manager import TokenRotatingPostgresSaver
from src.multiagent.agent.state import State
from src.multiagent.agent.orchestrator_agent import supervisor_node
from src.multiagent.agent.sap_agent import sap_agent_node
from src.multiagent.agent.snow_agent import servicenow_agent_node
from src.multiagent.agent.web_search_agent import web_agent_node
from src.multiagent.service.tools import sap_tools, snow_tools
from src.multiagent.service.web_tools import web_search_tools

############################################
#           Simple Routing Logic           #
############################################

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def route_to_agent(
    state: State,
) -> Literal["sap_agent", "servicenow_agent", "supervisor"]:
    """Simple routing based on supervisor decision."""

    # Get the last message from supervisor
    last_message = state["messages"][-1]

    if hasattr(last_message, "content") and "ROUTE_TO:" in last_message.content:
        route_decision = last_message.content.replace("ROUTE_TO:", "").strip()

        if "SAP_AGENT" in route_decision:
            return "sap_agent"
        elif "SERVICENOW_AGENT" in route_decision:
            return "servicenow_agent"
        elif "WEB_AGENT" in route_decision:
            return "web_agent"

    return "supervisor"


############################################
#        LakeBase PostGre Connection       #
############################################

# Global instance following Databricks patterns
_token_rotating_saver = None


def get_robust_checkpointer():
    """Get production-ready checkpointer with token rotation."""
    global _token_rotating_saver

    if _token_rotating_saver is None:
        _token_rotating_saver = TokenRotatingPostgresSaver()

    return _token_rotating_saver.get_checkpointer()


############################################
#        Simple Multi-Agent Graph          #
############################################


def build_simple_multiagent_system():
    """Build a clean, simple multi-agent system."""

    # Create workflow
    state_graph = StateGraph(State)

    # --- Nodes ---
    state_graph.add_node("supervisor", supervisor_node)
    state_graph.add_node("sap_agent", sap_agent_node)
    state_graph.add_node("servicenow_agent", servicenow_agent_node)
    state_graph.add_node("web_agent", web_agent_node)

    # Tool nodes
    state_graph.add_node("sap_tools", ToolNode(tools=sap_tools))
    state_graph.add_node("servicenow_tools", ToolNode(tools=snow_tools))
    state_graph.add_node("web_tools", ToolNode(tools=web_search_tools))

    # --- Edges ---
    # Start → Supervisor
    state_graph.add_edge(START, "supervisor")

    # Supervisor → Agent (based on routing)
    state_graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "sap_agent": "sap_agent",
            "servicenow_agent": "servicenow_agent",
            "web_agent": "web_agent",
            "supervisor": END,
        },
    )

    # SAP Agent → Tools OR return to Supervisor
    state_graph.add_conditional_edges(
        "sap_agent",
        tools_condition,
        {
            "tools": "sap_tools",  # goes to tools
            "__end__": "supervisor",  # no tool → direct to supervisor
        },
    )

    # ServiceNow Agent → Tools OR return to Supervisor
    state_graph.add_conditional_edges(
        "servicenow_agent",
        tools_condition,
        {
            "tools": "servicenow_tools",
            "__end__": "supervisor",
        },
    )

    # ServiceNow Agent → Tools OR return to Supervisor
    state_graph.add_conditional_edges(
        "web_agent",
        tools_condition,
        {
            "tools": "web_tools",
            "__end__": "supervisor",
        },
    )

    state_graph.add_edge("sap_tools", "supervisor")
    state_graph.add_edge("servicenow_tools", "supervisor")
    state_graph.add_edge("web_tools", "supervisor")

    # Get the persistent checkpointer
    memory = get_robust_checkpointer()

    log.info("Compiling graph with resilient PostgresSaver")
    return state_graph.compile(checkpointer=memory)
