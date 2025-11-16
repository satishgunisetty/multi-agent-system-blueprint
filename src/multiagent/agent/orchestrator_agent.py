import os
import logging
import mlflow

from src.multiagent.agent.state import State
from databricks_langchain.chat_models import ChatDatabricks
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from src.multiagent.util.constants import LLM_ENDPOINT_NAME, YES_KEYWORDS, NO_KEYWORDS
from src.multiagent.util.agent_utils import (
    safe_node_wrapper,
    detect_generic_help,
    detect_support_request,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Supervisor Agent - simple routing logic

# Initialize LLM
supervisor_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)


@safe_node_wrapper
@mlflow.trace(name="supervisor_agent")
def supervisor_node(state: State):
    """Supervisor: routes user requests to agents, and formats agent outputs."""
    log.info(" Entering supervisor_node")

    # Get current state values with defaults
    awaiting_ticket_confirmation = state.get("awaiting_ticket_confirmation", False)
    awaiting_ticket_description = state.get("awaiting_ticket_description", False)
    captured_issue_description = state.get("captured_issue_description", "")

    log.info(
        f"Current state - confirmation: {awaiting_ticket_confirmation}, description: {awaiting_ticket_description}"
    )

    log.info(f" State Schema: {state} ")
    last_msg = state["messages"][-1]
    log.info(f" supervisor last_msg type: {type(last_msg)}")

    # --- Case 1: Human user request ---
    if isinstance(last_msg, HumanMessage):
        log.info("supervisor processing HumanMessage/User message")
        user_message = last_msg.content

        # Case 1: Generic help request → explain capabilities
        if detect_generic_help(user_message):
            log.info(f"This is a generic message {user_message}")
            return {
                **state,
                "messages": [
                    AIMessage(
                        content=(
                            "I can assist you with Purchase Orders, Invoices, and other Source-to-Pay processes via SAP, "
                            "or help create ServiceNow tickets if you encounter issues or need support. "
                            "How would you like me to help today?"
                        )
                    )
                ],
            }

        # Case 2: Problem detected → offer ServiceNow ticket
        if (
            detect_support_request(user_message)
            and not state.get("awaiting_ticket_confirmation")
            and not state.get("awaiting_ticket_description")
        ):
            log.info(f"This is a support/problem request message {user_message}")
            log.info(f"Updated State Schema: {state} ")
            return {
                **state,
                "awaiting_ticket_confirmation": True,
                "messages": [
                    AIMessage(
                        content="It looks like you need support. I can create a ServiceNow ticket for you. Would you like me to create one? (Yes/No)"
                    )
                ],
            }

        # Step 3: Handle Yes/No response for ticket creation
        if state.get("awaiting_ticket_confirmation"):
            log.info(f"Expecting Yes/No from the user:  {user_message}")
            if any(kw in user_message.lower() for kw in YES_KEYWORDS):
                return {
                    **state,
                    "awaiting_ticket_confirmation": False,
                    "awaiting_ticket_description": True,
                    "messages": [
                        AIMessage(
                            content="Great! Please provide a short description of the issue so I can create the ServiceNow ticket."
                        )
                    ],
                }
            elif any(kw in user_message.lower() for kw in NO_KEYWORDS):
                return {
                    **state,
                    "awaiting_ticket_confirmation": False,
                    "messages": [
                        AIMessage(
                            content="No problem. Do you need any other information?"
                        )
                    ],
                }

        # Case 4: Capture issue description and route to ServiceNow
        if state.get("awaiting_ticket_description"):
            state["awaiting_ticket_description"] = False
            state["captured_issue_description"] = user_message

            user_message = HumanMessage(
                content=f"Create a ServiceNow ticket for this issue with the following description: '{user_message}'."
            )

        routing_prompt = f"""
        You are an intelligent routing supervisor for a Source-to-Pay (S2P) system. 
        Analyze the user request and route it to the most appropriate agent based on the content and intent.

        REQUEST: "{user_message}"

        ROUTING OPTIONS:

        **SAP_AGENT** - Handle requests related to:
        - Purchase Orders (PO creation, modification, status, approvals)
        - Invoices (processing, matching, disputes, payments)
        - Remittance advice and payment confirmations  
        - Vendor master data and supplier information
        - Contract execution and spend analytics
        - AP/AR processes and financial reconciliation

        **SERVICENOW_AGENT** - Handle requests for:
        - Incident creation for PO/Invoice issues
        - Service requests related to procurement processes
        - Workflow approvals and escalations
        - System access issues or technical problems
        - Change requests for S2P configurations
        - User support and troubleshooting

        **GREETING** - Handle conversational starters:
        - Greetings: "Hi", "Hello", "Good morning/afternoon"
        - General pleasantries and conversation openers
        - "How are you?" or similar social interactions

        **OUT_OF_SCOPE** - Route requests that are:
        - Unrelated to procurement, purchasing, or payment processes
        - General IT support not related to S2P systems
        - HR, legal, or other business functions outside procurement
        - Personal requests or non-business conversations

        ROUTING RULES:
        1. If the request mentions specific transaction numbers, amounts, or vendor names → likely SAP_AGENT
        2. If the request asks to "create a ticket" or mentions "issue" or "problem" → likely SERVICENOW_AGENT  
        3. When in doubt between SAP_AGENT and SERVICENOW_AGENT, default to SAP_AGENT for data queries, SERVICENOW_AGENT for problem resolution
        4. Be decisive - choose the single best option based on primary intent

        RESPONSE FORMAT:
        Provide ONLY the routing decision: SAP_AGENT, SERVICENOW_AGENT, GREETING, or OUT_OF_SCOPE
        """

        try:
            response = supervisor_llm.invoke([SystemMessage(content=routing_prompt)])
            agent_name = (response.content or "").strip().upper()
            log.info(f"supervisor agent_name: {agent_name}")
        except Exception as e:
            return {
                **state,
                "messages": [AIMessage(content="Error routing request. Please retry.")],
            }

        # Routing decision
        if "SAP" in agent_name:
            return {**state, "messages": [AIMessage(content="ROUTE_TO: SAP_AGENT")]}
        elif "SERVICENOW" in agent_name:
            return {
                **state,
                "messages": [AIMessage(content="ROUTE_TO: SERVICENOW_AGENT")],
            }
        elif "GREETING" in agent_name:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="Hello! How can I help with POs, invoices, or ServiceNow tickets today?"
                    )
                ],
            }
        elif "OUT_OF_SCOPE" in agent_name:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="Sorry, I only handle Source-to-Pay processes like POs, invoices, and ServiceNow tickets."
                    )
                ],
            }
        else:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="I’m not sure which agent should handle this. Can you clarify?"
                    )
                ],
            }

    # --- Case 2: Agent response (AIMessage / ToolMessage) ---
    elif isinstance(last_msg, (AIMessage, ToolMessage)):
        log.info(f"supervisor handling: Tool Message / AIMessage - {type(last_msg)}")
        msg_content = getattr(last_msg, "content", "")
        msg_content = msg_content if isinstance(msg_content, str) else str(msg_content)
        log.info(f"supervisor Message contenct - {msg_content}")
        final_prompt = f"""
        You are the supervisor. The following agent responded:

        {msg_content}

        Reformat this into a clear, user-friendly response with markdown(HTML). If you have list of items represent it as a table. If the response is too long, summarize it.
        End with: "Would you like me to take any further action?"
        """

        try:
            response = supervisor_llm.invoke([SystemMessage(content=final_prompt)])
            log.info(f"supervisor formated response- {response}")
            out_text = response.content or "Error formatting response."
        except Exception:
            out_text = msg_content  # fallback

        return {
            **state,
            "messages": [AIMessage(content=out_text)],
        }

    # --- Default fallback ---
    return {"messages": []}
