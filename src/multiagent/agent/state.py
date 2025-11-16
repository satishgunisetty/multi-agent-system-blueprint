from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    session_id: str
    awaiting_ticket_confirmation: bool
    awaiting_ticket_description: bool
    captured_issue_description: str
