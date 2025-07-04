from enum import Enum
from typing import TypedDict, List, NotRequired, Annotated, Sequence
from langgraph.graph.message import add_messages
from jira_service import Ticket

class BotState(str, Enum):
    NOT_STARTED = "not_started"
    TICKET_CHOSEN = "ticket_chosen"
    IN_PROGRESS = "in_progress"
    TICKET_PROCESSING_COMPLETED = "ticket_processing_completed"
    END_CONVERSATION = "end_conversation"

class BotFlow(str, Enum):
    MAIN_BOT_FLOW = "main_bot_flow"
    TICKET_PROCESSING_FLOW = "ticket_processing_flow"

class SystemCommand(TypedDict):
    command: str
    args: NotRequired[dict]

class TicketProcessorAgentState(TypedDict):
    botFlow: BotFlow
    all_tickets: List[Ticket]
    recently_processed_ticket_ids: NotRequired[List[str]]  # Optional field for recently processed ticket IDs
    current_ticket: Ticket
    processed_tickets: List[Ticket]  
    is_bot_starting: bool
    is_bot_conversation_continued: bool
    ticket_processing_state: BotState
    messages: Annotated[Sequence, add_messages]
    ticket_processing_messages: Annotated[Sequence, add_messages]