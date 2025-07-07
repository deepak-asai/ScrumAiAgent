from enum import Enum
from typing import TypedDict, List, NotRequired, Annotated, Sequence, Union
from langgraph.graph.message import add_messages
from jira_service import Ticket
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class BotFlow(str, Enum):
    MAIN_BOT_FLOW = "main_bot_flow"
    TICKET_PROCESSING_FLOW = "ticket_processing_flow"

class MainBotPhase(str, Enum):
    NOT_STARTED = "main_bot_not_started"
    RESTARTED = "main_bot_restarted"
    IN_PROGRESS = "main_bot_in_progress"
    TICKET_CHOSEN = "main_bot_ticket_chosen"
    COMPLETED = "main_bot_completed"
    END_CONVERSATION = "main_bot_end_conversation"

class TicketProcessingBotPhase(str, Enum):
    NOT_STARTED = "ticket_processing_bot_not_started"
    IN_PROGRESS = "ticket_processing_bot_in_progress"
    COMPLETED = "ticket_processing_bot_completed"
    END_CONVERSATION = "ticket_processing_bot_end_conversation"

class SystemCommand(TypedDict):
    command: str
    args: NotRequired[dict]

class TicketProcessorAgentState(TypedDict):
    bot_flow: BotFlow
    bot_state: Union[MainBotPhase, TicketProcessingBotPhase]
    recently_processed_ticket_ids: NotRequired[List[str]]
    current_ticket: Ticket
    messages: Annotated[Sequence, add_messages]
    ticket_processing_messages: Annotated[Sequence, add_messages]


class TicketProcessorPhase(str, Enum):
    NOT_STARTED = "ticket_processor_not_started"
    IN_PROGRESS = "ticket_processor_in_progress"
    PROCEED_TO_NEXT_STAGE = "ticket_processor_proceed_to_next_stage"
    TOOLS_CALL = "ticket_processor_tools_call"
    COMPLETED = "ticket_processor_completed"
    END_CONVERSATION = "ticket_processor_end_conversation"

class TicketProcessorStage(TypedDict):
    node: str
    prompt: str
    phase: TicketProcessorPhase
    next_stage_id: NotRequired[int]  # ID of the next stage to proceed to
    messages: Annotated[Sequence, add_messages]

class ScrumAgentTicketProcessorState(TypedDict):
    basic_instruction: str
    current_stage: int
    stages: List[TicketProcessorStage]