from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from models import (
    BotFlow,
    MainBotPhase,
    TicketProcessingBotPhase,
    TicketProcessorAgentState,
)
from helpers import deserialize_system_command
from tools import current_date
import json
from datetime import datetime
from jira_service import Ticket


def get_due_soon_note(ticket: Ticket) -> str:
    due_date_str = ticket.get("due_date") or ticket.get("duedate")
    today = datetime.strptime(current_date.invoke({}), "%Y-%m-%d").date()
    if due_date_str:
        try:
            due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
            if (due_date - today).days <= 2:
                return (
                    f'Note: The due date for this ticket is {due_date_str}, which is approaching soon. '
                    'Ask the user if this due date is acceptable and if they will be able to complete the ticket on time. '
                    'If not, prompt the user to provide a new due date and offer to update it using the tool.'
                )
        except Exception:
            pass
    return ""

def is_last_message_tool_call(agent_state: TicketProcessorAgentState) -> bool:
    return agent_state["ticket_processing_messages"] and isinstance(agent_state["ticket_processing_messages"][-1], ToolMessage)

def ticket_processing_bot(agent_state: TicketProcessorAgentState, llm):
    ticket = agent_state["current_ticket"]
    ticket_status = ticket.get("status", "").lower() if ticket else ""

    if ticket_status == "to do":
        status_note = (
            "- The ticket is in 'To Do' status. You should help the user with all basic information about the ticket, "
            "such as description, priority, and any clarifications. "
            "- Ask the user what their plan is for the day regarding this ticket. "
            "- Guide the user step by step through the process of starting work on this ticket."
        )
    elif ticket_status == "in progress":
        status_note = (
            "The ticket is already 'In Progress'. You can skip basic information about the ticket. "
            "Focus on asking for updates, proof of work (like PR or code commit), blockers, and next steps. "
            "Do not repeat basic ticket details unless the user asks."
        )
    else:
        status_note = (
            "Handle the ticket appropriately based on its status. If unsure, ask the user how they want to proceed."
        )

    prompt = f"""
    You are an agent to conduct a scrum meeting. You have to sound like the user's manager. Do not start with any greeting or introduction.
    The user has chosen to work on this ticket:

    {json.dumps(ticket, indent=2)}

    {status_note}

    Follow all these instructions strictly. Do not skip any of them:
    - Ask only one question at a time. Wait for the user's response before asking the next question.
    - If the user wants to update the ticket status, use the following transition IDs:
    "To Do": "11", "In Progress": "21", "Done": "31", "Blocked": "2"
    - You should also ask the user if they want to update the status of the ticket or add a comment to it.
    - Confirm with the user before updating the status of the ticket
    - Whenever the status is set to "In Progress", you MUST call the tool to update the start date to today's date (YYYY-MM-DD).**
    - Ask the user for a due date if it is not already set, and offer to update it using the tool.
    - {get_due_soon_note(ticket) or ""}
    - You can use tools to update status, fetch comments, or add comments.
    - If the user is done with this ticket or if the user says he is wants to work on someother ticket, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "ticket_processing_done"
    }}
    - If the users chooses to end the conversation, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "end_conversation"
    }}
    - Otherwise, continue the conversation as the scrum manager.
    - ** After the conversation about this ticket is finished, create a summary of what was discussed and show it to the user. Check if there is any important information that should be added to the ticket as a comment. Ask the user if you should add this summary as a comment to the ticket. Only add the summary as a comment if the user says yes. **

    Remember: Do not simulate user responses. Only ask or answer as the agent.
    """

    if agent_state["bot_state"] == MainBotPhase.TICKET_CHOSEN:
        agent_state["ticket_processing_messages"].append(SystemMessage(content=prompt))

    if agent_state["bot_state"] == MainBotPhase.TICKET_CHOSEN or is_last_message_tool_call(agent_state):
        response = llm.invoke(agent_state["ticket_processing_messages"])
        return {
            "bot_flow": BotFlow.TICKET_PROCESSING_FLOW,
            "bot_state": TicketProcessingBotPhase.IN_PROGRESS,
            "current_ticket": agent_state["current_ticket"],
            "messages": agent_state["messages"],
            "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
            "ticket_processing_messages": list(agent_state["ticket_processing_messages"]) + [response]
        }

    user_input = input("\n👤 User: ")
    agent_state["ticket_processing_messages"].append(HumanMessage(content=user_input))

    response = llm.invoke(agent_state["ticket_processing_messages"])

    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "ticket_processing_done":
            # If the user is done with the ticket processing, we can end the conversation
            return {
                "bot_flow": BotFlow.MAIN_BOT_FLOW,
                "bot_state": MainBotPhase.RESTARTED,
                "current_ticket": agent_state["current_ticket"],
                "recently_processed_ticket_ids": list(set((agent_state.get("recently_processed_ticket_ids") or []) + [agent_state["current_ticket"]["id"]])),
                "messages": [],
                "ticket_processing_messages": []
            }
        
        if systemCommand["command"] == "end_conversation":
            # If the user wants to end the conversation, we can end the conversation
            return {
                "ticket_processing_state": TicketProcessingBotPhase.END_CONVERSATION,
            }
        
    except (json.JSONDecodeError, TypeError):
        pass
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "bot_flow": BotFlow.TICKET_PROCESSING_FLOW,
        "bot_state": TicketProcessingBotPhase.IN_PROGRESS,
        "current_ticket": agent_state["current_ticket"],
        "messages": agent_state["messages"],
        "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
        "ticket_processing_messages": list(agent_state["ticket_processing_messages"]) + [response]
    }