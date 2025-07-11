from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from jira_service import JiraService, Ticket
from models import (
    BotFlow,
    MainBotPhase,
    TicketProcessorAgentState,
)
from helpers import deserialize_system_command
import json

def fetch_jira_tickets(user_id) -> list[Ticket]:
    """Fetch Jira tickets for a given user using Jira REST API."""
    service = JiraService.get_instance()
    return service.fetch_user_tickets(user_id, "APP")

def main_bot(agent_state: TicketProcessorAgentState, main_llm=None):
    tickets = fetch_jira_tickets("deepak.a.1996@gmail.com")
    tickets_str = json.dumps(tickets, indent=2)

    conversation_note = (
        """
        This is a continuation of a previous conversation. Continue helping the user with their tickets. Ask the user what is the next ticket they want to discuss about any other ticket or else if you could end the conversation. "
        """
        if agent_state['bot_state'] == MainBotPhase.RESTARTED
        else "This is a new conversation. Start by greeting the user and helping them choose a ticket. Give a small introduction about the bot and its purpose."
    )

    restarted_bot_prompt = f"""
    - Previous ticket discussion is complete. This is a continuation of the scrum meeting.
    - Ask the user which ticket they want to discuss next, or if they want to end the conversation.
    - Recently processed tickets: {agent_state["recently_processed_ticket_ids"] or []}
        (Show these at the end of the list. Do not mention them explicitly.)
    - If the user selects a recently discussed ticket, confirm if they want to continue with it or choose a different ticket.
    """

    prompt = f"""
    You are an agent conducting a scrum meeting. Speak as the user's manager. {conversation_note}

    Context:
    - You have a list of tickets assigned to the user.
    - Each ticket has: Ticket ID, Summary, Status, Priority, Start Date, Due Date.

    Follow all these instructions strictly. Do not skip any of them:
    - Show the list of tickets in this format:
        Ticket ID: <id>
        Summary: <summary>
        Status: <status>
        Priority: <priority>
        Start Date: <start_date>
        Due Date: <due_date>
    - Order the tickets as follows:
        1. Tickets with status "In Progress"
        2. Tickets with status "To Do"
        3. Recently processed tickets (show these last; do not mention them explicitly to the user)
    - {restarted_bot_prompt}
    - Ask the user to choose a ticket to discuss, or if they want to end the conversation.
    - If the user requests a ticket's description, provide it.
    - If the user selects a ticket, respond ONLY with this JSON (no extra text):
        {{
            "command": "ticket_chosen",
            "args": {{
                "ticket_id": "<ticket_id>"
            }}
        }}
        (Replace <ticket_id> with the actual ticket id.)
    - If the user does not select a ticket, continue the conversation.
    - If the user wants to end the conversation, respond ONLY with:
        {{
            "command": "end_conversation"
        }}

    Tickets:
    {tickets_str}
    """

    # restarted_bot_prompt = (
    #     f"""
    #     Previous ticket which the user chose has been discussed. This is a continuation of a previous conversation. Continue helping the user with their tickets. Ask the user what is the next ticket they want to discuss about any other ticket or else if you could end the conversation. "

    #     The user recently processed the following tickets: 
    #     {agent_state["recently_processed_ticket_ids"] or []}

    #     Show the recently processed tickets at the last. Do not tell about the recently discussed ticket to the user. Just start the conversation with to choose the next ticket or end the conversation.
        
    #     If the user again chooses a recently discussed ticket, confirm with the user that they want to continue with the same ticket or if they want to choose a different ticket.
    #     """
    # )
    

    breakpoint()  # For debugging purposes, remove in production


    if agent_state["bot_state"] == MainBotPhase.RESTARTED:
        agent_state["messages"] = []

    if agent_state["bot_state"] == MainBotPhase.NOT_STARTED or MainBotPhase.RESTARTED:
        agent_state["messages"].append(SystemMessage(content=prompt))
    
    if agent_state["bot_state"] in [MainBotPhase.NOT_STARTED, MainBotPhase.RESTARTED]:
        response = main_llm.invoke(agent_state["messages"])
        return {
            "bot_flow": BotFlow.MAIN_BOT_FLOW,
            "bot_state": MainBotPhase.IN_PROGRESS,
            "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
            "messages": list(agent_state["messages"]) + [response]
        }

    user_input = input("\n👤 User: ")
    agent_state["messages"].append(HumanMessage(content=user_input))

    response = main_llm.invoke(agent_state["messages"])
    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "ticket_chosen" and "ticket_id" in systemCommand["args"]:
            return {
                "bot_flow": BotFlow.TICKET_PROCESSING_FLOW,
                "bot_state": MainBotPhase.TICKET_CHOSEN,
                "current_ticket": next((t for t in tickets if t["id"] == systemCommand["args"]["ticket_id"]), None),
                "messages": list(agent_state["messages"]),
                "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
                "ticket_processing_messages": [response]
            }
        
        if systemCommand["command"] == "end_conversation":
            return {
                "bot_state": MainBotPhase.END_CONVERSATION,
            }
        
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "bot_flow": BotFlow.MAIN_BOT_FLOW,
        "bot_state": MainBotPhase.IN_PROGRESS,
        "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
        "messages": list(agent_state["messages"]) + [response]
    }