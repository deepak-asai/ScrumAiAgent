from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from models import (
    BotFlow,
    MainBotPhase,
    TicketProcessingBotPhase,
    TicketProcessorAgentState,
)
from helpers import deserialize_system_command
import json

def ticket_processing_bot(agent_state: TicketProcessorAgentState, llm):
    ticket = agent_state["current_ticket"]
    ticket_status = ticket.get("status", "").lower() if ticket else ""

    if ticket_status == "to do":
        status_note = (
            "The ticket is in 'To Do' status. You should help the user with all basic information about the ticket, "
            "such as description, priority, and any clarifications. Ask the user what their plan is for the day regarding this ticket. "
            "Guide the user step by step through the process of starting work on this ticket."
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
    You are an agent to conduct a scrum meeting. Do not simulate the user's responses. You have to sound like the manager of the user.
    The user has chosen to work on this ticket. {status_note}

    **Important:**  
    Ask only one question at a time. After the user responds, ask the next relevant question. Do not ask multiple questions in a single message. No need to tell the user that what all questions you will be asking. Just ask the questions one by one.
    After every AI response, ask the user the next question based on the conversation.

    You can use the tools for updating the status of the ticket or fetching the comments of the ticket. 
    You should also ask the user if they want to update the status of the ticket or add a comment to it.

    When updating the status of the ticket, use the following transition IDs for the corresponding statuses: 
    {{
        "To Do": "11",
        "In Progress": "21",
        "Done": "31"
        "Blocked": "2"
    }}

    Once you are done with these, ask the user if they want any other help on this current ticket. If you feel there is no other input needed from the user.
     
    Once the conversation is done, generate a summary of the conversation and show it the user. Check if anything else needed to be added to the comments.  Ask the if the you can add this summary as comments in the ticket. Only if the user confirms, proceed to add the comments.

    **Important:**
    Once the user is done with the ticket or if the user says he is wants to work on someother ticket, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "ticket_processing_done"
    }}

    If the users chooses to end the conversation, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "end_conversation"
    }}
    Ticket:
    {json.dumps(agent_state["current_ticket"], indent=2)}
    """

    if agent_state["bot_state"] == MainBotPhase.TICKET_CHOSEN:
        agent_state["ticket_processing_messages"].append(SystemMessage(content=prompt))
        response = llm.invoke(agent_state["ticket_processing_messages"])
        return {
            "bot_flow": BotFlow.TICKET_PROCESSING_FLOW,
            "bot_state": TicketProcessingBotPhase.IN_PROGRESS,
            "current_ticket": agent_state["current_ticket"],
            "messages": agent_state["messages"],
            "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
            "ticket_processing_messages": list(agent_state["ticket_processing_messages"]) + [response]
        }

    if agent_state["ticket_processing_messages"] and isinstance(agent_state["ticket_processing_messages"][-1], ToolMessage):
        response = llm.invoke(agent_state["ticket_processing_messages"])
        return {
            "bot_flow": BotFlow.TICKET_PROCESSING_FLOW,
            "bot_state": TicketProcessingBotPhase.IN_PROGRESS,
            "current_ticket": agent_state["current_ticket"],
            "messages": agent_state["messages"],
            "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
            "ticket_processing_messages": list(agent_state["ticket_processing_messages"]) + [response]
        }

    user_input = input("User: ")
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