from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from jira_service import JiraService, Ticket, Comment
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from enum import Enum
from typing import cast, NotRequired
from tools import fetch_comments, add_comment, update_status

import os
import json

load_dotenv()
prev_message = ""


class TicketProcessingState(str, Enum):
    NOT_STARTED = "not_started"
    CHOSEN = "chosen"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
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
    ticket_processing_state: TicketProcessingState
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticket_processing_messages: Annotated[Sequence[BaseMessage], add_messages]

def fetch_jira_tickets(user_id) -> list[Ticket]:
    """Fetch Jira tickets for a given user using Jira REST API."""
    service = JiraService.get_instance()
    return service.fetch_user_tickets(user_id, "APP")

tools = [fetch_comments, add_comment, update_status]
llm = ChatOpenAI(model="gpt-4", temperature=0.5).bind_tools(tools)

main_llm = ChatOpenAI(model="gpt-4", temperature=0.5)

def print_structured_messages(messages):
    print("\n--- Conversation History ---")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        # For ToolMessage, show tool_call_id if available
        if hasattr(msg, "tool_call_id"):
            print(f"{i+1}. [{msg_type}] (tool_call_id={getattr(msg, 'tool_call_id', '')}): {msg.content}")
        else:
            print(f"{i+1}. [{msg_type}]: {msg.content}")
    print("--- End of History ---\n")

def deserialize_system_command(json_str: str) -> SystemCommand:
    """
    Deserialize a JSON string into a SystemCommand TypedDict.
    Raises ValueError if the JSON is invalid or missing required keys.
    """
    data = json.loads(json_str)
    if not isinstance(data, dict) or "command" not in data:
        raise ValueError("Invalid SystemCommand format")
    # If 'args' is missing, add it as None or an empty dict (as you prefer)
    if "args" not in data:
        data["args"] = None  # or data["args"] = {}
    return cast(SystemCommand, data)

def init_bot(agent_state: TicketProcessorAgentState):
    tickets = fetch_jira_tickets("deepak.a.1996@gmail.com")
    tickets_str = json.dumps(tickets, indent=2)

    agent_state["messages"] = []

    conversation_note = (
        """
        This is a continuation of a previous conversation. Continue helping the user with their tickets. Ask the user what is the next ticket they want to discuss about any other ticket or else if you could end the conversation. "
        """
        if agent_state['is_bot_conversation_continued']
        else "This is a new conversation. Start by greeting the user and helping them choose a ticket. Give a small introduction about the bot and its purpose."
    )
    
    prompt = f"""
    You are an agent to conduct a scrum meeting. 
    {conversation_note}
    You have to sound like the manager of the user. You have a list of tickets. All these tickets are assigned to the user. These tickets will have ids, summary and description, priority and status. You should ask the user to choose a ticket to start discussing on. The user will reply with the ticket id or name. The user might need to know about any ticket's description. You should help with it. Show the list of tickets in the a format that is easy to read:
        Ticket ID: <id>
        Summary: <summary>
        Status: <status>
        Priority: <priority>
        
    Tickets:
    {tickets_str}

    The order of the tickets should be as follows:
    1. Tickets with status "In Progress"
    2. Tickets with status "To Do"
    3. Tickets which are recently processed by the user should be shown at the end of the list.

    These are the tickets ids which are recently processed by the user:
    {agent_state["recently_processed_ticket_ids"] or []}
    Do not tell about the recently processed tickets to the user. Just use this information to order the tickets.

    If the user selects a ticket, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:

    {{
        "command": "ticket_chosen",
        "args": {{
            "ticket_id": "<ticket_id>"
        }},
    }}

    Replace <ticket_id> with the actual ticket id selected by the user.

    If the user has not selected a ticket, continue the conversation as usual.

    If the user does not choose any ticket or if the users chooses to end the conversation, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "end_conversation"
    }}
    """

    if agent_state["is_bot_starting"] or agent_state["is_bot_conversation_continued"]:
        agent_state["messages"].append(SystemMessage(content=prompt))
        response = main_llm.invoke(agent_state["messages"])
        print("AI response: ", response.content)
        agent_state["messages"].append(AIMessage(content=response.content))

    user_input = input("User: ")
    agent_state["messages"].append(HumanMessage(content=user_input))

    response = main_llm.invoke(agent_state["messages"])
    # if(isinstance(response.content)
    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "ticket_chosen" and "ticket_id" in systemCommand["args"]:
            return {
                "botFlow": BotFlow.TICKET_PROCESSING_FLOW,
                "is_bot_starting": False,
                "is_bot_conversation_continued": False,
                "ticket_processing_state": TicketProcessingState.CHOSEN,
                "all_tickets": tickets,
                "current_ticket": next((t for t in tickets if t["id"] == systemCommand["args"]["ticket_id"]), None),
                "messages": list(agent_state["messages"]),
                "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
                "ticket_processing_messages": [response]
            }
        
        if systemCommand["command"] == "end_conversation":
            # If the user wants to end the conversation, we can end the conversation
            return {
                "ticket_processing_state": TicketProcessingState.END_CONVERSATION,
            }
        
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "botFlow": BotFlow.MAIN_BOT_FLOW,
        "is_bot_starting": False,
        "is_bot_conversation_continued": False,
        "all_tickets": tickets,
        "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
        "messages": list(agent_state["messages"]) + [response]
    }


def process_ticket(agent_state: TicketProcessorAgentState):
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

    if agent_state["ticket_processing_state"] == TicketProcessingState.CHOSEN:
        agent_state["ticket_processing_messages"].append(SystemMessage(content=prompt))
        response = llm.invoke(agent_state["ticket_processing_messages"])
        print("AI response: ", response.content)
        agent_state["ticket_processing_messages"].append(AIMessage(content=response.content))

    if agent_state["ticket_processing_messages"] and isinstance(agent_state["ticket_processing_messages"][-1], ToolMessage):
        response = llm.invoke(agent_state["ticket_processing_messages"])
        print("AI response: ", response.content)
        agent_state["ticket_processing_messages"].append(AIMessage(content=response.content))

    user_input = input("User: ")
    agent_state["ticket_processing_messages"].append(HumanMessage(content=user_input))

    response = llm.invoke(agent_state["ticket_processing_messages"])

    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "ticket_processing_done":
            # If the user is done with the ticket processing, we can end the conversation
            return {
                "botFlow": BotFlow.MAIN_BOT_FLOW,
                "all_tickets": agent_state["all_tickets"],
                "is_bot_starting": True,
                "is_bot_conversation_continued": True,
                "ticket_processing_state": TicketProcessingState.COMPLETED,
                "current_ticket": agent_state["current_ticket"],
                "recently_processed_ticket_ids": list(set((agent_state.get("recently_processed_ticket_ids") or []) + [agent_state["current_ticket"]["id"]])),
                "messages": [],
                "ticket_processing_messages": []
            }
        
        if systemCommand["command"] == "end_conversation":
            # If the user wants to end the conversation, we can end the conversation
            return {
                "ticket_processing_state": TicketProcessingState.END_CONVERSATION,
            }
        
    except (json.JSONDecodeError, TypeError):
        pass
    
    # print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "botFlow": BotFlow.TICKET_PROCESSING_FLOW,
        "all_tickets": agent_state["all_tickets"],
        "is_bot_starting": False,
        "is_bot_conversation_continued": False,
        "ticket_processing_state": TicketProcessingState.IN_PROGRESS,
        "current_ticket": agent_state["current_ticket"],
        "messages": agent_state["messages"],
        "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
        "ticket_processing_messages": list(agent_state["ticket_processing_messages"]) + [response]
    }

def should_continue_main_bot(state: TicketProcessorAgentState) -> str:
    """Determine if we should continue or end the conversation."""

    if "ticket_processing_state" in state and state["ticket_processing_state"] == TicketProcessingState.END_CONVERSATION:
        return "end_conversation"

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    if "ticket_processing_state" in state and state["ticket_processing_state"] == TicketProcessingState.CHOSEN:
        return "ticket_chosen"
   
    return "continue"

def should_continue_ticket_processing_bot(state: TicketProcessorAgentState) -> str:
    if "ticket_processing_state" in state and state["ticket_processing_state"] == TicketProcessingState.END_CONVERSATION:
        return "end_conversation"
    
    messages = state["ticket_processing_messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools_call"  # This tells LangGraph to follow the normal edge (to ToolNode)
    
    if not messages:
        return "continue"
    
    if "ticket_processing_state" in state and state["ticket_processing_state"] == TicketProcessingState.COMPLETED:
        return "ticket_processing_done"
    
    return "continue"

# Build the LangGraph
graph = StateGraph(TicketProcessorAgentState)
graph.add_node("init_bot", init_bot)
graph.set_entry_point("init_bot")

graph.add_conditional_edges(
    "init_bot",
    should_continue_main_bot,
    {
        "continue": "init_bot",
        "ticket_chosen": "process_ticket",
        "end_conversation": END,  # This will end the conversation
    }
)

graph.add_node("process_ticket", process_ticket)
graph.add_node("tools", ToolNode(tools, messages_key="ticket_processing_messages"))

graph.add_conditional_edges(
    "process_ticket",
    should_continue_ticket_processing_bot,
    {
        "continue": "process_ticket",
        "ticket_processing_done": "init_bot",
        "tools_call": "tools",
        "end_conversation": END,
    }
)
graph.add_edge("tools", "process_ticket")
app = graph.compile()


def print_messages(messages):
    global prev_message
    if not messages or messages[-1].content == '':
        return
    

    elif isinstance(messages[-1], AIMessage):
        print(f"\n🤖 AI: {messages[-1].content}")
    elif isinstance(messages[-1], HumanMessage):
        print(f"\n👤 USER: {messages[-1].content}")
    prev_message = messages[-1].content

# Entry point
def main():
    print("\n ===== Scrum Bot =====")
    
    state = {
        "botFlow": BotFlow.MAIN_BOT_FLOW,
        "messages": [],
        "is_bot_starting": True,
        "is_bot_conversation_continued": False,
        "is_ticket_processing": False,
        "recently_processed_ticket_ids": [],
    }
    
    for current_step in app.stream(state, stream_mode="values"):
        if current_step["botFlow"] == BotFlow.MAIN_BOT_FLOW and "messages" in current_step and current_step["is_bot_conversation_continued"] == False:
            print_messages(current_step["messages"])
        elif current_step["botFlow"] == BotFlow.TICKET_PROCESSING_FLOW and "ticket_processing_messages" in current_step:
            print_messages(current_step["ticket_processing_messages"])

    
if __name__ == "__main__":
    main()