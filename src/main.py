from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict, List
from langgraph.graph.message import add_messages
# # from langgraph.schema import Message
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from jira_service import JiraService, Ticket, Comment
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from enum import Enum


import os
import json

load_dotenv()
prev_message = ""


class TicketProcessingState(str, Enum):
    NOT_STARTED = "not_started"
    CHOSEN = "chosen"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class TicketProcessorAgentState(TypedDict):
    all_tickets: List[Ticket]
    current_ticket: Ticket
    processed_tickets: List[Ticket]  
    is_bot_starting: bool
    is_bot_conversation_continued: bool
    ticket_processing_state: TicketProcessingState
    messages: Annotated[Sequence[BaseMessage], add_messages]

def fetch_jira_tickets(user_id) -> list[Ticket]:
    """Fetch Jira tickets for a given user using Jira REST API."""
    service = JiraService.get_instance()
    return service.fetch_user_tickets(user_id, "APP")

@tool
def update_status(ticket_id: str, transition_id: str) -> str:
    """
    Update the status of a specific ticket.

    Args:
        ticket_id (str): The ID of the ticket to update.
        transition_id (str): The ID of the transition to apply to the ticket.
    """
    service = JiraService.get_instance()
    service.update_ticket_status(ticket_id, transition_id)
    return "Ticket status updated successfully."

@tool
def add_comment(ticket_id: str, content: str) -> str:
    """
    Add a comment to a specific ticket.

    Args:
        ticket_id (str): The ID of the ticket.
        content (str): The content of the comment to be added.
    """
    service = JiraService.get_instance()
    service.add_comment(ticket_id, content)
    return "Comment added successfully to the ticket."
    
@tool
def fetch_comments(ticket_id: str, user_query: str) -> list[Comment]:
    """
    Fetch comments for a specific ticket and uses llm to process the comments.

    Args:
        ticket_id (str): The ID of the Jira ticket.
        user_query (str): The user's query related to the comments.
    """
    service = JiraService.get_instance()
    comments = service.fetch_ticket_comments(ticket_id)
    return comments

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

def init_bot(agent_state: TicketProcessorAgentState):
    tickets = fetch_jira_tickets("deepak.a.1996@gmail.com")
    tickets_str = json.dumps(tickets, indent=2)

    conversation_note = (
        "This is a continuation of a previous conversation. Continue helping the user with their tickets."
        if agent_state['is_bot_conversation_continued']
        else "This is a new conversation. Start by greeting the user and helping them choose a ticket. Give a small introduction about the bot and its purpose."
    )
    
    prompt = f"""
    You are an agent to conduct a scrum meeting. You have to sound like the manager of the user. You have a list of tickets. All these tickets are assigned to the user. These tickets will have ids, summary and description, priority and status. You should ask the user to choose a ticket to start working on. The user will reply with the ticket id. The user might need to know about any ticket's description. You should help with it. Show the list of tickets in the a format that is easy to read:
        Ticket ID: <id>
        Summary: <summary>
        Status: <status>
        Priority: <priority>
        
    Tickets:
    {tickets_str}

    If the user selects a ticket, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:

    {{
    "current_ticket": "<ticket_id>"
    }}

    Replace <ticket_id> with the actual ticket id selected by the user.

    If the user has not selected a ticket, continue the conversation as usual.

    {conversation_note}
    
    If the user does not choose any ticket, respond ONLY with the following text: bot_processing_done. Don't send any other message after this.
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
        data = json.loads(response.content)
        if "current_ticket" in data:
            return {
                "is_bot_starting": False,
                "is_bot_conversation_continued": False,
                "ticket_processing_state": TicketProcessingState.CHOSEN,
                "all_tickets": tickets,
                "current_ticket": next((t for t in tickets if t["id"] == data["current_ticket"]), None),
                "messages": list(agent_state["messages"]) + [response]
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "is_bot_starting": False,
        "is_bot_conversation_continued": False,
        "all_tickets": tickets,
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
    Once the user is done with the a ticket, respond ONLY with the following text: ticket_processing_done. Don't send any other message after this.

    Ticket:
    {json.dumps(agent_state["current_ticket"], indent=2)}
    """

    if agent_state["ticket_processing_state"] == TicketProcessingState.CHOSEN:
        agent_state["messages"].append(SystemMessage(content=prompt))
        response = llm.invoke(agent_state["messages"])
        print("AI response: ", response.content)
        agent_state["messages"].append(AIMessage(content=response.content))

    # breakpoint()  # For debugging purposes, remove in production
    if agent_state["messages"] and isinstance(agent_state["messages"][-1], ToolMessage):
        response = llm.invoke(agent_state["messages"])
        print("AI response: ", response.content)
        agent_state["messages"].append(AIMessage(content=response.content))

    user_input = input("User: ")
    agent_state["messages"].append(HumanMessage(content=user_input))

    response = llm.invoke(agent_state["messages"])

    if response.content == "ticket_processing_done":
        # If the user is done with the ticket processing, we can end the conversation
        return {
            "all_tickets": agent_state["all_tickets"],
            "is_bot_starting": True,
            "is_bot_conversation_continued": True,
            "ticket_processing_state": TicketProcessingState.COMPLETED,
            "current_ticket": agent_state["current_ticket"],
            "messages": list(agent_state["messages"]) + [response]
        }
    
    

    # print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "all_tickets": agent_state["all_tickets"],
        "is_bot_starting": False,
        "is_bot_conversation_continued": False,
        "ticket_processing_state": TicketProcessingState.IN_PROGRESS,
        "current_ticket": agent_state["current_ticket"],
        "messages": list(agent_state["messages"]) + [response]
    }

def should_continue_main_bot(state: TicketProcessorAgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # last_msg = messages[-1]
    if "ticket_processing_state" in state and state["ticket_processing_state"] == TicketProcessingState.CHOSEN:
        return "ticket_chosen"
    
    return "continue"

def should_continue_ticket_processing_bot(state: TicketProcessorAgentState) -> str:
    # breakpoint()
    messages = state["messages"]
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
    }
)

graph.add_node("process_ticket", process_ticket)
graph.add_node("tools", ToolNode(tools))

graph.add_conditional_edges(
    "process_ticket",
    should_continue_ticket_processing_bot,
    {
        "continue": "process_ticket",
        "ticket_processing_done": "init_bot",
        "tools_call": "tools",
    }
)
# graph.add_edge("process_ticket", "tools")  # Connect process_ticket to tools
graph.add_edge("tools", "process_ticket")


app = graph.compile()


# # Define the flow
# graph.add_edge("intro_and_fetch_tickets", "ask_for_updates")

def print_messages(messages):
    # breakpoint()  # For debugging purposes, remove in production
    global prev_message
    if not messages or messages[-1].content == '':
        return
    # if not messages or messages[-1].content is None or prev_message == messages[-1].content:
    #     return
    
    # if isinstance(messages[-1], ToolMessage):
    #     print(f"\n🛠️ AI (tool): {messages[-1].content}")
    elif isinstance(messages[-1], AIMessage):
        print(f"\n🤖 AI: {messages[-1].content}")
    elif isinstance(messages[-1], HumanMessage):
        print(f"\n👤 USER: {messages[-1].content}")
    prev_message = messages[-1].content

# Entry point
def main():
    print("\n ===== DRAFTER =====")
    
    state = {
        "messages": [],
        "is_bot_starting": True,
        "is_bot_conversation_continued": False,
        "is_ticket_processing": False,
    }
    
    for current_step in app.stream(state, stream_mode="values"):
        if "messages" in current_step:
            print_messages(current_step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    main()