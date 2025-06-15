from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
# # from langgraph.schema import Message
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from jira_service import JiraService, Ticket, Comment
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

import os
import json

load_dotenv()

class AgentState(TypedDict):
    ticket: Ticket
    messages: Sequence[BaseMessage]

# class Comment(TypedDict):
#     id: str
#     data: str
#     created_at: datetime
#     created_by: str

# class Ticket(TypedDict):
#     id: str
#     title: str
#     description: str
#     priority: NotRequired[str]
#     comments: NotRequired[list[str]]

# class AgentState(TypedDict):
#     user_id: str
#     project_id: NotRequired[str]

#     tickets: list[Ticket]

def fetch_jira_tickets(user_id) -> list[Ticket]:
    # breakpoint() 
    """Fetch Jira tickets for a given user using Jira REST API."""
    jira_url = os.getenv("JIRA_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_token = os.getenv("JIRA_API_TOKEN")
    service = JiraService(jira_url, jira_email, jira_token)
    return service.fetch_user_tickets(user_id, "APP")

@tool
def fetch_comments(ticket_id: str) -> list[Comment]:
    """
    Fetch comments for a specific ticket.

    Args:
        ticket_id (str): The ID of the Jira ticket.

    Returns:
        list[Comment]: A list of comments for the ticket.
    """
    # jira_url = os.getenv("JIRA_URL")
    # jira_email = os.getenv("JIRA_EMAIL")
    # jira_token = os.getenv("JIRA_API_TOKEN")
    # service = JiraService(jira_url, jira_email, jira_token)
    return [{
        "id": "1",
        "author": {
            "accountId": "user1",
            "displayName": "User One",
            "emailAddress": "user1@example.com"
        },
        "body": "This is a comment.",
        "created": "2023-01-01T00:00:00.000Z",
    }]
    # return service.fetch_ticket_comments(ticket_id)

tools = [fetch_comments]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# def fetch_ticket_comments(ticket_id):
#     # breakpoint() 
#     """Fetch Jira comments for a given ticket using Jira REST API."""
#     jira_url = os.getenv("JIRA_URL")
#     jira_email = os.getenv("JIRA_EMAIL")
#     jira_token = os.getenv("JIRA_API_TOKEN")
#     service = JiraService(jira_url, jira_email, jira_token)
#     # return service.fetch_ticket_comments(ticket_id)
#     return service.get_ticket_transitions(ticket_id)

def init_bot():
    breakpoint()  # For debugging purposes, remove in production
    tickets = fetch_jira_tickets("deepak.a.1996@gmail.com")
    tickets_str = json.dumps(tickets, indent=2)
    prompt = f"""
    You are an agent to conduct a scrum meeting. You have a list of tickets. All these tickets are assigned to the user. These tickets will have ids, summary and description. You should ask the user to choose a ticket to start working on. The user will reply with the ticket id. The user might need to know about any ticket's description. You should help with it. Show the list of tickets in the a format that is easy to read:
        Ticket ID: <id>
        Summary: <summary>
    
    Tickets:
    {tickets_str}
    """

    # print(prompt)
    
    messages = []
    messages.append(SystemMessage(content=prompt))
    response = llm.invoke(messages)
    print("AI:", response.content)
    messages.append(AIMessage(content=response.content))
    while(True):
        print("Waiting for user input...")
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        print("AI:", response.content)
        messages.append(AIMessage(content=response.content))

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

def process_ticket(agent_state: AgentState):
    print_structured_messages(agent_state["messages"])
    # messages = agent_state["messages"]
    prompt = f"""
    You are an agent to conduct a scrum meeting. Do not simulate the user's responses. You have a ticket assigned to the user. The user has chosen to work on this ticket. You should help the user with any information they need about this ticket. The user might ask for the description of the ticket or any other details. You can use the tools for updating the status of the ticket or fetching the comments of the ticket. You should also ask the user if they want to update the status of the ticket or add a comment to it. If the user wants to start working on the ticket, you should be asking what is the plan for day. If the ticket was already started, you should ask for the proof of work - like PR or code commit. 
    
    Once you are done with these, ask for the user if they want anyother help on this current ticket? If you feel there is no other input needed from the user, you have to end the conversation with this data:
    {{
        "status": "done"
    }}

    Ticket:
    {json.dumps(agent_state["ticket"], indent=2)}
    """

    if not agent_state["messages"]:
        agent_state["messages"].append(SystemMessage(content=prompt))
        response = llm.invoke(agent_state["messages"])
        print("AI response: ", response.content)
        agent_state["messages"].append(AIMessage(content=response.content))
    
    user_input = input("User: ")
    agent_state["messages"].append(HumanMessage(content=user_input))

    response = llm.invoke(agent_state["messages"])

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    agent_state["messages"].append(AIMessage(content=response.content))

    # agent_state["messages"]
    # print_structured_messages(agent_state["messages"])
    return agent_state


    # print(len(messages))
    # response = llm.invoke(messages)
    print("AI response:", response.content)
    messages.append(AIMessage(content=response.content))
    # breakpoint()  # For debugging purposes, remove in production
    # while(True):
    #     # breakpoint()  # For debugging purposes, remove in production
    #     print("Waiting for user input...")
    #     user_input = input("User: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     messages.append(HumanMessage(content=user_input))
    #     response = llm.invoke(messages)
    #     breakpoint()  # For debugging purposes, remove in production
    #     print("AI:", response.content)
    #     messages.append(AIMessage(content=response.content))
    # print("AI:", response.content)


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # implement quit logic
        
    return "continue"
# tools = [fetch_jira_tickets]
# model = ChatOpenAI(
#     model="gpt-40"
# ).bind_tools(tools)



# # Node 1: Introduction and fetch tickets
# def intro_and_fetch_tickets(state):
#     user_id = state.get("user_id", "unknown")
#     tickets = fetch_jira_tickets(user_id)
#     intro_message = (
#         "👋 Hi! I'm your AI Scrum Master. I'll help you manage your Jira tickets.\n"
#         "Here are your current tickets:\n"
#     )
#     for ticket in tickets:
#         intro_message += f"- {ticket['id']}: {ticket['summary']}\n"
#     return {"tickets": tickets, "message": intro_message}

# # Node 2: Ask user which items to start
# def ask_for_updates(state):
#     tickets = state.get("tickets", [])
#     message = (
#         "Which of these tickets are you planning to start working on today?\n"
#         "Please reply with the ticket IDs."
#     )
#     return {"message": message, "tickets": tickets}

# def model_cll()

# # Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("process_ticket", process_ticket)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("process_ticket")
graph.add_edge("process_ticket", "tools")  # Connect process_ticket to tools
graph.add_edge("tools", "process_ticket")

app = graph.compile()
# graph.add_conditional_edges(
#     "tools",
#     should_continue,
#     {
#         "continue": "agent",
#         "end": END,
#     }
# )

# # Define the flow
# graph.add_edge("intro_and_fetch_tickets", "ask_for_updates")

def print_messages(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

# Entry point
def main():
    # Example initial state
    # state = {"user_id": "user@example.com"}
    # g = Graph(graph, start="intro_and_fetch_tickets")
    # for node_name, state in g.run(state):
    #     print(state["message"])
    # comments = fetch_ticket_comments("CMA-1")
    # print(json.dumps(tickets, indent=2))
    # print(json.dumps(comments, indent=2))
    # init_bot()
    # process_ticket({
    #     "id": "APP-1",
    #     "title": "Fix the login issue",
    #     "description": "The login page is not working as expected.",
    #     "priority": "High"
    # })
    print("\n ===== DRAFTER =====")
    
    state = {
        "ticket": {
            "id": "APP-1",
            "title": "Fix the login issue",
            "description": "The login page is not working as expected.",
            "priority": "High"
        },
        "messages": []
    }
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    main()