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
prev_message = ""

class AgentState(TypedDict):
    ticket: Ticket
    messages: Annotated[Sequence[BaseMessage], add_messages]

def fetch_jira_tickets(user_id) -> list[Ticket]:
    """Fetch Jira tickets for a given user using Jira REST API."""
    jira_url = os.getenv("JIRA_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_token = os.getenv("JIRA_API_TOKEN")
    service = JiraService(jira_url, jira_email, jira_token)
    return service.fetch_user_tickets(user_id, "APP")

@tool
def add_comment(ticket_id: str, content: str) -> str:
    """
    Add a comment to a specific ticket.

    Args:
        ticket_id (str): The ID of the Jira ticket.
        content (str): The content of the comment to be added.
    """
    # jira_url = os.getenv("JIRA_URL")
    # jira_email = os.getenv("JIRA_EMAIL")
    # jira_token = os.getenv("JIRA_API_TOKEN")
    # service = JiraService(jira_url, jira_email, jira_token)
    # return service.add_comment(ticket_id, content)
    print(f"Comment added to ticket {ticket_id}: {content}")
    
@tool
def fetch_comments(ticket_id: str, user_query: str) -> list[Comment]:
    """
    Fetch comments for a specific ticket and uses llm to process the comments.

    Args:
        ticket_id (str): The ID of the Jira ticket.
        user_query (str): The user's query related to the comments.
    """
    # jira_url = os.getenv("JIRA_URL")
    # jira_email = os.getenv("JIRA_EMAIL")
    # jira_token = os.getenv("JIRA_API_TOKEN")
    # service = JiraService(jira_url, jira_email, jira_token)
    comments = [{
        "id": "1",
        "author": {
            "accountId": "user1",
            "displayName": "User One",
            "emailAddress": "user1@example.com"
        },
        "body": "This is a comment.",
        "created": "2023-01-01T00:00:00.000Z",
    },
    {
        "id": "2",
        "author": {
            "accountId": "user1",
            "displayName": "User One",
            "emailAddress": "user1@example.com"
        },
        "body": "There is a blocker issue with the login page.",
        "created": "2023-01-01T00:00:00.000Z",
    },{
        "id": "3",
        "author": {
            "accountId": "user1",
            "displayName": "User One",
            "emailAddress": "user1@example.com"
        },
        "body": "Waiting for the backend team to fix the API issue.",
        "created": "2023-01-01T00:00:00.000Z",
    },{
        "id": "4",
        "author": {
            "accountId": "user1",
            "displayName": "User One",
            "emailAddress": "user1@example.com"
        },
        "body": "The API issue has been fixed. Please check the latest commit.",
        "created": "2023-01-01T00:00:00.000Z",
    }]

    prompt = f"""
    You are an agent to help users with the comments that are added to a ticket. You should be able to help user the asks related to comments. When presenting comments to the user, do not use phrases like "Based on the data provided" or "Here is the information." Instead, directly show the response in a clear and concise manner.
    Comments: {comments}
    User Query: {user_query}
    """

    llm = ChatOpenAI(model="gpt-4")
    llm_response = llm.invoke([SystemMessage(content=prompt)])
    return llm_response.content

tools = [fetch_comments]
llm = ChatOpenAI(model="gpt-4").bind_tools(tools)

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
    prompt = f"""
    You are an agent to conduct a scrum meeting. Do not simulate the user's responses. You have a ticket assigned to the user. The user has chosen to work on this ticket. You should help the user with any information they need about this ticket. The user might ask for the description of the ticket or any other details. You can use the tools for updating the status of the ticket or fetching the comments of the ticket. You should also ask the user if they want to update the status of the ticket or add a comment to it. If the user wants to start working on the ticket, you should ask what is the plan for the day. If the ticket was already started, you should ask for the proof of work - like PR or code commit.

    Once you are done with these, ask the user if they want any other help on this current ticket. If you feel there is no other input needed from the user.
     
    Once the conversation is done, generate a summary of the conversation and add it as comments in the ticket. Ask for a user confirmation and check if anything else needed to be added to the comments. Use the tool to add comments to the ticket.

    Once done, you have to simply return the string "ticket_processing_done". Don't send any other message after this.

    **Important:**  
    Ask only one question at a time. After the user responds, ask the next relevant question. Do not ask multiple questions in a single message. No need to tell the user that what all questions you will be asking. Just ask the questions one by one.
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

    # print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # agent_state["messages"].append(AIMessage(content=response.content))

    # agent_state["messages"]
    # print("\n--- Ending state ---")
    # print_structured_messages(agent_state["messages"])
    # print("--- Ending state end ---\n")
    return {
        "ticket": agent_state["ticket"],
        # "messages": agent_state["messages"]
        "messages": list(agent_state["messages"]) + [response]
    }

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # implement quit logic
        
    return "continue"


# Build the LangGraph
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
    global prev_message
    if not messages or messages[-1].content is None or prev_message == messages[-1].content:
        return
    
    if isinstance(messages[-1], ToolMessage):
        print(f"\n🛠️ AI (tool): {messages[-1].content}")
    elif isinstance(messages[-1], AIMessage):
        print(f"\n🤖 AI: {messages[-1].content}")
    elif isinstance(messages[-1], HumanMessage):
        print(f"\n👤 USER: {messages[-1].content}")
    prev_message = messages[-1].content

# Entry point
def main():
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