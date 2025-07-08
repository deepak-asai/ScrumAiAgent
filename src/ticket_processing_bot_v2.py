from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from models import ScrumAgentTicketProcessorState, TicketProcessorPhase, MainBotPhase
from tools import current_date, fetch_comments, add_comment, update_status, update_ticket_dates
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from helpers import deserialize_system_command
from tools import current_date
from langchain_core.messages import ToolMessage
from jira_service import JiraService, Ticket
from langchain_core.prompts import PromptTemplate
from prompts import ticket_processor_stage_prompt, ticket_processor_base_prompt
import json

tools = [current_date, fetch_comments, add_comment, update_status, update_ticket_dates]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5).bind_tools(tools)

    

def handler_not_started_phase(state: ScrumAgentTicketProcessorState):
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]

    ticket_processor_prompt = ticket_processor_base_prompt(state)
    current_stage_prompt = ticket_processor_stage_prompt(state, current_stage["node"])
    current_stage["messages"].append(SystemMessage(content=ticket_processor_prompt + " \n " + current_stage_prompt))
    response = llm.invoke(current_stage["messages"])
    print(f"\n👤 AI: {response.content}")
    current_stage["messages"].append(response)
    current_stage["phase"] = TicketProcessorPhase.IN_PROGRESS
    return state

def invoke_llm_call(state: ScrumAgentTicketProcessorState):
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]

    # if current_stage_id == 4:
    #     breakpoint()
    response = llm.invoke(current_stage["messages"])
    print(f"\n👤 AI: {response.content}")
    current_stage["messages"].append(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        # breakpoint()
        current_stage["phase"] = TicketProcessorPhase.TOOLS_CALL
        return state
    
    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "proceed_to_next_stage":
            current_stage["phase"] = TicketProcessorPhase.PROCEED_TO_NEXT_STAGE
            if "args" in systemCommand and "next_stage_id" in systemCommand["args"]:
                current_stage["next_stage_id"] = systemCommand["args"].get("next_stage_id", -1)
            return state
        
        if systemCommand["command"] == "end_conversation":
            current_stage["phase"] = TicketProcessorPhase.END_CONVERSATION

            return state
        
    except (json.JSONDecodeError, TypeError):
        pass

    return state

# def generate_summary(state: ScrumAgentTicketProcessorState):
#     current_stage_id = state["ticket_processing_current_stage"]
#     current_stage = state["ticket_processing_stages"][current_stage_id]
    

#     current_stage["summary"] = summary
#     print(f"\nSummary of the conversation:\n{summary}")
    
#     return state

def fetch_jira_tickets(user_id) -> list[Ticket]:
    """Fetch Jira tickets for a given user using Jira REST API."""
    service = JiraService.get_instance()
    return service.fetch_user_tickets(user_id, "APP")


def main_bot(agent_state: ScrumAgentTicketProcessorState):
    # breakpoint()  # For debugging purposes, remove in production
    tickets = fetch_jira_tickets("deepak.a.1996@gmail.com")
    tickets_str = json.dumps(tickets, indent=2)

    conversation_note = (
        """
        This is a continuation of a previous conversation. Continue helping the user with their tickets. Ask the user what is the next ticket they want to discuss about any other ticket or else if you could end the conversation. "
        """
        if agent_state['main_bot_phase'] == MainBotPhase.RESTARTED
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

    # breakpoint()  # For debugging purposes, remove in production
    if agent_state["main_bot_phase"] == MainBotPhase.RESTARTED:
        agent_state["main_bot_messages"] = []

    if agent_state["main_bot_phase"] == MainBotPhase.NOT_STARTED or MainBotPhase.RESTARTED:
        agent_state["main_bot_messages"].append(SystemMessage(content=prompt))
    
    if agent_state["main_bot_phase"] in [MainBotPhase.NOT_STARTED, MainBotPhase.RESTARTED]:
        response = llm.invoke(agent_state["main_bot_messages"])
        print(f"\n👤 AI: {response.content}")
        return {
            "main_bot_phase": MainBotPhase.IN_PROGRESS,
            "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
            "main_bot_messages": list(agent_state["main_bot_messages"]) + [response]
        }

    user_input = input("\n👤 User: ")
    agent_state["main_bot_messages"].append(HumanMessage(content=user_input))

    response = llm.invoke(agent_state["main_bot_messages"])
    print(f"\n👤 AI: {response.content}")
    try:
        systemCommand = deserialize_system_command(response.content)
        if systemCommand["command"] == "ticket_chosen" and "ticket_id" in systemCommand["args"]:
            return {
                "main_bot_phase": MainBotPhase.TICKET_CHOSEN,
                "current_ticket": next((t for t in tickets if t["id"] == systemCommand["args"]["ticket_id"]), None),
                "main_bot_messages": list(agent_state["main_bot_messages"]),
                "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
                "ticket_processing_messages": [response]
            }
        
        if systemCommand["command"] == "end_conversation":
            return {
                "main_bot_phase": MainBotPhase.END_CONVERSATION,
            }
        
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "main_bot_phase": MainBotPhase.IN_PROGRESS,
        "recently_processed_ticket_ids": agent_state["recently_processed_ticket_ids"],
        "main_bot_messages": list(agent_state["main_bot_messages"]) + [response]
    }

def main_bot_flow_decision(state: ScrumAgentTicketProcessorState):
    # breakpoint()  # For debugging purposes, remove in production
    if "bot_state" in state and state["bot_state"] == MainBotPhase.COMPLETED:
        return "end_conversation"

    messages = state["main_bot_messages"]

    if not messages:
        return "continue"

    if "main_bot_phase" in state and state["main_bot_phase"] == MainBotPhase.TICKET_CHOSEN:
        return "ticket_processing_bot"
   
    return "continue"

def execute_stage(state: ScrumAgentTicketProcessorState):
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]

    # if current_stage_id == 3 or current_stage_id == 4:
    #     breakpoint()
    if current_stage["phase"] == TicketProcessorPhase.PROCEED_TO_NEXT_STAGE:
        state["ticket_processing_current_stage"] = current_stage["next_stage_id"]
        return state

    # print(f"\n👤 Node: {current_stage['node']}, Phase: {current_stage['phase']}")
    if current_stage["phase"] == TicketProcessorPhase.NOT_STARTED:
        return handler_not_started_phase(state)
    
    if is_last_message_tool_call(current_stage["messages"]):
        # breakpoint()
        return invoke_llm_call(state)

    user_input = input(f"\n👤 User from execute: ")
    user_message = HumanMessage(content=user_input)
    current_stage["messages"].append(user_message)

    return invoke_llm_call(state)

def summarize_conversation_node(state: ScrumAgentTicketProcessorState):
        # Collect all messages from all stages
        # breakpoint()  # For debugging purposes, remove in production
        summary_prompt = ticket_processor_stage_prompt(state, "summarize_conversation")
        # Use the LLM to generate the summary
        response = llm.invoke([SystemMessage(content=summary_prompt)])
        summary = response.content.strip()

        state["ticket_processing_current_stage"] = 3
        current_stage = state["ticket_processing_stages"][3]
        # Store or print the summary
        current_stage["summary"] = summary
        current_stage["phase"] = TicketProcessorPhase.PROCEED_TO_NEXT_STAGE
        current_stage["next_stage_id"] = 4
        # print(f"\nSummary of the conversation:\n{summary}")

        return state
def ticket_processing_end_node(state: ScrumAgentTicketProcessorState):
    state["main_bot_phase"] = MainBotPhase.RESTARTED
    state["ticket_processing_current_stage"] = 0

    return state

def stage_flow_decision(state):
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]
    # if current_stage_id == 2:
    #     breakpoint()  # For debugging purposes, remove in production
    return current_stage["phase"]

def custom_tool_node(state):
    # breakpoint()
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]
    # Get messages for this stage
    messages = current_stage["messages"]
    
    # Assume the last AI message contains a tool call in the expected format
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            function_name = tool_call["name"]
            params = tool_call.get("args", {})
            # Get the tool function from your tools list or a mapping
            tool_map = {
                "current_date": current_date,
                "fetch_comments": fetch_comments,
                "add_comment": add_comment,
                "update_status": update_status,
                "update_ticket_dates": update_ticket_dates,
            }
            tool_func = tool_map.get(function_name)
            if tool_func:
                # Call the tool with extracted params
                result = tool_func.invoke(params)
                # Optionally, append the result as a ToolMessage
                # breakpoint()
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call.get("id", "")))
    current_stage["phase"] = TicketProcessorPhase.IN_PROGRESS
    current_stage["messages"] = messages
    return state

def is_last_message_tool_call(messages) -> bool:
    return isinstance(messages[-1], ToolMessage)



graph = StateGraph(ScrumAgentTicketProcessorState)
graph.add_node("blocker_check_custom_tool_node", custom_tool_node)
graph.add_node("confirm_summary_custom_tool_node", custom_tool_node)

graph.add_node("basic_info", execute_stage)
graph.add_node("plan_for_the_day", execute_stage)
graph.add_node("blocker_check", execute_stage)
graph.add_node("summarize_conversation", summarize_conversation_node)
graph.add_node("confirm_summary", execute_stage)
graph.add_node("ticket_processing_end_node", ticket_processing_end_node)

graph.set_entry_point("basic_info")
graph.add_conditional_edges(
    "basic_info",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "basic_info",
        TicketProcessorPhase.IN_PROGRESS: "basic_info",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "plan_for_the_day",
        # TicketProcessorPhase.TOOLS_CALL: END,  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: END,  # This will end the conversation
        TicketProcessorPhase.COMPLETED: END,  # This will end the conversation
    }
)

graph.add_conditional_edges(
    "plan_for_the_day",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "plan_for_the_day",
        TicketProcessorPhase.IN_PROGRESS: "plan_for_the_day",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "blocker_check",
        # TicketProcessorPhase.TOOLS_CALL: END,  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: END,  # This will end the conversation
        TicketProcessorPhase.COMPLETED: END,  # This will end the conversation
    }
)

graph.add_conditional_edges(
    "blocker_check",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "blocker_check",
        TicketProcessorPhase.IN_PROGRESS: "blocker_check",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "summarize_conversation",
        TicketProcessorPhase.TOOLS_CALL: "blocker_check_custom_tool_node",  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: "ticket_processing_end_node",  # This will end the conversation
        TicketProcessorPhase.COMPLETED: "ticket_processing_end_node",  # This will end the conversation
    }
)
graph.add_edge("summarize_conversation", "confirm_summary")
graph.add_conditional_edges(
    "confirm_summary",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "confirm_summary",
        TicketProcessorPhase.IN_PROGRESS: "confirm_summary",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "ticket_processing_end_node",
        TicketProcessorPhase.TOOLS_CALL: "confirm_summary_custom_tool_node",  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: "ticket_processing_end_node",  # This will end the conversation
        TicketProcessorPhase.COMPLETED: "ticket_processing_end_node",  # This will end the conversation
    }
)
graph.add_edge("ticket_processing_end_node", END)
# graph.add_edge("custom_tool_node", "basic_info")
# graph.add_edge("custom_tool_node", "plan_for_the_day")
graph.add_edge("blocker_check_custom_tool_node", "blocker_check")
graph.add_edge("confirm_summary_custom_tool_node", "confirm_summary")
subgraph_app = graph.compile()



main_graph = StateGraph(ScrumAgentTicketProcessorState)
main_graph.add_node("main_bot", main_bot)
main_graph.add_node("ticket_processing_bot", subgraph_app)  # graph is your subgraph, not subgraph_app

main_graph.set_entry_point("main_bot")
main_graph.add_conditional_edges(
    "main_bot",
    main_bot_flow_decision,
    {
        "continue": "main_bot",
        "ticket_processing_bot": "ticket_processing_bot",
        "end_conversation": END,  # This will end the conversation
    }
)
main_graph.add_edge("ticket_processing_bot", "main_bot")

main_graph_app = main_graph.compile()

initial_state = {
    "main_bot_phase": MainBotPhase.NOT_STARTED,
    "recently_processed_ticket_ids": [],
    "main_bot_messages": [],
    "ticket_processing_current_stage": 0,
    "ticket_processing_stages": [
        {
            "id": 0,
            "node": "basic_info",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 1,
            "node": "plan_for_the_day",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 2,
            "node": "blocker_check",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 3,
            "node": "summarize_conversation",
            "summary": "",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 4,
            "node": "confirm_summary",
            "messages": [],
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1
        }
    ]
}

main_graph_app.invoke(initial_state)
# for step in app.stream(initial_state, stream_mode="values"):
#     pass  # The logic in your nodes will handle printing and user interaction