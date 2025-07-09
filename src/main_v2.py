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
from main_bot_v2 import main_bot
from ticket_processor_bot_v2 import execute_stage, custom_tool_node, summarize_conversation_node, ticket_processing_end_node

tools = [current_date, fetch_comments, add_comment, update_status, update_ticket_dates]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5).bind_tools(tools)

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



def stage_flow_decision(state):
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]
    # if current_stage_id == 2:
    #     breakpoint()  # For debugging purposes, remove in production
    return current_stage["phase"]


graph = StateGraph(ScrumAgentTicketProcessorState)
graph.add_node("basic_info_custom_tool_node", custom_tool_node)
graph.add_node("plan_for_the_day_custom_tool_node", custom_tool_node)
graph.add_node("blocker_check_custom_tool_node", custom_tool_node)
graph.add_node("confirm_summary_custom_tool_node", custom_tool_node)

graph.add_node("basic_info", lambda state: execute_stage(state, llm))
graph.add_node("plan_for_the_day", lambda state: execute_stage(state, llm))
graph.add_node("blocker_check", lambda state: execute_stage(state, llm))
graph.add_node("summarize_conversation", lambda state: summarize_conversation_node(state, llm))
graph.add_node("confirm_summary", lambda state: execute_stage(state, llm))
graph.add_node("ticket_processing_end_node", ticket_processing_end_node)

graph.set_entry_point("basic_info")
graph.add_conditional_edges(
    "basic_info",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "basic_info",
        TicketProcessorPhase.IN_PROGRESS: "basic_info",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "plan_for_the_day",
        TicketProcessorPhase.TOOLS_CALL: "basic_info_custom_tool_node",  # This will call the tool node
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
        TicketProcessorPhase.TOOLS_CALL: "plan_for_the_day_custom_tool_node",  # This will call the tool node
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
        TicketProcessorPhase.TOOLS_CALL: "blocker_check_custom_tool_node",
        TicketProcessorPhase.END_CONVERSATION: "ticket_processing_end_node",
        TicketProcessorPhase.COMPLETED: "ticket_processing_end_node",
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
        TicketProcessorPhase.TOOLS_CALL: "confirm_summary_custom_tool_node",
        TicketProcessorPhase.END_CONVERSATION: "ticket_processing_end_node",
        TicketProcessorPhase.COMPLETED: "ticket_processing_end_node",
    }
)
graph.add_edge("ticket_processing_end_node", END)
graph.add_edge("basic_info_custom_tool_node", "basic_info")
graph.add_edge("plan_for_the_day_custom_tool_node", "plan_for_the_day")
graph.add_edge("blocker_check_custom_tool_node", "blocker_check")
graph.add_edge("confirm_summary_custom_tool_node", "confirm_summary")
subgraph_app = graph.compile()



main_graph = StateGraph(ScrumAgentTicketProcessorState)
main_graph.add_node("main_bot", lambda state: main_bot(state, llm))
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
    "ticket_processing_current_stage": "basic_info",
    "ticket_processing_stages": {
        "basic_info": {
            "id": 0,
            "node": "basic_info",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        "plan_for_the_day": {
            "id": 1,
            "node": "plan_for_the_day",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        "blocker_check": {
            "id": 2,
            "node": "blocker_check",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        "summarize_conversation": {
            "id": 3,
            "node": "summarize_conversation",
            "summary": "",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        "confirm_summary": {
            "id": 4,
            "node": "confirm_summary",
            "messages": [],
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1
        }
    }
}

main_graph_app.invoke(initial_state)