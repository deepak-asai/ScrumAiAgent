import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from models import ScrumAgentTicketProcessorState, TicketProcessorPhase, MainBotPhase
from prompts import ticket_processor_base_prompt, ticket_processor_stage_prompt
from helpers import deserialize_system_command
from tools import (
    current_date,
    fetch_comments,
    add_comment,
    update_status,
    update_ticket_dates,
)

def handler_not_started_phase(state: ScrumAgentTicketProcessorState, llm=None):
    # breakpoint()  # For debugging purposes, remove in production
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

def invoke_llm_call(state: ScrumAgentTicketProcessorState, llm=None):
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


def execute_stage(state: ScrumAgentTicketProcessorState, llm=None):
    # breakpoint()  # For debugging purposes, remove in production
    current_stage_id = state["ticket_processing_current_stage"]
    current_stage = state["ticket_processing_stages"][current_stage_id]

    # if current_stage_id == 3 or current_stage_id == 4:
    #     breakpoint()
    if current_stage["phase"] == TicketProcessorPhase.PROCEED_TO_NEXT_STAGE:
        state["ticket_processing_current_stage"] = current_stage["next_stage_id"]
        return state

    # print(f"\n👤 Node: {current_stage['node']}, Phase: {current_stage['phase']}")
    if current_stage["phase"] == TicketProcessorPhase.NOT_STARTED:
        return handler_not_started_phase(state, llm)
    
    if is_last_message_tool_call(current_stage["messages"]):
        # breakpoint()
        return invoke_llm_call(state, llm)

    user_input = input(f"\n👤 User from execute: ")
    user_message = HumanMessage(content=user_input)
    current_stage["messages"].append(user_message)

    return invoke_llm_call(state, llm)

def summarize_conversation_node(state: ScrumAgentTicketProcessorState, llm=None):
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
    state["recently_processed_ticket_ids"].append(state["current_ticket"]["id"])

    return state

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
