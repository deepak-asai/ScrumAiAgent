from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from models import ScrumAgentTicketProcessorState, TicketProcessorPhase
from tools import current_date, fetch_comments, add_comment, update_status, update_ticket_dates
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from helpers import deserialize_system_command
from tools import current_date
from langchain_core.messages import ToolMessage
import json

tools = [current_date, fetch_comments, add_comment, update_status, update_ticket_dates]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5).bind_tools(tools)

def handler_not_started_phase(state: ScrumAgentTicketProcessorState):
    current_stage_id = state["current_stage"]
    current_stage = state["stages"][current_stage_id]

    current_stage["messages"].append(SystemMessage(content=state["basic_instruction"] + " \n " + current_stage["prompt"]))
    response = llm.invoke(current_stage["messages"])
    print(f"\n👤 AI: {response.content}")
    current_stage["messages"].append(response)
    current_stage["phase"] = TicketProcessorPhase.IN_PROGRESS
    return state

def invoke_llm_call(state: ScrumAgentTicketProcessorState):
    current_stage_id = state["current_stage"]
    # if current_stage_id == 2:
    #     breakpoint()
    current_stage = state["stages"][current_stage_id]

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

def execute_stage(state: ScrumAgentTicketProcessorState):
    # breakpoint()
    current_stage_id = state["current_stage"]
    # if current_stage_id == 2:
    #     breakpoint()
    current_stage = state["stages"][current_stage_id]

    if current_stage["phase"] == TicketProcessorPhase.PROCEED_TO_NEXT_STAGE:
        state["current_stage"] = current_stage["next_stage_id"]
        return state

    # print(f"\n👤 Node: {current_stage['node']}, Phase: {current_stage['phase']}")
    if current_stage["phase"] == TicketProcessorPhase.NOT_STARTED:
        return handler_not_started_phase(state)
    
    if is_last_message_tool_call(current_stage["messages"]):
        # breakpoint()
        return invoke_llm_call(state)

    user_input = input(f"\n👤 User: ")
    user_message = HumanMessage(content=user_input)
    current_stage["messages"].append(user_message)

    return invoke_llm_call(state)

# def summarize_converstaion(state: ScrumAgentTicketProcessorState):


def stage_flow_decision(state):
    current_stage_id = state["current_stage"]
    current_stage = state["stages"][current_stage_id]
    
    return current_stage["phase"]

def custom_tool_node(state):
    # breakpoint()
    current_stage_id = state["current_stage"]
    current_stage = state["stages"][current_stage_id]
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
graph.add_node("update_ticket_status_custom_tool_node", custom_tool_node)
graph.add_node("blocker_check_custom_tool_node", custom_tool_node)
graph.add_node("basic_info", execute_stage)
graph.add_node("plan_for_the_day", execute_stage)
graph.add_node("update_ticket_status", execute_stage)
graph.add_node("blocker_check", execute_stage)

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
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "update_ticket_status",
        # TicketProcessorPhase.TOOLS_CALL: END,  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: END,  # This will end the conversation
        TicketProcessorPhase.COMPLETED: END,  # This will end the conversation
    }
)

graph.add_conditional_edges(
    "update_ticket_status",
    stage_flow_decision,
    {
        TicketProcessorPhase.NOT_STARTED: "update_ticket_status",
        TicketProcessorPhase.IN_PROGRESS: "update_ticket_status",
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: "blocker_check",
        TicketProcessorPhase.TOOLS_CALL: "update_ticket_status_custom_tool_node",  # This will call the tool node
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
        TicketProcessorPhase.PROCEED_TO_NEXT_STAGE: END,
        TicketProcessorPhase.TOOLS_CALL: "blocker_check_custom_tool_node",  # This will call the tool node
        TicketProcessorPhase.END_CONVERSATION: END,  # This will end the conversation
        TicketProcessorPhase.COMPLETED: END,  # This will end the conversation
    }
)
# graph.add_edge("custom_tool_node", "basic_info")
# graph.add_edge("custom_tool_node", "plan_for_the_day")
graph.add_edge("update_ticket_status_custom_tool_node", "update_ticket_status")
graph.add_edge("blocker_check_custom_tool_node", "blocker_check")

app = graph.compile()

# try:
#     from IPython.display import Image, display
#     app.get_graph().draw_png("graph.png")
# except Exception as e:
#     breakpoint()
#     print("Could not display image. Ensure pygraphviz is installed.")


ticket = {
    "id": "APP-1",
    "title": "Fix login bug",
    "description": "User is unable to login with correct credentials.",
    "status": "To Do",
    "priority": "High"
}

initial_state = {
    "basic_instruction": 
    f"""
    You are an agent to conduct a scrum meeting. You have to sound like the user's manager. Do not start with any greeting or introduction. The user has chosen to work on this ticket: APP-1
        - {ticket}

    Instructions:
        - If the user wants to update the ticket status, use the following transition IDs: "To Do": "11", "In Progress": "21", "Done": "31", "Blocked": "2"
        - If the user is done with this ticket or if the user says he is wants to work on someother ticket, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
        {{
            "command": "ticket_processing_done"
        }}
        - If the users chooses to end the conversation, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
        {{
            "command": "end_conversation"
        }}
    """,
    "current_stage": 0,
    "processed_stages": [],
    "stages": [
        {
            "id": 0,
            "node": "basic_info",
            "prompt": """
                Ask for the user whether they need any information about the ticket. Use the tools available to you to assist the user. For every response from AI, ask the user if they have any other questions.
                Once the user is not having any questions, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
                {{
                    "command": "proceed_to_next_stage",
                    "args": {{
                        "next_stage_id": 1
                    }}
                }}
                """,
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 1,
            # Only when the the user shares their plan, ask for confirmation that the plan is correct. Do not ask for confirmation if the user does not share any plan.
            "node": "plan_for_the_day",
            "prompt": """
                Ask the user what their plan is for the day regarding this ticket. 
                Do not provide any context or information about the ticket unless the user specifically asks for it.
                No need to use the any tools unless the user asks specifically asks for something.
                Once the user gives the plan, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
                {{
                    "command": "proceed_to_next_stage",
                    "args": {{
                        "next_stage_id": 2
                    }}
                }}
                """,
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        # {
        #     "node": "update_ticket_status",
        #     "prompt": """
        #         Ask the user if you can update the status of the ticket to 'In Progress' since they are working on it.
        #         If the user agrees, you MUST use the 'update_status' tool to update the ticket status.
        #         After the tool call, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
        #         {{
        #             "command": "proceed_to_next_stage",
        #             "args": {{
        #                 "next_stage_id": 3
        #             }}
        #         }}
        #         """,
        #     "phase": TicketProcessorPhase.NOT_STARTED,
        #     "next_stage_id": -1,
        #     "messages": []
        # },
        {
            "id": 2,
            "node": "blocker_check",
            "prompt": """
            Ask the user if they foresee any challenges or blockers in proceeding with the ticket.

            If the user mentions any blockers:
            - Ask if you should update the ticket status to 'Blocked'.
                - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'Blocked'.

            Ask the user if they want to add a comment to the ticket about the blockers.
                - If the user agrees, use the 'add_comment' tool to add the comment.

            If the user does not mention any blockers and if the status of the ticket is 'To Do', you can skip this step.
                - Ask the user if you can update the status of the ticket to 'In Progress' since they are working on it.
                - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'In Progress'.

            After performing the above steps, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
            {
                "command": "proceed_to_next_stage",
                "args": {
                    "next_stage_id": 3
                }
            }
            """,
            # "instruction": "Can we update the status of the ticket to 'In Progress'?",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        },
        {
            "id": 3,
            "node": "summarize_conversation",
            "prompt": """
            Ask the user if they foresee any challenges or blockers in proceeding with the ticket.

            If the user mentions any blockers:
            - Ask if you should update the ticket status to 'Blocked'.
                - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'Blocked'.

            Ask the user if they want to add a comment to the ticket about the blockers.
                - If the user agrees, use the 'add_comment' tool to add the comment.

            If the user does not mention any blockers and if the status of the ticket is 'To Do', you can skip this step.
                - Ask the user if you can update the status of the ticket to 'In Progress' since they are working on it.
                - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'In Progress'.

            After performing the above steps, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
            {
                "command": "proceed_to_next_stage",
                "args": {
                    "next_stage_id": 3
                }
            }
            """,
            # "instruction": "Can we update the status of the ticket to 'In Progress'?",
            "phase": TicketProcessorPhase.NOT_STARTED,
            "next_stage_id": -1,
            "messages": []
        }
    ]
}

app.invoke(initial_state)
# for step in app.stream(initial_state, stream_mode="values"):
#     pass  # The logic in your nodes will handle printing and user interaction