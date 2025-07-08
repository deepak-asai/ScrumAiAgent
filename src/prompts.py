from langchain_core.messages import HumanMessage, AIMessage
from models import ScrumAgentTicketProcessorState
from langchain_core.prompts import PromptTemplate
import json

def ticket_processor_base_prompt(state: ScrumAgentTicketProcessorState) -> str:
    ticket_processor_prompt_template = PromptTemplate.from_template("""
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
    """)

    return ticket_processor_prompt_template.format(ticket=json.dumps(state["current_ticket"], indent=2))

def ticket_processor_stage_prompt(state: ScrumAgentTicketProcessorState, node: str):
    prompt_func = globals().get(f"{node}_prompt")
    if prompt_func is None:
        raise ValueError(f"No prompt function found for node: {node}")
    return prompt_func(state)
  
def basic_info_prompt(state: ScrumAgentTicketProcessorState) ->str:
    return """
    Ask for the user whether they need any information about the ticket. Use the tools available to you to assist the user. For every response from AI, ask the user if they have any other questions.
    Once the user is not having any questions, respond ONLY with the following JSON format and do not include any other text, explanation, or greeting:
    {{
        "command": "proceed_to_next_stage",
        "args": {{
            "next_stage_id": 1
        }}
    }}
    """

def plan_for_the_day_prompt(state: ScrumAgentTicketProcessorState) -> str:
    return """
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
    """

def blocker_check_prompt(state: ScrumAgentTicketProcessorState) -> str:
    return """
    Ask the user if they foresee any challenges or blockers in proceeding with the ticket.

    If the user mentions any blockers:
    - Ask if you should update the ticket status to 'Blocked'.
        - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'Blocked'.

    Ask the user if they want to add a comment to the ticket about the blockers.
        - If the user agrees, use the 'add_comment' tool to add the comment.

    If the user does not mention any blockers and if the status of the ticket is 'To Do', you can skip this step.

    If the ticket status is NOT 'In Progress', ask the user if you can update the status of the ticket to 'In Progress' since they are working on it.
        - If the user agrees, you MUST use the 'update_status' tool to update the ticket status to 'In Progress'.

    If the ticket status is already 'In Progress', do NOT ask to update the status again.

    After performing the above steps, respond ONLY with this JSON (do not include any other text, explanation, or formatting):
    {{
        "command": "proceed_to_next_stage",
        "args": {{
            "next_stage_id": 3
        }}
    }}
    """

def summarize_conversation_prompt(state: ScrumAgentTicketProcessorState) -> str:
    all_messages = []
    for stage in state["ticket_processing_stages"]:
        for msg in stage.get("messages", []):
            if isinstance(msg, (AIMessage, HumanMessage)):
                role = "AI" if isinstance(msg, AIMessage) else "User"
                all_messages.append(f"{role}: {msg.content.strip()}")

    # Build the summary prompt
    summary_prompt = (
        "Summarize the following scrum conversation between the user and the AI agent as the user's manager. "
        "Highlight the ticket, actions taken, blockers, and next steps if any.\n\n"
        "Conversation:\n"
        + "\n".join(all_messages)
        + "\n\nSummary:"
    )
    return summary_prompt

def confirm_summary_prompt(state: ScrumAgentTicketProcessorState) -> str:
    return f"""
    Ask the user to confirm the summary of the conversation. Or else ask the user what else to add to this summary. Tell them that you will add this summary as comments in the ticket. 
    Summary: {state["ticket_processing_stages"][3]["summary"]}
    If the user confirms, you MUST use the tool `add_comment` to add the summary to the ticket.
    If the user does not confirm, ask the user what else to add to the summary.
    After performing the above steps, respond ONLY with this JSON (do not include any other text, explanation, or formatting):
    {{
        "command": "end_conversation"
    }}
    """