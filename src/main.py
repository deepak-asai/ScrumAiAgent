from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from tools import fetch_comments, add_comment, update_status
from models import (
    BotFlow,
    MainBotPhase,
    TicketProcessingBotPhase,
    TicketProcessorAgentState,
)
from main_bot import main_bot
from ticket_processing_bot import ticket_processing_bot

load_dotenv()
prev_message = ""
bot_messages = []

tools = [fetch_comments, add_comment, update_status]
llm = ChatOpenAI(model="gpt-4", temperature=0.5).bind_tools(tools)
main_bot_llm = ChatOpenAI(model="gpt-4", temperature=0.5)


def should_continue_main_bot(state: TicketProcessorAgentState) -> str:
    """Determine if we should continue or end the conversation."""

    if "bot_state" in state and state["bot_state"] == MainBotPhase.COMPLETED:
        return "end_conversation"

    messages = state["messages"]
    
    if not messages:
        return "continue"

    if "bot_state" in state and state["bot_state"] == MainBotPhase.TICKET_CHOSEN:
        return "ticket_chosen"
   
    return "continue"

def should_continue_ticket_processing_bot(state: TicketProcessorAgentState) -> str:
    if "bot_state" in state and state["bot_state"] == TicketProcessingBotPhase.END_CONVERSATION:
        return "end_conversation"
    
    messages = state["ticket_processing_messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools_call"
    
    if not messages:
        return "continue"

    if "bot_state" in state and state["bot_state"] == MainBotPhase.RESTARTED:
        return "ticket_processing_done"
    
    return "continue"

# Build the LangGraph
graph = StateGraph(TicketProcessorAgentState)
graph.add_node("main_bot", lambda state: main_bot(state, main_bot_llm))
graph.set_entry_point("main_bot")

graph.add_conditional_edges(
    "main_bot",
    should_continue_main_bot,
    {
        "continue": "main_bot",
        "ticket_chosen": "ticket_processing_bot",
        "end_conversation": END,  # This will end the conversation
    }
)

graph.add_node("ticket_processing_bot", lambda state: ticket_processing_bot(state, llm))
graph.add_node("tools", ToolNode(tools, messages_key="ticket_processing_messages"))

graph.add_conditional_edges(
    "ticket_processing_bot",
    should_continue_ticket_processing_bot,
    {
        "continue": "ticket_processing_bot",
        "ticket_processing_done": "main_bot",
        "tools_call": "tools",
        "end_conversation": END,
    }
)
graph.add_edge("tools", "ticket_processing_bot")
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
        "bot_flow": BotFlow.MAIN_BOT_FLOW,
        "bot_state": MainBotPhase.NOT_STARTED,
        "messages": [],
        "is_ticket_processing": False,
        "recently_processed_ticket_ids": [],
    }
    
    for current_step in app.stream(state, stream_mode="values"):
        if current_step["bot_flow"] == BotFlow.MAIN_BOT_FLOW and "messages" in current_step and current_step["bot_state"] != MainBotPhase.RESTARTED:
            print_messages(current_step["messages"])
        elif current_step["bot_flow"] == BotFlow.TICKET_PROCESSING_FLOW and "ticket_processing_messages" in current_step:
            print_messages(current_step["ticket_processing_messages"])

    
if __name__ == "__main__":
    main()