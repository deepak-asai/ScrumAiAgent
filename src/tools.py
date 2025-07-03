from langchain_core.tools import tool
from jira_service import JiraService

@tool
def fetch_comments(ticket_id: str) -> list:
    """
    Fetch comments for a specific Jira ticket.
    """
    service = JiraService.get_instance()
    comments = service.get_comments(ticket_id)
    return comments

@tool
def update_status(ticket_id: str, transition_id: str) -> str:
    """
    Update the status of a Jira ticket using a transition ID.
    """
    service = JiraService.get_instance()
    result = service.update_ticket_status(ticket_id, transition_id)
    return "Status updated successfully." if result else "Failed to update status."

@tool
def add_comment(ticket_id: str, comment: str) -> str:
    """
    Add a comment to a Jira ticket.
    """
    service = JiraService.get_instance()
    result = service.add_comment(ticket_id, comment)
    return "Comment added successfully." if result else "Failed to add comment."
