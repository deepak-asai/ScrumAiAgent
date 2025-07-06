from langchain_core.tools import tool
from jira_service import JiraService
from datetime import date


@tool
def current_date() -> str:
    """
    Returns today's date in ISO format.
    """
    return date.today().isoformat()
    # return "2025-07-09"

@tool
def fetch_comments(ticket_id: str) -> list:
    """
    Fetch comments for a specific Jira ticket.
    """
    service = JiraService.get_instance()
    comments = service.fetch_ticket_comments(ticket_id)
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

@tool
def update_ticket_dates(ticket_id: str, start_date: str = None, end_date: str = None) -> str:
    """
    Update the start date and/or end date (due date) of a Jira ticket.
    Dates should be in ISO format: 'YYYY-MM-DD'.
    """
    service = JiraService.get_instance()
    result = service.update_ticket_dates(ticket_id, start_date, end_date)
    return "Ticket dates updated successfully." if result else "Failed to update ticket dates."
