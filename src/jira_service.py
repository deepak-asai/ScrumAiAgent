import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, NotRequired
import json

class Author(TypedDict):
    accountId: str
    displayName: str
    emailAddress: NotRequired[str]

class Comment(TypedDict):
    id: str
    author: Author
    body: str
    created: str
    updated: NotRequired[str]

class Ticket(TypedDict):
    id: str
    title: str
    description: str
    status: NotRequired[str]
    priority: NotRequired[str]
    comments: NotRequired[List[Comment]]

class JiraService:
    _instance = None

    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.headers = {"Accept": "application/json"}

    @staticmethod
    def get_instance():
        load_dotenv()
        if JiraService._instance is None:
            jira_url = os.getenv("JIRA_URL")
            jira_email = os.getenv("JIRA_EMAIL")
            jira_token = os.getenv("JIRA_API_TOKEN")
            JiraService._instance = JiraService(jira_url, jira_email, jira_token)
        return JiraService._instance

    def fetch_user_tickets(self, user_email: str, project_key: str = None) -> List[Ticket]:
        # Build JQL with optional project filter
        jql = f'assignee = "{user_email}" AND resolution = Unresolved'
        if project_key:
            jql = f'project = "{project_key}" AND ' + jql
        url = f"{self.base_url}/rest/api/2/search"
        params = {"jql": jql}
        response = requests.get(url, headers=self.headers, params=params, auth=self.auth)
        response.raise_for_status()
        issues = response.json().get("issues", [])

        tickets: List[Ticket] = []
        for issue in issues:
            fields = issue["fields"]
            ticket: Ticket = {
                "id": issue["key"],
                "title": fields.get("summary", ""),
                "description": fields.get("description", ""),
                "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                "status": fields.get("status", {}).get("name") if fields.get("status") else None,
                "start_date": fields.get("customfield_10015"),
                "due_date": fields.get("duedate"),
            }
            tickets.append(ticket)
        return tickets

    def fetch_ticket_comments(self, issue_key: str) -> List[Comment]:
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment"
        response = requests.get(url, headers=self.headers, auth=self.auth)
        response.raise_for_status()
        data = response.json()
        comments: List[Comment] = []
        for comment in data.get("comments", []):
            author_data = comment.get("author", {})
            author: Author = {
                "accountId": author_data.get("accountId", ""),
                "displayName": author_data.get("displayName", ""),
            }
            if "emailAddress" in author_data:
                author["emailAddress"] = author_data["emailAddress"]
            comment_obj: Comment = {
                "id": comment.get("id", ""),
                "author": author,
                "body": comment.get("body", ""),
                "created": comment.get("created", ""),
            }
            if "updated" in comment:
                comment_obj["updated"] = comment["updated"]
            comments.append(comment_obj)
        return comments

    def add_comment(self, issue_key: str, comment_body: str) -> Dict[str, Any]:
        """
        Add a comment to a Jira ticket.
        """
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment"
        payload = {"body": comment_body}
        response = requests.post(url, headers=self.headers, auth=self.auth, json=payload)
        response.raise_for_status()
        return response.json()

    def update_ticket_status(self, issue_key: str, transition_id: str) -> bool:
        """
        Update the status of a Jira ticket by performing a transition.
        You must provide the correct transition_id for the desired status.
        """
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"
        payload = {"transition": {"id": transition_id}}
        response = requests.post(url, headers=self.headers, auth=self.auth, json=payload)
        response.raise_for_status()
        return response.status_code == 204
    
    def get_transitions(self, issue_key: str,) -> bool:
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"
        
        response = requests.get(url, headers=self.headers, auth=self.auth)
        response.raise_for_status()
        return response.json().get("transitions", [])

    def update_ticket_dates(self, issue_key: str, start_date: str = None, end_date: str = None) -> bool:
        """
        Update the start date and/or end date (due date) of a Jira ticket.
        Dates should be in ISO format: 'YYYY-MM-DD'.
        """
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        fields = {}
        if start_date:
            fields["customfield_10015"] = start_date  # Replace with your Jira instance's actual custom field ID for start date
        if end_date:
            fields["duedate"] = end_date  # 'duedate' is standard for end/due date in Jira

        if not fields:
            raise ValueError("At least one of start_date or end_date must be provided.")

        payload = {"fields": fields}
        response = requests.put(url, headers=self.headers, auth=self.auth, json=payload)
        response.raise_for_status()
        return response.status_code == 204
    
    def delete_all_comments(self, issue_key: str) -> None:
        """
        Delete all comments from a Jira ticket.
        """
        comments = self.fetch_ticket_comments(issue_key)
        for comment in comments:
            comment_id = comment["id"]
            url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment/{comment_id}"
            response = requests.delete(url, headers=self.headers, auth=self.auth)
            response.raise_for_status()

    def fetch_ticket_by_id(self, issue_key: str) -> Ticket:
        """
        Fetch a single Jira ticket by its issue key.
        """
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        response = requests.get(url, headers=self.headers, auth=self.auth)
        response.raise_for_status()
        issue = response.json()
        fields = issue.get("fields", {})
        ticket: Ticket = {
            "id": issue.get("key", ""),
            "title": fields.get("summary", ""),
            "description": fields.get("description", ""),
            "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
            "status": fields.get("status", {}).get("name") if fields.get("status") else None,
            "start_date": fields.get("customfield_10015"),
            "due_date": fields.get("duedate"),
        }
        return ticket


# Example usage:
# jira_service = JiraService.get_instance()
# jira_service.delete_all_comments("APP-1")
# jira_service.delete_all_comments("APP-4")
# jira_service.update_ticket_dates(
#     issue_key="APP-1",  # Replace with your actual issue key
#     start_date="2025-10-01",
#     end_date="2025-10-31"
# )
# issue_key = "APP-1" 
# transitions = jira_service.get_transitions(issue_key)
# print(f"Transitions for {issue_key}: {json.dumps(transitions, indent=2)}")

# import os
# import requests

# JIRA_URL = os.getenv("JIRA_URL")
# JIRA_EMAIL = os.getenv("JIRA_EMAIL")
# JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# url = f"{JIRA_URL.rstrip('/')}/rest/api/2/field"
# response = requests.get(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN), headers={"Accept": "application/json"})
# response.raise_for_status()
# fields = response.json()

# import json
# print(json.dumps(fields, indent=2))

# comments_to_add =[
#     "Please ensure the login flow covers both email/password and social login options as discussed in the last sprint planning.",
#     "The UX team has provided updated wireframes for the login screens. Please refer to the latest designs in Figma.",
#     "Let's make sure error messages are clear and actionable for users who enter incorrect credentials."
# ]
# jira_service = JiraService.get_instance()
# for comment in comments_to_add:
#     jira_service.add_comment("APP-1", comment)  # Replace with your actual issue key