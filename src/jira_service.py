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

# jira_service = JiraService.get_instance()
# issue_key = "APP-1"  # Replace with your actual issue key
# transitions = jira_service.get_transitions(issue_key)
# print(f"Transitions for {issue_key}: {json.dumps(transitions, indent=2)}")