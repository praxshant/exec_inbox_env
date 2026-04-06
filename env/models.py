# env/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# Public-facing email (no hidden ground truth)
class PublicEmail(BaseModel):
    id: str
    sender: str
    subject: str
    body: str


# Internal email with ground truth (used only by env + grader)
class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    label: Optional[Literal["important", "spam", "promo"]] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    deadline: Optional[int] = None  # time step by which action must be taken


# Task derived from email processing
class Task(BaseModel):
    id: str
    description: str
    deadline: int
    completed: bool = False


# What the agent sees
class Observation(BaseModel):
    inbox: List[PublicEmail]
    current_time: int
    pending_tasks: List[Task]
    completed_tasks: List[Task]


# What the agent does
class Action(BaseModel):
    type: Literal["classify", "reply", "prioritize", "archive", "noop"]
    email_id: Optional[str] = None
    label: Optional[Literal["important", "spam", "promo"]] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    content: Optional[str] = None


# Step-level reward
class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    reason: str