from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str | None = None


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool