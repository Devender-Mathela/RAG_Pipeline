from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    context: str
    documents: list
    answer: str
    route: str
    search_strategy: str
    retry_count: int
    documents_relevant: bool
    evidence_status: str
    evidence_reason: str

