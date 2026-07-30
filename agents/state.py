from typing_extensions import TypedDict

class AgentState(TypedDict):
    question:str
    context:str
    documents:list
    answer:str
    route: str

