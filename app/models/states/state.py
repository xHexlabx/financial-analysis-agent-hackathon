from typing_extensions import TypedDict , List

class State(TypedDict):
    queries : List[str]
    queries_type : List[int]
    outputs : List[str]

class InputState(TypedDict) :
    queries : List[str]
    queries_type : List[int]
