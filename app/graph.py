#states
from models.states.state import State, InputState

#langgraph libraries
from langgraph.graph import StateGraph , START , END

#import nodes
from nodes.generate_response import generate_response

#query sample
import pandas as pd

workflow = StateGraph(
    state_schema = State,
    input_state_schema = InputState
)

#add nodes
workflow.add_node("generate_response",generate_response)

#add edges
workflow.add_edge(START, "generate_response")
workflow.add_edge("generate_response", END)

#graph compile
graph = workflow.compile()

#write graph
with open("./graph.png" , "wb") as file :
    file.write(graph.get_graph().draw_mermaid_png())

queries = pd.read_csv("./data/test.csv")['query'][0 : 4]
queries_type = [0, 1, 1, 0]
answer = graph.invoke({"queries": queries, "queries_type" : queries_type})['outputs']

print("Answer =" , answer)
