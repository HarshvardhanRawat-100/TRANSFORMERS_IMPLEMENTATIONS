from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class batsman(TypedDict) :
    runs : int 
    ball : int
    fours : int
    six : int 

    sr: float
    bpb: float
    boundary_percent: float
    summary: str

def calculate_sr(state: batsman):

    sr = (state['runs']/state['ball'])*100
    
    return {'sr': sr}

def calculate_bpb(state: batsman):

    bpb = state['ball']/(state['fours'] + state['six'])

    return {'bpb': bpb}

def calculate_boundary_percent(state : batsman):

    boundary_percent = (((state['fours'] * 4) + (state['six'] * 6))/state['runs'])*100

    return {'boundary_percent': boundary_percent}

def summary(state: batsman):

    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percent']}
"""
    
    return {'summary': summary}

#creating graph
graph = StateGraph(batsman)

#add node
graph = StateGraph(batsman)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# edges

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()


intial_state = {
    'runs': 100,
    'ball': 50,
    'fours': 6,
    'six': 4
}

print(workflow.invoke(intial_state))