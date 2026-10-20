#just basic linear graph workflow 

from langgraph.graph import StateGraph , START , END
from typing import TypedDict

#create state
class BMI(TypedDict) :
    wt : float
    ht : float 
    bmi : float

def BMI_cal(state: BMI) -> BMI :

    weight = state['wt']
    height = state['ht']

    bmi = weight/(height**2)

    state['bmi'] = round(bmi, 2)

    return state

#define graph 
graph = StateGraph(BMI)

#nodes
graph.add_node('calculate_bmi', BMI_cal)

#edges
graph.add_edge(START,'calculate_bmi')
graph.add_edge('calculate_bmi',END)

#compile
workflow = graph.compile()

#execute
intial_state = {'wt':80, 'ht':1.73}

final_state = workflow.invoke(intial_state)

print(final_state)