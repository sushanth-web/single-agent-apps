from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph import add_messages
from IPython.display import Image

class Chatstate(BaseModel):
    messages : Annotated[list,add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")

def Chatnode(state:Chatstate) -> Chatstate:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

graph = StateGraph(Chatstate)

graph.add_node("model", Chatnode)

graph.add_edge(START,"model")

graph.add_edge("model",END)

graph = graph.compile(checkpointer=InMemorySaver())

Image(graph.get_graph().draw_mermaid_png())

while True:
    query = input("user:")
    if query.lower() in ["exit","quit","bye"]:
        break
    res = graph.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}}
    )

    print("Ai:",res["messages"][-1].content)
