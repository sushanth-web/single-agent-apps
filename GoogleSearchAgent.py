from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model = "openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model = llm,
    tools = [search.run],
    system_prompt = "You are an google search agent",
    checkpointer = MemorySaver()
)

while True:
    query = input("User:")
    if query.lower() in ["exit","quit","bye"]:
        print("GoodBye")
        break
    response = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable":{"thread_id":"1"}}
        )
    print("AI:", response["messages"][-1].content) 