from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

llm=ChatOpenAI(model='gpt-4o-mini')

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages=state['messages']
    response=llm.invoke(messages)
    return {'messages':[response]}

checkpoint=MemorySaver()

graph=StateGraph(ChatState)

#Add Nodes

graph.add_node('chat_node',chat_node)

#Add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpoint)
    