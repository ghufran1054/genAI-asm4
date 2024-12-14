import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import gradio as gr

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
PINE_API_KEY = os.getenv('PINE_API_KEY')

pc = Pinecone(api_key=PINE_API_KEY)
index = pc.Index('netsol-finance-asm4')
vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
model = ChatGroq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def category_decider(state: MessagesState):
    """
    Determine which retriever to use based on query.
    """
    message = state['messages'][-1].content.lower()
    return "vector_store" if "netsol" in message else "tavily"

def pinecone_retriever(state: MessagesState):
    query = state['messages'][-1].content
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)
    return {
        "messages": [query, " ".join([doc.page_content for doc in docs])]
    }

def tavily_retriever(state: MessagesState):
    query = state['messages'][-1].content
    results = tavily.search(query)['results']
    return {
        "messages": [query, " ".join([f"Title: {result['title']} Content: {result['content']}" for result in results])]
    }

def call_model(state: MessagesState):
    context = state['messages'][-1].content
    query = state['messages'][0].content
    messages = [
        ('system', "Answer based on context."),
        ('human', f"Query: {query}\nContext: {context}")
    ]
    response = model.invoke(messages)
    return {"messages": [response.content]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tavily", tavily_retriever)
workflow.add_node("vector_store", pinecone_retriever)
workflow.add_conditional_edges(START, category_decider)
workflow.add_edge("tavily", "agent")
workflow.add_edge("vector_store", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

def gradio_interface(query):
    response = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": 40}}
    )
    return response['messages'][-1].content

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your query", placeholder="Ask something..."),
    outputs=gr.Textbox(label="Response"),
    title="RAG Pipeline"
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch(share=True)

