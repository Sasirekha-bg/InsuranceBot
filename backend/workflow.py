# workflow.py

from typing import TypedDict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Define state
class ChatState(TypedDict):
    user_input: str
    is_insurance_related: Optional[bool]
    retrieved_answer: Optional[str]
    final_answer: Optional[str]

# Load embedding model and FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "vectorstore/insurance_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = ChatGroq(model="llama3-70b-8192")

# Node 1: Check if the question is about insurance
def check_intent(state: ChatState) -> ChatState:
    prompt = PromptTemplate.from_template(
        "Is the following question about insurance? Answer only 'yes' or 'no'.\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["user_input"]})
    state["is_insurance_related"] = "yes" in result.lower()
    return state

# Node 2: Retrieve relevant context using RAG


# Updated prompt template for RAG
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert insurance advisor.

Use the context below to answer the user's question clearly and concisely. If the user is asking for a comparison, provide it in a structured format:
- Plan names
- Key features
- Sum insured range
- Special coverage (e.g., maternity, diabetes)
- Premium estimate (if available)
- Pros and cons
- Summary recommendation

Always aim to be informative, neutral, and user-friendly.
"""),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

def retrieve_info(state: ChatState) -> ChatState:
    query = state["user_input"]
    
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Get structured response using prompt chain
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    # Add source URLs from metadata (if available)
    sources = set()
    for doc in docs:
        if 'source' in doc.metadata:
            sources.add(doc.metadata['source'])

    if sources:
        formatted_sources = "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in sources)
        answer += formatted_sources

    state["retrieved_answer"] = answer
    return state




# Node 3: Simplify and format the answer
def simplify_answer(state: ChatState) -> ChatState:
    prompt = f"""
Please rewrite the following answer to make it:
- Easy to understand (like explaining to a 10-year-old)
- Well-formatted using markdown or bullet points
- Friendly and supportive in tone

Also, if any plan details are missing, politely mention that more information may be needed for a complete comparison.

Here is the answer to format:

{state['retrieved_answer']}
"""
    result = llm.invoke(prompt)

    # Add soft follow-up CTA
    follow_up = """
💡 Let me know if you'd like help with:
- Comparing plans side by side
- Understanding confusing terms
- Getting suggestions based on your age or health profile!
"""
    state["final_answer"] = result.strip() + "\n" + follow_up.strip()
    return state



# Node 4: Fallback for non-insurance questions
def general_fallback(state: ChatState) -> ChatState:
    state["final_answer"] = "🤖 I specialize in insurance-related questions. Could you ask me something about insurance?"
    return state

# Build the LangGraph workflow
builder = StateGraph(ChatState)

builder.set_entry_point("check_intent")
builder.add_node("check_intent", check_intent)
builder.add_node("retrieve_info", retrieve_info)
builder.add_node("simplify_answer", simplify_answer)
builder.add_node("general_fallback", general_fallback)

# Route based on insurance intent
builder.add_conditional_edges(
    "check_intent",
    lambda state: "insurance" if state["is_insurance_related"] else "general",
    {
        "insurance": "retrieve_info",
        "general": "general_fallback"
    }
)

builder.add_edge("retrieve_info", "simplify_answer")
builder.add_edge("simplify_answer", END)
builder.add_edge("general_fallback", END)

# Compile the graph
chat_graph = builder.compile()

def get_insurance_bot_response(user_query: str) -> str:
    response = chat_graph.invoke({"user_input": user_query})
    return response["final_answer"]

