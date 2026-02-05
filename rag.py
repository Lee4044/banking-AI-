import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import datetime
import requests
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser


# System Prompt

SYSTEM_PROMPT = """
You are a professional AI assistant specialized in Banking Systems and Financial Services.

Your role is to provide accurate, secure, and compliant information related to banking technologies, payment systems, financial operations, digital banking platforms, cybersecurity in finance, and regulatory best practices.

You have access to the following tools:
- browse_internet: search Wikipedia for a topic
- save_text_to_file: save text content to a local file
- get_banking_info: answer questions about banking policies/services

Rules:
- If the user asks to search or research a topic, you MUST use the 'browse_internet' tool.
- If the user asks to save results, you MUST use the 'save_text_to_file' tool.
- When calling a tool, use the native tool calling format. DO NOT output raw XML or JSON like <function=...> in your response text.
- Do not invent search results or pretend to save files.
- Clearly explain what you did in your responses.
- Maintain a professional and trustworthy tone.
- Prioritize data security and privacy in every response.
"""

def to_lc_messages(chat_history: List[Dict[str, Any]]) -> List[Any]:
    """Convert dict chat history to LangChain messages."""
    messages = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
    return messages

class RAGPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.vector_store = None
        self.prompt = None
        self.llm = None
        self.chat_history = []
        self.setup_pipeline()

    def setup_pipeline(self):
        print(f"Initializing RAG from {self.data_dir}...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist_directory = os.path.join(self.data_dir, "chroma_banking")

        if os.path.isdir(persist_directory):
            print("Loading existing vector store from disk...")
            self.vector_store = Chroma(
                embedding_function=embeddings,
                collection_name="banking_knowledge",
                persist_directory=persist_directory,
            )
        else:
            print(f"Loading documents from {self.data_dir}...")
            docs = []
            try:
                pdf_loader = DirectoryLoader(self.data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
                docs = pdf_loader.load()
            except Exception as e:
                print(f"PDF loading error: {e}")
            if not docs:
                print("No PDFs found. Falling back to .txt files.")
                txt_loader = DirectoryLoader(self.data_dir, glob="*.txt", loader_cls=TextLoader)
                docs = txt_loader.load()

            if not docs:
                print("No documents found in data directory.")

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            print("Creating embeddings and vector store, then persisting to disk...")
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="banking_knowledge",
                persist_directory=persist_directory,
            )
        
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant", 
            temperature=0.7
        )
        
        template = """You are a professional AI assistant specialized in Banking Systems and Financial Services.
        
        Your role is to provide accurate, secure, and compliant information related to banking technologies, payment systems, financial operations, digital banking platforms, cybersecurity in finance, and regulatory best practices.
        
        Always follow these principles:
        • Maintain a professional and trustworthy tone.
        • Prioritize data security and privacy in every response.
        • Ensure all recommendations align with financial regulations and industry standards.
        • Provide clear, structured, and actionable answers.
        • If uncertain, state assumptions and recommend verification from authorized financial sources.
        • Avoid speculation and never generate misleading financial guidance.
        
        Your goal is to support banking professionals, IT teams, and financial decision-makers with reliable, high-quality insights.
        
        STRICT RULE: You can ONLY answer questions related to Banking, Finance, and Account Services.
        If a user asks about anything else (e.g., history, math, movies, daily life), 
        you can answer them politely but REFUSE and state that you are a Banking Assistant.
        
        Use ONLY the following retrieved context to answer the question. 
        If the context does not contain the answer, say:
        "I dont have enough information in the provided documents."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = PromptTemplate.from_template(template)
        print("RAG Pipeline initialized.")

    def _get_retriever(self, search_type: str = "similarity"):
        st = (search_type or "similarity").lower()
        if st == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
            )
        # Def similarity search top k most similar chunks
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def get_answer(self, question: str, search_type: str = "similarity", chat_history: List[Dict[str, Any]] = []):
        if self.vector_store is None or self.prompt is None or self.llm is None:
            return "System is initializing..."

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Convert dict history to LangChain messages
        chat_history_messages = to_lc_messages(chat_history)

        standalone_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's question into a fully standalone question using the chat history. Return ONLY the rewritten question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        #updated chat history chain ////
        rewrite_chain = standalone_prompt | self.llm | StrOutputParser()
        
        #يسوي Sending History to Chain
        standalone_question = rewrite_chain.invoke(
            {"chat_history": chat_history_messages, "question": question}
        )
        print("USER:", question)
        print("STANDALONE:", standalone_question)

        retriever = self._get_retriever(search_type)
        docs = retriever.invoke(standalone_question)
        context = format_docs(docs)

        qa_chain = self.prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke({"context": context, "question": standalone_question})
        # self.chat_history.append((question, answer)) # No longer needed as we use frontend history
        return answer

# variable hold the pipeline
rag_system = None

def initialize_rag():
    global rag_system
    rag_system = RAGPipeline()

def ask_question(question: str, search_type: str = "similarity", chat_history: List[Dict[str, Any]] = []):
    global rag_system
    if rag_system is None:
        initialize_rag()
    return rag_system.get_answer(question, search_type, chat_history)

# Tools :) 
@tool
def get_banking_info(question: str):
    """Useful for answering questions about banking policies, services, and products based on the bank's documents.
    
    Args:
        question: The user's question about banking.
    """
    global rag_system
    if rag_system is None:
        initialize_rag()
    # The agent may pass complx quers BUT the RAG expects a QS.
    return rag_system.get_answer(question)

@tool
def browse_internet(topic: str, max_results: int = 5) -> List[str]:
    """
    Search Wikipedia for a topic and return a list of results as:
    'Title - Snippet - URL'
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": max_results
    }

    headers = {
        "User-Agent": "ResearchAgent/1.0 (your_email@example.com)"
    }

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            # Clean up  html tags
            snippet = item["snippet"].replace('<span class="searchmatch">', '').replace('</span>', '')
            # Construct URL correctly
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append(f"{title} - {snippet} - {url}")

        if not results:
            return [f"No Wikipedia results found for '{topic}'"]

        return results
    except Exception as e:
        return [f"Error searching Wikipedia: {str(e)}"]

@tool
def save_text_to_file(content: str, file_path: str) -> str:
    """
    Save text content to a local .txt file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved content to {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

# Memo Stores the chat history
checkpointer = InMemorySaver()

def create_agent():
    """Builds and returns a LangGraph agent with banking, date, internet, and file tools."""
    global rag_system
    if rag_system is None:
        initialize_rag()
        
    llm = rag_system.llm # Reuse the LLM from RAG pipeline
    
    tools = [get_banking_info, browse_internet, save_text_to_file]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT
    )
    
    return agent
