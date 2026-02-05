from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uvicorn
import os

from rag import initialize_rag, ask_question, create_agent

# variables
load_dotenv()

app = FastAPI()

# Global Agent Executor
agent_executor = None

# files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# RAG Pipeline 
@app.on_event("startup")
async def startup_event():
    global agent_executor
    print("Initializing RAG system...")
    initialize_rag()
    print("Initializing Agent...")
    agent_executor = create_agent()

class QueryRequest(BaseModel):
    question: str
    search_type: Optional[str] = "similarity"
    use_agent: Optional[bool] = False
    chat_history: List[Dict[str, Any]] = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_endpoint(query: QueryRequest):
    global agent_executor
    try:
        if query.use_agent:
            if agent_executor is None:
                 agent_executor = create_agent()
            
            # LangGraph invocation
            # Use a static thread_id for this demo, effectively making it a single global session
            config = {"configurable": {"thread_id": "web-session-1"}}
            
            response = agent_executor.invoke(
                {"messages": [{"role": "user", "content": query.question}]},
                config=config
            )
            
            # Extract the last message content
            last_message = response["messages"][-1]
            return {"answer": last_message.content}
        else:
            # Pass chat_history to ask_question
            response = ask_question(query.question, query.search_type or "similarity", query.chat_history)
            return {"answer": response}
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"answer": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
