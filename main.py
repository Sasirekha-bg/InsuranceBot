# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.workflow import get_insurance_bot_response
  # import from workflow.py

class Query(BaseModel):
    message: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(query: Query):
    try:
        result = get_insurance_bot_response(query.message)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Backend is running"}
