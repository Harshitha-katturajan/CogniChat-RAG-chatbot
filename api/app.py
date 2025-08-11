from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai  import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

app=FastAPI(
    title="Langchain Server",
    version="0.0.1",
    description="A Simple API Server",
    # root_path="/api" 
)
add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()
##ollama llama2
llm=Ollama(model="llama2")
prompt1=ChatPromptTemplate.from_template("write me an essay about {topic}with 100 words")
prompt2=ChatPromptTemplate.from_template("write me an poem about {topic}with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)
add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

# if __name__=="__main__":
#   uvicorn.run(app,host="localhost",port=8001)
    
from typing import Union

from fastapi import FastAPI

# app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__=="__main__":
  uvicorn.run(app,host="localhost",port=8001)