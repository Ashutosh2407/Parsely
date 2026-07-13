import os, re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.callbacks import get_openai_callback
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from src.langraph.test_graph import graph
from src.api.schemas import AnswerSchema, QueryRequest, SourceSchema
from src.weaviate.query import query_all
from src.langraph.memory_agent import chat
import logging
import json
from fastapi.responses import StreamingResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
# -------------
# Retriever
# -------------
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="test-index", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":5})

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message":"Hello World!"}

@app.get("/health")
async def check_health():
    return {"status": "ok"}

@app.post("/query")
async def query(request: QueryRequest):
    question = request.question
    memory_result = chat(session_id=request.thread_id,
                         user_message=question,
                         db = request.db)
    context = memory_result["retrieved_context"]
    question = memory_result["resolved_query"]

    initial_state = {
        "current_query": question,
        "retrieved_context": context,
        "chunks": [],
        "answer": "",
        "grade": 0.0,
        "confidence": 0.0,
        "blocked": False,
        "blocked_reason": "",
        "approved": False,
        "gen_count": 0,
    }
    # Step C: same thread_id drives BOTH graphs' checkpointers
    config = {"configurable":{"thread_id": request.thread_id}}
    #1. Pinecone retrieval for sources only
    if request.db == "pinecone":
        docs = retriever.invoke(question)
    else:
    #2. Weaviate Retrieval for sources only
        raw = query_all(question,limit=5,search_type="hybrid")
        docs = [
            Document(
                page_content=obj.properties.get("text"),
                metadata = {
                    "ticker": obj.properties.get("ticker"),
                    "year": obj.properties.get("year"),
                    "quarter": obj.properties.get("quarter"),
                    "filing_type": obj.properties.get("filing_type"),
                    "sources": obj.properties.get("source"),
                }
            ) 
            for obj in raw
        ]
    
    #3. Generate metadata here 
    sources = [
        {
            "chunk_id": i,
            "source": doc.metadata.get("source", "?"),
            "ticker": doc.metadata.get("ticker", "?"),
            "year": str(doc.metadata.get("year", "?")),
            "preview": doc.page_content[:200],
        }
    for i,doc in enumerate(docs)
    ]
    
    #4. Streaming helper method
    async def stream_response():
        full_response = ""
        final_state = {}
        try:
            with get_openai_callback() as cb:
                async for stream_type,event_data in graph.astream(initial_state, 
                                                                  config=config,
                                                                  stream_mode=["messages","updates"],
                                                                  subgraphs=True):
                    if stream_type=="updates":
                        final_state.update(event_data)
                    if stream_type == "updates" and "__interrupt__" in event_data:
                        payload = event_data["__interrupt__"][0].value
                        yield f"data: {json.dumps({'type':'interrupt','payload':payload})}\n\n"
                        return
                    if stream_type=="messages":
                        msg_chunk, _ = event_data
                        token = msg_chunk.content
                        full_response += token
                        if "```json" not in full_response:
                            yield f"data:{json.dumps({'type': 'token','content':token})}\n\n"      
        except Exception as e:
            yield f"data:{json.dumps({'type':'error','detail': str(e)})}\n\n"
            return
        citations = []
        match = re.search(r"```json\s*(\{.*?\})\s*```", full_response, re.DOTALL)
        if match:
            try:
                meta = json.loads(match.group(1))
                citations = meta.get("citations", [])
            except json.JSONDecodeError as e:
                logger.info(f"Could not parse metadata: {e}")
        #Step 6: Build the final AnswerSchema and send it as one closing event 
        final = AnswerSchema(**{
            "answer": final_state.get("answer",full_response.split("```json")[0].strip()),
            "confidence": final_state.get("confidence",0),
            "citations": citations,
            "sources": sources,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "cost_usd": cb.total_cost
        })
        
        yield f"data:{json.dumps({"type":"final", "data": final.model_dump()})}\n\n"
        yield f"data:[DONE]\n\n"
    
    #7.Return the stream to FastAPI
    return StreamingResponse(stream_response(),media_type="text/event-stream")