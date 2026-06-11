import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.callbacks import get_openai_callback
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from src.api.schemas import AnswerSchema, QueryRequest
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="test-index", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":5})

# ---------------------------------------------------------------------------
# LLM chain
# ---------------------------------------------------------------------------
def build_llm_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096
    )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful RAG assistant. Answer using ONLY the retrieved context.
        At the end of the end of your answer, OUTPUT a json block (and nothing after it) in this exact format:
         ```json
         {{
            "confidence":0.92,
            "citations":["[Chunk 0]","[Chunk 1]"]
         }}
         ```
         """),
        ("human","Question:{question}\n\nContext:\n{context}\n\nProvide a structured answer."),
    ])
    return prompt | llm

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message":"Hello World!"}

@app.get("/health")
async def check_health():
    return {"status": "ok"}

@app.post("/query", response_model=AnswerSchema)
async def query(request: QueryRequest)-> AnswerSchema:
    question = request.question
    # 1. Real Pinecone retrieval
    docs = retriever.invoke(question)
    # 2. Build context from your real chunks
    context = "\n\n".join(
        f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
        f"Ticker: {doc.metadata.get('ticker', '?')} | "
        f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    # 3. Generate structured answer
    chain = build_llm_chain()
    try:
        with get_openai_callback() as cb:
            result: AnswerSchema = await chain.ainvoke(
                {"question":question, "context":context}
            )
        result.prompt_tokens = cb.prompt_tokens
        result.completion_tokens = cb.completion_tokens
        result.cost_usd = cb.total_cost
        logger.info(f"Prompt tokens: {cb.prompt_tokens}")
        logger.info(f"Completion tokens: {cb.completion_tokens}")
        logger.info(f"Total tokens: {cb.total_tokens}")
        logger.info(f"Cost: ${cb.total_cost:.6f}")
    except Exception as e:
        raise HTTPException(status_code=502,detail=f"LLM call failed: {e}")
    
    # 4. Attach Real source metadata
    result.sources.extend([
        {
            "chunk_id": i,
            "source": doc.metadata.get("source", "?"),
            "ticker": doc.metadata.get("ticker", "?"),
            "year": doc.metadata.get("year", "?"),
            "preview": doc.page_content[:200],
        }
    for i,doc in enumerate(docs)
    ])
    return result

    