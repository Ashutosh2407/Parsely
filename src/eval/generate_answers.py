# 1)Load test.json
# 2)Generate answers for the question using My LLM
# 3)Compare the answers with test.json using RAG evaluation on metrics
#     a) faithfulness
#     b)answer_relevance
#     c)context_precision
# 4)Once the result is generated, save it in eval_dataset.json

import os, json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv
from src.api.main import build_llm_chain
from src.api.schemas import AnswerSchema,EvalQuestionSchema
import asyncio
import logging
import time 

load_dotenv()

with open("src/eval/test_set.json", "r") as f:
    test_set = json.load(f)

os.makedirs("src/eval/datasets", exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(embedding=embeddings,index_name="test-index")
retriever = vectorstore.as_retriever(search_kwargs={"k":5})
RESULTS = []

async def get_answer(request: EvalQuestionSchema):
    for item in request:
        question = item["question"]
        #Pinecone retrieval
        docs = retriever.invoke(question)
    
        #Build context from your real chunks
        context = "\n\n".join(
            f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
            f"Ticker: {doc.metadata.get('ticker', '?')} | "
            f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        #Generate Structured Answer
        chain = build_llm_chain()
        try:
            result= await chain.ainvoke(
                {"question":question, "context":context}
            )
        except Exception as e:
            raise ValueError(f"Could not generate answer for {question}.")
        
        RESULTS.append(
            {
                "question":question,
                "answer":result.answer,
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in docs],
                "category": item["category"],
                "expected_source": item["expected_source"],
                "actual_sources": result.sources,
                "target_companies": item["target_companies"],
                "reference_period": item["reference_period"],
                "citations": result.citations,
            }
        )
        #break
        time.sleep(5)
    with open("src/eval/datasets/eval_dataset.json","w") as f:
        json.dump(RESULTS,f, indent=2)

result = asyncio.run(get_answer(test_set["questions"]))



