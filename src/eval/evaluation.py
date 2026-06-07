from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness,AnswerRelevancy,ContextPrecision
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
import asyncio
import os
import csv
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
#dataset = Dataset.from_list(RESULTS)
dataset_dict = load_dataset("json", data_files="src/eval/datasets/eval_dataset.json")
dataset = dataset_dict["train"]



#GROQ
# groq_client = AsyncOpenAI(
#     api_key=os.environ["GROQ_API_KEY"],
#     base_url="https://api.groq.com/openai/v1"
# )
#ragas_llm = llm_factory(model="llama-3.3-70b-versatile",client=groq_client)

#Google Gemini
# gemini_client = AsyncOpenAI(
#     api_key=os.environ["GOOGLE_API_KEY"],
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# ragas_llm = llm_factory(model="gemini-2.0-flash", client=gemini_client)

#Open AI
openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
print("Key loaded:", os.environ["OPENAI_API_KEY"][:8], flush=True)  # print first 8 chars to verify
ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
# ragas_llm = LangchainLLMWrapper(
#     ChatOpenAI(
#         model="gpt-4o-mini", 
#         client=openai_client
#         )
# )
logger.info("ragas_llm created.")

#Embeddings
ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
logger.info("ragas_embedding created.")

faithfulness_score = Faithfulness(llm=ragas_llm)
logger.info("faithfulness_score created.")
answer_relevancy_score = AnswerRelevancy(llm=ragas_llm,embeddings = ragas_embeddings)
logger.info("answer_relevancy_score created.")
context_precision_score = ContextPrecision(llm=ragas_llm)
logger.info("context_precision_score created.")

async def run_ragas(dataset):
    results = []
    logger.info("results=[] created.")
    for i,row in enumerate(dataset):
        logger.info(f"Starting record {i}...")
        try:
            f = await asyncio.wait_for(
                faithfulness_score.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts= "\n\n".join(row["contexts"])
            )
            ,timeout=300) 
            logger.info(f"Faithfulness for record {i} calculated... {f.value}")
        except asyncio.TimeoutError:
            logger.info(f"Q{i} timed out for faithfuless, skipping. Aborting process.")
            break 
        try:
            ars = await asyncio.wait_for(
                answer_relevancy_score.ascore(
                user_input=row["question"],
                response=row["answer"],    
            )
            ,timeout=300)
            logger.info(f"Answer relevancy score for record {i} calculated....{ars.value}")
        except asyncio.TimeoutError:
            logger.info(f"Q{i} timed out for answer relevancy, skipping. Aborting process.")
            break 
        try:
            cps = await asyncio.wait_for(
                context_precision_score.ascore(
                user_input= row["question"],
                retrieved_contexts= row["contexts"],
                reference= row["ground_truth"],
            ), 
            timeout=300
            )
            logger.info(f"Context precision for record {i} calculated...{cps.value}")
        except asyncio.TimeoutError:
            logger.info(f"Q{i} timed out for context preciosion, skipping. Aborting process.")
            break 
        results.append({
            "question": row["question"],
            "answer": row["answer"],
            "faithfulness": f.value,
            "answer_relevance": ars.value,
            "context_precision": cps.value,
        })
        logging.info(f"Record {i} is done.")
        await asyncio.sleep(3) 
    return results

result = asyncio.run(run_ragas(dataset=dataset))
if not result:
    print("No results returned — all questions timed out or failed.")
else:    
    with open("src/eval/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)

