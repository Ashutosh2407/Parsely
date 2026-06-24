from datasets import load_dataset
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from src.eval.metrics import FaithfulnessMetric,ContextPrecisionMetric,AnswerRelevancyMetric, ContextRecallMetric
from openai import AsyncOpenAI
import asyncio
import os
import csv
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#Read the Data
dataset_dict = load_dataset("json", data_files="src/eval/datasets/eval_dataset_weaviate.json")
dataset_intermediate = dataset_dict["train"]
dataset = [dict(row) for row in dataset_intermediate]

#Start time
start = time.time()

#Open AI
openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
print("Key loaded:", os.environ["OPENAI_API_KEY"][:8], flush=True)  
ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
logger.info("ragas_llm created.")

#Embeddings
ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
logger.info("ragas_embedding created.")


#Batch Helper Function
def batch_helper(dataset,size):
    for i in range(0,len(dataset),size):
        yield dataset[i:i+size]

#Setup The Configs for Metrics
metrics = [
    FaithfulnessMetric(llm = ragas_llm),
    AnswerRelevancyMetric(llm=ragas_llm,embeddings=ragas_embeddings),
    ContextPrecisionMetric(llm = ragas_llm),
    ContextRecallMetric(llm = ragas_llm)
]


async def run_ragas(dataset, metrics):
    results = []
    logger.info("results=[] created.")

    for i, batch_items in enumerate(batch_helper(dataset,size= 5)):
        logger.info(f"Processing batch {i+1}, questions {i*5+1} to {min((i+1)*5, len(dataset))}")
        for item in batch_items:
            record = {
                "strategy": item["strategy"],
                "question": item["question"],
                "answer": item["answer"],
            }
            
            for metric in metrics:
                try:
                    score = await asyncio.wait_for(
                        metric.score(item),
                        timeout = 600
                    )
                    record[metric.name] = score
                    logger.info(f"{metric.name} for record {i} calculated... {score}")
                except asyncio.TimeoutError:
                    logger.info(f"Q{i} timed out for faithfuless, skipping. Aborting process.")
                    record[metric.name] = None
            
            results.append(record)   

        file_exists = os.path.exists("src/eval/results.csv")
        with open("src/eval/results.csv", "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results[-len(batch_items):])
        logger.info(f"Batch {i+1} saved.")
        await asyncio.sleep(10) 
    return results

result = asyncio.run(run_ragas(dataset=dataset,metrics=metrics))
end = time.time()
print(f"Total time required: {end - start:.2f}s")
