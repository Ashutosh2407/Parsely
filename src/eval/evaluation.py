from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings.base import embedding_factory,LangchainEmbeddingsWrapper
from ragas.metrics.collections import Faithfulness,AnswerRelevancy,ContextPrecision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

#dataset = Dataset.from_list(RESULTS)
dataset_dict = load_dataset("json", data_files="src/eval/datasets/eval_dataset.json")
dataset = dataset_dict["train"]
# ragas_llm = LangchainLLMWrapper(
#     ChatGroq(
#         model = "gemma2-9b-it",
#         temperature=0,
#         api_key=os.environ.get("GROQ_API_KEY")
#     )
# )

# #Embeddings
# ragas_embeddings = LangchainEmbeddingsWrapper(
#     HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
# )

# faithfulness_score = Faithfulness(llm=ragas_llm)
# answer_relevancy_score = AnswerRelevancy(llm=ragas_llm)
# context_precision_score = ContextPrecision(llm=ragas_llm)

for row in dataset:
    print(row)