from abc import ABC, abstractmethod
from ragas.metrics import NonLLMContextRecall
from ragas.metrics.collections import Faithfulness,AnswerRelevancy,ContextPrecision, ContextRecall

class BaseMetric(ABC):
    name: str #used as csv column name

    @abstractmethod
    async def score(self, item: dict)-> float:
        pass

class FaithfulnessMetric(BaseMetric):
    name = "faithfulness"

    def __init__(self,llm):
        self.metric = Faithfulness(llm=llm)
        print(f"{FaithfulnessMetric.name} created...")
    
    async def score(self,item:dict) -> float:
        result = await self.metric.ascore(
                        user_input=item["question"],
                        response=item["answer"],
                        retrieved_contexts= "\n\n".join(item["contexts"])
                    )
        return result.value      
        

class AnswerRelevancyMetric(BaseMetric):
    name = "answer_relevance"

    def __init__(self,llm,embeddings):
        self.metric = AnswerRelevancy(llm=llm,embeddings=embeddings)
        print(f"{AnswerRelevancyMetric.name} created...")

    async def score(self,item:dict)-> float:
        result = await self.metric.ascore(
                        user_input=item["question"],
                        response=item["answer"],    
                    )
        return result.value
    
class ContextPrecisionMetric(BaseMetric):
    name = "context_precision"

    def __init__(self,llm):
        self.metric = ContextPrecision(llm= llm)
        print(f"{ContextPrecisionMetric.name} created...")

    async def score(self, item:dict) -> float:
        result = await self.metric.ascore(
                        user_input= item["question"],
                        retrieved_contexts= item["contexts"],
                        reference= item["ground_truth"],
                    )
        return result.value
                    

class ContextRecallMetric(BaseMetric):
    name= "recall@6"

    def __init__(self, llm):
        self.metric = ContextRecall(llm=llm)

    async def score(self,item:dict) -> float:
        result = await self.metric.ascore(
            user_input= item["question"],
            retrieved_contexts= item["contexts"],
            reference= item["ground_truth"]
        )
        return result.value


