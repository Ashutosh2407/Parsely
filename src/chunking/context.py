from langchain_core.documents import Document
from src.chunking.base import BaseChunker

class ChunkingContext():
    """
    The Context in Strategy Pattern.
    Holds a reference to a strategy and delegates to it.
    ingest.py only ever touches this class — never concrete chunkers.
    """
    def __init__(self, strategy: BaseChunker):
        self._strategy = strategy

    def set_strategy(self, strategy: BaseChunker):
        self._strategy = strategy

    def chunk(self, docs: list[Document]) -> list[Document]:
        return self._strategy.split(docs)
    
    def stats(self, docs: list[Document]) -> dict:
        return self._strategy.stats(docs=docs)
