from src.chunking.strategies import FinanceSectionChunker,FixedSizedChunker,RecursiveTextChunker
from src.chunking.context import ChunkingContext

_REGISTRY = {
    "fixed": FixedSizedChunker,
    "recursive": RecursiveTextChunker,
    "section": FinanceSectionChunker
}

def get_chunking_context(strategy: str, **kwargs) -> ChunkingContext:
    """
    Returns a ChunkingContext loaded with the requested strategy.
    
    Usage:
        ctx = get_chunking_context("section", max_section_size=2000)
        chunks = ctx.chunk(docs)
    """
    if strategy not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from : {list(_REGISTRY)}")
    strategy_object = _REGISTRY[strategy](**kwargs)
    return ChunkingContext(strategy_object)