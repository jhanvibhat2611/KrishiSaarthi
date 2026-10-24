from typing import Protocol, Sequence
from ..schemas import Passage, Query, RetrievalResult


class Retriever(Protocol):
    retriever_id: str

    def index(self, passages: Sequence[Passage]) -> None: ...
    def search(self, query: Query, k: int) -> Sequence[RetrievalResult]: ...
