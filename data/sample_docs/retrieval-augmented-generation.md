# Retrieval-Augmented Generation

Retrieval-augmented generation, usually shortened to RAG, is a way of
answering questions with a large language model while grounding the answer
in documents the model did not see during training. Instead of relying only
on what the model memorized, the system first searches a knowledge base for
passages relevant to the question, then gives those passages to the model
along with the question.

The pipeline has three stages. First, documents are prepared ahead of time:
they are loaded, split into chunks, turned into embedding vectors, and
stored in a vector database. Second, at query time, the user's question is
embedded the same way and the nearest chunks are retrieved. Third, the
retrieved chunks are placed into a prompt that instructs the model to answer
using only that context.

Grounding matters because language models hallucinate. A model asked a
question it cannot answer will often produce a fluent, confident, and wrong
response. RAG reduces that risk in two ways: the model has real text to draw
from, and the answer can carry citations back to the source documents, so a
reader can verify every claim.

RAG is not a replacement for fine-tuning. Fine-tuning teaches a model new
behavior or style; RAG gives it new facts at query time. For a knowledge
base that changes often — company docs, product manuals, research notes —
RAG is usually the cheaper and more maintainable choice, because updating
the index is far easier than retraining a model.
