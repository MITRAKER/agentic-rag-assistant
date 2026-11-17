# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a search step with a language model.
Instead of answering from memory alone, the system first retrieves passages
from a knowledge base and then asks the model to answer using only those passages.

## Why RAG?

Language models are trained on a fixed snapshot of data, so they do not know
about private documents or recent events. They can also "hallucinate":
produce fluent but unsupported statements. Grounding the answer in retrieved
passages reduces hallucination and lets the system cite its sources.

## The two phases

The indexing phase runs ahead of time: documents are loaded, split into chunks,
embedded into vectors and stored in a vector database.

The query phase runs for every question: the question is embedded, the most
similar chunks are retrieved, and the language model writes an answer that
cites those chunks.
