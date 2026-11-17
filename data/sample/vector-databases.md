# Vector Databases

A vector database stores embeddings, which are lists of numbers that represent
the meaning of a piece of text. Texts with similar meaning have vectors that
point in similar directions.

## Similarity search

To compare two vectors, most systems use cosine similarity: the cosine of the
angle between them. A score near 1 means very similar meaning. A score near 0
means unrelated.

Searching millions of vectors one by one would be slow, so vector databases
build an approximate nearest neighbour index. Qdrant uses HNSW, a layered graph
that lets a search jump quickly towards the closest vectors.

## Payloads and filters

Alongside each vector, Qdrant stores a JSON payload such as the original text,
the file name and the page number. Searches can filter on the payload, for
example "only chunks from the HR handbook".
