# Vector Search

Vector search is the problem of finding the stored vectors closest to a
query vector. In a RAG system the stored vectors are chunk embeddings and
the query vector is the embedding of the user's question. The closest chunks
are the ones most likely to contain the answer.

The naive approach — comparing the query against every stored vector — works
fine for thousands of chunks but gets slow as the index grows. Production
systems use approximate nearest neighbor indexes instead. HNSW, the
algorithm behind Chroma's default index, organizes vectors into a layered
graph so a search hops toward the neighborhood of the query instead of
scanning everything. It trades a small amount of accuracy for a large gain
in speed.

Distance is a choice, not a given. Cosine distance measures the angle
between vectors and ignores their length, which suits text embeddings.
Euclidean distance measures straight-line distance and is sensitive to
magnitude. This repo uses cosine space, matching the normalized embeddings
produced at index time.

Vector search has a blind spot: it ranks by overall similarity, so a chunk
that mentions the question's key term exactly once can lose to a chunk that
is merely on the same general topic. BM25 keyword search has the opposite
strength — it rewards exact term matches. Hybrid retrieval runs both and
merges the rankings with reciprocal rank fusion, so a chunk that scores
well on either method has a path into the final top-k.
