# Embeddings

An embedding is a list of numbers — a vector — that represents the meaning
of a piece of text. Similar meanings land close together in vector space,
so the distance between two embeddings is a rough measure of how related
the two texts are. This is what makes semantic search possible: instead of
matching keywords, the system matches meaning.

Embedding models are trained on large amounts of text so that paraphrases
end up near each other. A small local model such as all-MiniLM-L6-v2 turns a
sentence into a 384-dimensional vector and runs on an ordinary CPU in
milliseconds. Larger models produce richer vectors, but for many retrieval
tasks the small model is good enough, and it costs nothing to run.

Before comparing vectors, they are usually normalized to unit length. With
normalized vectors, cosine similarity and dot product give the same ranking,
which keeps the math simple and the results comparable across queries. This
repo normalizes embeddings at index time and at query time.

Embeddings are not magic. They capture broad topical similarity well, but
they can miss exact details: a product code, a specific number, or a rare
term may not move the vector much. That is one reason retrieval systems
often pair vector search with keyword search — the two methods fail in
different ways, and together they miss less.
