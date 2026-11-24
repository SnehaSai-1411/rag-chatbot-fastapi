# Chunking Strategies

Chunking is the step that turns long documents into the small pieces a
retrieval system actually searches over. A chunk is usually a few hundred
characters: long enough to hold a complete idea, short enough that its
embedding represents one topic rather than a mixture of several.

The simplest approach is fixed-size chunking, which cuts text every N
characters regardless of content. It is easy to implement but can split a
sentence or an argument in half, which hurts both retrieval and the quality
of the answer built from the chunk.

A better default is paragraph-aware chunking. Paragraphs are kept whole when
they fit inside the size limit, and several short paragraphs are joined into
one chunk. Only a paragraph longer than the limit is hard-split, and an
overlap is kept across the cut so the context at the boundary is not lost.
This repo's chunker uses this strategy, with a default size of 800
characters and an overlap of 100.

Chunk size is a trade-off, not a fixed constant. Small chunks retrieve more
precisely — the embedding matches a narrow topic — but they can lack the
surrounding context the language model needs to answer well. Large chunks
carry more context but dilute the match, and they consume more of the
model's prompt budget. The right size depends on the documents and the
questions, and it is worth measuring with an evaluation set rather than
guessing.
