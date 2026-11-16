# Side Ablations

Use the fixed Dev Split for all five runs. The reference is Hybrid retrieval with
200-word sentence Chunks and one sentence of overlap, Qwen3-Embedding 0.6B
with its bake-off query prefix, BGE-Reranker-v2-m3 over 30 candidates, and the
provisional Qwen3.5 4B generator. Keep these model settings fixed across the runs.

## Chunk size

Compare 100, 200 and 400 words per Chunk with one sentence of overlap in three
separate collections. At a fixed retrieval depth, larger Chunks can contain more
Evidence facts by construction and give the generator more text. Read Evidence
Recall@10 alongside answer accuracy and mean generator context tokens. The token
metric uses the final generation stage's input, which includes prompt framing.

<!-- table: rag table CHUNK_100_RUN CHUNK_200_RUN CHUNK_400_RUN -->
<!-- conclusion: Explain the direction and uncertainty of the measured changes, including any negative result. -->

## Semantic chunking

Compare the reference's sentence packing with a semantic index. Both variants are
configured for 200 words and one sentence of overlap. The semantic index starts a
Chunk at adjacent sentence cosine distances strictly above the Article's 90th
percentile. It carries the previous Chunk's last sentence across topic boundaries
and word-limit cuts when it fits. Both strategies keep sentences whole, including
individual sentences longer than the word budget. The semantic policy can change
the number of Chunks, the amount of Evidence per Chunk and the text available to
the generator. Its sentence embedding pass also adds build work.

<!-- table: rag table CHUNK_200_RUN SEMANTIC_RUN -->
<!-- conclusion: Explain the measured trade-off between retrieval, answering and build work. -->

## HyDE

Compare the reference with a variant that uses the same index and generates a
hypothetical passage as input to the dense arm. The sparse arm keeps the original
question in both runs. Semantic matches can improve or drift, and a model call
adds tokens and latency. Use the `hyde` stage to account for that call when
interpreting any change in retrieval or accuracy.

<!-- table: rag table CHUNK_200_RUN HYDE_RUN -->
<!-- conclusion: Explain whether the measured changes justify the extra generation call. -->
