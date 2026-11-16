# Qdrant (local, Docker) is the search backend

Qdrant runs dense, sparse (BM25) and Reciprocal Rank Fusion natively in one query, supports multi-vector fields for a later late-interaction variant, and fits in 16 GB alongside local models. Postgres with pgvector and ParadeDB was the main alternative: more familiar in production, but fusion would be hand-written and the hybrid query less inspectable. OpenSearch was rejected as too heavy for the machine.
