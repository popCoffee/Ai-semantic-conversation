

# **OpenSearch & Redis powered by OCI Generative AI**
This chatbot application delivers fast, accurate, and context‑aware responses by combining:

- OpenSearch – Indexes and retrieves structured and semantic search data.
- Redis (Cache) – Provides ultra‑fast, in‑memory access to frequently used results.
- Flask Backend (Python) – Orchestrates queries, integrates with OpenSearch & Redis, and exposes APIs.
- Frontend (JavaScript, HTML, CSS) – A lightweight interface for seamless user interaction.
The chatbot intelligently ranks and retrieves the best matches from OpenSearch, while Redis ensures rapid response times. It is designed for customer support, knowledge bases

**How It Works**

- The user submits his query in the chatbot UI.

- The Flask backend processes the request and:
First checks Redis for cached responses.
If not found, queries OpenSearch for the best-matching results.
Returns the response to the user while optionally caching it in Redis for future queries.

- The chatbot continuously learns from user queries and stored knowledge, improving search accuracy.


