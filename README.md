# GenAI Chatbot - OpenSearch & Redis with Generative AI OCI 

This is a chatbot application that intelligently retrieves the best responses using:

- OpenSearch – Indexes and retrieves structured search data.

- Redis (Cache) – Provides fast, in-memory data retrieval.

- Flask Backend (Python) – Handles queries and integrates with OpenSearch & Redis.

- Frontend (JavaScript, HTML, CSS) – User interface for interacting with the chatbot.

The chatbot intelligently finds and ranks the best match from OpenSearch and Redis, ensuring fast and accurate responses. It is designed for customer support, knowledge bases, and AI-driven assistance.

**Features** 

🔍 Smart Search – Queries OpenSearch for indexed content and ranks results.

⚡ Cache Optimization – Uses Redis to store frequent responses for faster access.

🌍 Real-time Response – Integrates OpenSearch & Redis for instant data retrieval.

🔄 API-Based Architecture – Exposes endpoints for external integration.

🎨 Minimal UI – Simple JavaScript-based frontend for seamless interaction.

**How It Works**

- A user submits a query in the chatbot UI.

- The Flask backend processes the request and:
First checks Redis for cached responses.
If not found, queries OpenSearch for the best-matching results.
Returns the response to the user while optionally caching it in Redis for future queries.

- The chatbot continuously learns from user queries and stored knowledge, improving search accuracy.


