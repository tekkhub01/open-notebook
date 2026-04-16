# Quick Start - Docker Setup

## Prerequisites

- Docker and Docker Compose installed
- An AI provider API key (e.g., OpenRouter, OpenAI, Anthropic)

## 1. Configure Environment

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `OPEN_NOTEBOOK_ENCRYPTION_KEY` — any secret string (min 16 chars)
- Your AI provider key (e.g., `OPENROUTER_API_KEY=sk-or-...`)

## 2. Run (Official Image)

```bash
docker compose up -d
```

This pulls the pre-built image and starts SurrealDB + Open Notebook.

- **Web UI**: http://localhost:8502
- **API**: http://localhost:5055
- **API Docs**: http://localhost:5055/docs

Check logs: `docker compose logs -f`

## 3. Run (Local Development)

To build from your local source code (after making changes):

```bash
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml build --no-cache open_notebook
docker compose -f docker-compose.dev.yml up -d
```

Same ports as above. Use this when developing features or testing modifications.

**After rebuild, verify:**
```bash
# Check API is running
curl http://localhost:5055/health

# View startup logs
docker compose -f docker-compose.dev.yml logs -f open_notebook
```

## Useful Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop everything
docker compose down

# Rebuild after code changes (dev mode)
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml build --no-cache open_notebook
docker compose -f docker-compose.dev.yml up -d

# Full reset (removes database data)
docker compose down -v && rm -rf surreal_data notebook_data
```

## Troubleshooting

### "OpenAI API key not found" errors

If you see this error but have configured an embedding model:

1. Check credentials in UI: Settings > API Keys
2. Test connection: Settings > API Keys > [Your Credential] > Test Connection
3. **Restart container** to reload environment variables:
   ```bash
   docker compose restart open_notebook
   ```

The hybrid chat endpoint (`/api/chat/rag/execute`) automatically falls back to text search if vector search fails, so it will still work.

### Port conflicts

If ports 5055 or 8502 are already in use:

```bash
# Edit docker-compose.yml ports section
ports:
  - "8503:8502"  # Change 8502 to 8503
  - "5056:5055"  # Change 5055 to 5056
```

### Database connection issues

```bash
# Check SurrealDB is running
docker compose ps

# View SurrealDB logs
docker compose logs surrealdb

# Restart database
docker compose restart surrealdb
```

## LAN Access

To access from other devices on your network, use your machine's IP (e.g., `http://192.168.1.200:8502`). The Docker container already binds to all interfaces.

## Widget Integration

If you're integrating an AI chat widget (like the Next.js widget in `/DEV/ai-widget`), configure it to use the hybrid endpoint:

**Widget `.env.local`:**
```env
OPEN_NOTEBOOK_ENDPOINT=http://192.168.1.200:5055
OPEN_NOTEBOOK_NOTEBOOK_ID=notebook:YOUR_NOTEBOOK_ID
OPEN_NOTEBOOK_STRATEGY_MODEL=model:YOUR_MODEL_ID
OPEN_NOTEBOOK_CHAT_MODEL=model:YOUR_MODEL_ID
```

**Get your notebook and model IDs:**
```bash
# List notebooks
curl http://localhost:5055/api/notebooks | jq '.[] | {id, name}'

# List models
curl http://localhost:5055/api/models | jq '.[] | select(.type=="language") | {id, name}'
```

The widget will automatically:
- Create chat sessions with cookies
- Retrieve relevant document chunks per question  
- Maintain conversational memory
- Work with text search (no vector embeddings required)

## New Features

### Hybrid Chat + RAG Endpoint

The `/api/chat/rag/execute` endpoint combines:
- **Multi-search strategy** — generates 3-5 optimized queries per question
- **Conversational memory** — maintains chat history across requests
- **Dynamic RAG** — retrieves relevant document chunks per question
- **Text search fallback** — works even without vector embeddings

**Test it:**

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:5055/api/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"notebook_id":"NOTEBOOK_ID","title":"Test Chat"}' | jq -r '.id')

# Ask questions with memory
curl -X POST http://localhost:5055/api/chat/rag/execute \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"What is this notebook about?\",
    \"stream\": false
  }"

# Follow-up question (remembers context)
curl -X POST http://localhost:5055/api/chat/rag/execute \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Tell me more about that\",
    \"stream\": false
  }"
```

**Why use it vs `/api/search/ask`?**
- ✅ Conversational memory (follow-up questions work naturally)
- ✅ Works with text search (no embedding model required)
- ✅ Better for chat interfaces and assistants
- ❌ `/api/search/ask` is simpler for one-shot queries

See [CHAT_RAG_ENDPOINT.md](docs/CHAT_RAG_ENDPOINT.md) for full documentation.

## Notes

- The database (`surreal_data/`) and app data (`notebook_data/`) persist in local directories
- API keys can also be configured through the UI at Settings > API Keys
- First startup may take a moment while the database migrations run
- For chat widgets, use `/api/chat/rag/execute` for conversational memory
