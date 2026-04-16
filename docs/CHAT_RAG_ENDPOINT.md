# Chat + RAG Hybrid Endpoint

## Overview

New endpoint `/api/chat/rag/execute` that combines:

1. **Advanced multi-search strategy** from `/api/search/ask`
   - Generates 3-5 optimized search queries from user question
   - Executes parallel vector searches
   - Deduplicates results

2. **Conversational memory** from `/api/chat/sessions`
   - Maintains chat history in LangGraph
   - Supports model overrides
   - Persistent sessions

## Why This Endpoint?

**Problem with `/api/search/ask`:**
- ✅ Advanced multi-search strategy
- ❌ No conversational memory
- ❌ Every question is isolated

**Problem with `/api/chat/execute`:**
- ✅ Conversational memory
- ❌ Requires manual context building
- ❌ No dynamic RAG retrieval

**Solution: `/api/chat/rag/execute`:**
- ✅ Multi-search strategy (dynamic RAG)
- ✅ Conversational memory
- ✅ Automatic chunk retrieval per question
- ✅ Best of both worlds

## API Specification

### Endpoint

```
POST /api/chat/rag/execute
```

### Request Body

```json
{
  "session_id": "chat_session:abc123",
  "message": "What are the main findings?",
  "notebook_id": "notebook:xyz789",         // optional, for scoped search
  "strategy_model": "model:strategy123",    // optional, defaults to answer model
  "model_override": "model:chat456",        // optional, per-request model
  "stream": true                            // optional, default true
}
```

### Response (Streaming SSE)

```
data: {"type":"strategy","chunks_retrieved":15}

data: {"type":"answer","content":"Based on the research..."}

data: {"type":"complete","chunks_used":15}
```

### Response (Non-streaming)

```json
{
  "session_id": "chat_session:abc123",
  "messages": [
    {
      "id": "final",
      "type": "assistant",
      "content": "Based on the research..."
    }
  ],
  "chunks_used": 15
}
```

## Usage Flow

### 1. Create a chat session

```bash
curl -X POST http://localhost:5055/api/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "notebook_id": "notebook:xyz789",
    "title": "Research Chat"
  }'

# Response: {"id": "chat_session:abc123", ...}
```

### 2. Send messages with RAG

```bash
curl -N -X POST http://localhost:5055/api/chat/rag/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat_session:abc123",
    "message": "What are the key findings about AI safety?",
    "stream": true
  }'
```

### 3. Continue conversation

The session remembers previous exchanges:

```bash
curl -N -X POST http://localhost:5055/api/chat/rag/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat_session:abc123",
    "message": "Can you elaborate on the first point?",
    "stream": true
  }'
```

The agent will:
1. Remember what "the first point" refers to
2. Execute new vector searches based on current question
3. Generate answer using conversation context + fresh RAG results

## Integration with AI Chat Widget

Update the widget API route to use this endpoint:

```typescript
// app/api/chat/route.ts

export async function POST(req: Request) {
  const { messages } = await req.json()
  const lastMessage = messages[messages.length - 1]
  
  // Get or create session
  let sessionId = getSessionFromCookie() // implement cookie storage
  if (!sessionId) {
    const createResponse = await fetch(`${OPEN_NOTEBOOK}/api/chat/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        notebook_id: process.env.NOTEBOOK_ID,
        title: 'Widget Chat'
      })
    })
    const session = await createResponse.json()
    sessionId = session.id
    setSessionCookie(sessionId) // implement cookie storage
  }
  
  // Call hybrid endpoint
  const response = await fetch(`${OPEN_NOTEBOOK}/api/chat/rag/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: lastMessage.content,
      stream: true
    })
  })
  
  // Stream response
  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache'
    }
  })
}
```

## Benefits

### For Users
- Natural conversation flow
- Agent remembers context
- More accurate answers (dynamic RAG per question)

### For Developers
- Single endpoint for chat + RAG
- Automatic multi-search optimization
- No manual chunk management
- Built-in deduplication

### Performance
- Parallel vector searches (faster than sequential)
- Deduplication reduces token usage
- Only relevant chunks per question (not entire notebook)

## Architecture

```
User Question
    ↓
[Generate Strategy]
    ↓
Query 1    Query 2    Query 3
    ↓         ↓          ↓
[Vector Search (parallel)]
    ↓
[Deduplicate Chunks]
    ↓
[Chat Graph with Context]
    ↓
Response (with memory)
```

## Configuration

### Required Models

1. **Embedding model** - for vector search
2. **Strategy model** - for query generation (can be same as answer)
3. **Chat model** - for response generation

Configure via `/api/models` endpoint.

### Environment Variables

No new variables required. Uses existing Open-Notebook configuration.

## Limitations

- Requires embedding model configured
- Session management required (cookie/localStorage in widget)
- Strategy generation adds ~1-2s latency per request
- Maximum 50 chunks retrieved (configurable in code)

## Testing

Test with curl:

```bash
# Create session
SESSION=$(curl -X POST http://localhost:5055/api/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"notebook_id":"notebook:test","title":"Test"}' | jq -r '.id')

# Send message
curl -N -X POST http://localhost:5055/api/chat/rag/execute \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Tell me about machine learning\",
    \"stream\": true
  }"
```

## Next Steps

1. **Restart Open-Notebook** to load new endpoint
2. **Test** with curl or Swagger UI (http://localhost:5055/docs)
3. **Update widget** to use `/api/chat/rag/execute`
4. **Add session management** (cookies or localStorage)

## File Locations

- Router: `/api/routers/chat_rag.py`
- Main app: `/api/main.py` (router registered)
- Documentation: `/docs/CHAT_RAG_ENDPOINT.md`
