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

Events arrive in pipeline order. A client may ignore any event it does not
know; the only ones it must handle are `answer_delta`, `error` and `complete`.

| event | payload | meaning |
|---|---|---|
| `planning` | — | keep-alive while condensing / planning (~every 3s) |
| `plan` | `reasoning` | the planner's rationale, truncated to 180 chars |
| `searching` | — | keep-alive while the searches run |
| `strategy` | `chunks_retrieved` | search finished, before the context budget |
| `sources` | `items`, `retrieved`, `best_score`, `weak` | the passages the answer is built on |
| `answer_delta` | `content` | answer token(s); append |
| `answer` | `content` | the whole answer again, once (keeps the non-streaming path working) |
| `suggestions` | `items` | follow-up questions, or reformulations after a refusal |
| `complete` | `chunks_used`, `no_context?` | end of stream |
| `error` | `message` | fatal for this turn |

```
data: {"type":"planning"}

data: {"type":"plan","reasoning":"The user is asking about manufacturer duties…"}

data: {"type":"strategy","chunks_retrieved":49}

data: {"type":"sources","items":[{"id":"source:abc_chunk_12","parent_id":"source:abc","title":"Regolamento (UE) 2023/1230 — Regolamento macchine","order":12,"score":0.735,"metadata":{"celex":"32023R1230","eli":"http://data.europa.eu/eli/reg/2023/1230/oj","testo_aggiornato":true}}],"retrieved":49,"best_score":0.735,"weak":false}

data: {"type":"answer_delta","content":"In base al "}

data: {"type":"complete","chunks_used":18}
```

`sources` is emitted **before the first `answer_delta`**, so a client can
resolve the `[source:<id>_chunk_<n>]` citations in the answer text as they
stream. `items` carries no chunk text — fetch a passage on demand from
`/api/chunks/{id}`. `metadata` is the whitelisted subset of the owning
document's frontmatter (`CLIENT_METADATA_FIELDS`, see ADR-008); it is `{}` for
notes and for documents ingested without one.

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
[Condense against history]        only when the session has prior turns
    ↓
[Generate Strategy]
    ↓
Query 1    Query 2    Query 3     up to five
    ↓         ↓          ↓
[Vector Search (parallel)]        chunk-level, min_similarity 0.5
    ↓
[Deduplicate + rank]              relevance band, then source precedence
    ↓
[Context budget]                  head of the ranked list that fits
    ↓
[Answer, streamed]                + weak-retrieval guardrail if thin
    ↓
[Persist state] ‖ [Follow-ups]    in parallel
```

### Ranking and context selection

Chunks are ordered by similarity **band** first and only then by the source's
`priorita_fonte`, so curated precedence breaks ties the retriever could not
separate without ever overriding a clearly better match (`SIMILARITY_BAND`).
Content with no declared precedence sorts after ranked sources inside the same
band.

The ranked list is then cut to what fits `MAX_CONTEXT_CHUNKS` /
`MAX_CONTEXT_CHARS`. Truncation always removes the tail, so the answer cites
the same passages it would have cited from the full list. `strategy` reports
what was retrieved, `complete` what was actually used; the gap between them is
how much the budget is cutting.

### When retrieval is thin or empty

- **Nothing retrieved** → the model is given a strict refusal prompt (no
  general-knowledge fallback, no invented citations), the turn is *not* written
  to the session history, and `suggestions` carries questions the corpus can
  actually answer, derived from the notebook's document list.
- **Best match below `WEAK_RETRIEVAL_SCORE`** → the answer is still produced,
  but an extra system message tells the model to say the coverage is marginal
  and to propose a better-scoped question. `sources.weak` reports it so a
  client can show the caveat independently of the model complying.

The threshold is calibrated against observed `rag_trace` history rather than
guessed — see the comment on the constant before changing it.

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
- Three sequential model calls run before the first answer token (condense,
  plan, then the answer itself). On `rag_trace` history that is a median ~7s to
  first token without conversation history, ~9s with it.
- Up to 50 chunks retrieved (5 searches × 10), of which `MAX_CONTEXT_CHUNKS`
  reach the prompt
- Similarity tracks how *specific* the question is nearly as much as how well
  the corpus covers it: short or generic questions score lower and can fall
  below `min_similarity` entirely, taking the no-context path even when the
  corpus does hold the answer
- A planner failure ends the turn — unlike condensation, follow-ups and
  reformulations, `generate_search_strategy` has no fail-open fallback

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
