# 🚀 FastAPI Crash Course

> Learned through our `client/server.py` — so every example is from your own code.

---

## 1. What is FastAPI?

A Python framework that turns type-annotated functions into REST APIs. No XML config, no separate routing files, no manual JSON parsing.

```
Your Python function  +  type hints  →  production REST API
```

---

## 2. The app object

```python
from fastapi import FastAPI

app = FastAPI(title="PoetryDuel-GPT API")
```

This single object is your entire server. It holds routes, middleware, startup hooks, and configuration. Pass it to uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
#            ↑     ↑
#       file:object
```

---

## 3. Routes = HTTP method + URL path → function

```python
@app.get("/")                         # GET  /
def root():
    return {"name": "PoetryDuel", "model": model_info}

@app.post("/chat")                    # POST /chat
def chat(req: ChatRequest):
    ...
    return ChatResponse(response="...", prompt="...")
```

| Decorator | HTTP method | Example |
|-----------|-------------|---------|
| `@app.get("/")` | GET | Read data, status checks |
| `@app.post("/chat")` | POST | Send data, trigger action |
| `@app.put("/item/{id}")` | PUT | Update existing resource |
| `@app.delete("/item/{id}")` | DELETE | Remove a resource |

The function name doesn't matter to FastAPI — only the decorator path matters.

---

## 4. Type hints = automatic request validation

### Without FastAPI (what you'd write manually):

```python
# ❌ Manual validation — tedious and error-prone
def handle_post(request):
    body = json.loads(request.body)
    prompt = body.get("prompt")
    if not prompt:
        return 400, {"error": "prompt is required"}
    if not isinstance(prompt, str):
        return 400, {"error": "prompt must be a string"}
    temp = body.get("temperature", 0.75)
    try:
        temp = float(temp)
    except ValueError:
        return 400, {"error": "temperature must be a number"}
    if temp <= 0:
        return 400, {"error": "temperature must be positive"}
    # ... repeat for top_k, top_p, max_tokens
```

### With FastAPI (your server.py):

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str                          # required, must be string
    temperature: float = 0.75            # optional, default 0.75
    top_k: int = 50                      # optional, default 50
    top_p: float | None = 0.92           # optional, can be None
    max_tokens: int = 64                 # optional, default 64

@app.post("/chat")
def chat(req: ChatRequest):              # ← FastAPI auto-validates
    print(req.prompt)                     # it's already a string, guaranteed
    print(req.temperature)                # it's already a float, guaranteed
```

**What happens on bad input:**
```json
// Request:  {"prompt": 123}
// Response: 422 Unprocessable Entity
// {
//   "detail": [{
//     "loc": ["body", "prompt"],
//     "msg": "Input should be a valid string",
//     "type": "string_type"
//   }]
// }
```

No code written — FastAPI generates this from the type hint alone.

---

## 5. Response models = automatic output filtering

```python
class ChatResponse(BaseModel):
    response: str
    prompt: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return ChatResponse(response="viết...", prompt=req.prompt)
```

FastAPI auto-converts to JSON. If you accidentally return extra fields, they get stripped (security). If you forget a required field, it errors before sending the response (safety).

```python
# This would be caught before reaching the client:
return ChatResponse(response="...")     # ❌ missing "prompt" field
```

---

## 6. Path parameters = URL variables

```python
# Not in your project, but common:
@app.get("/checkpoint/{step}")
def get_checkpoint(step: int):
    return {"path": f"checkpoints/step_{step}.pt"}

# GET /checkpoint/5000 → {"path": "checkpoints/step_5000.pt"}
# GET /checkpoint/abc  → 422 (not an int)
```

---

## 7. Query parameters = URL ?key=value

```python
# Not in your project, but useful:
@app.get("/search")
def search(q: str = "", limit: int = 10):
    return {"results": f"searching for '{q}' with limit {limit}"}

# GET /search?q=lục bát&limit=20
# GET /search                    (uses defaults)
```

---

## 8. CORS middleware = let the browser talk to the API

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # any website can call this API
    allow_methods=["*"],       # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # any request headers
)
```

**Why you need this:** Your React frontend runs on `localhost:3000`. Your API runs on `localhost:8000`. Browsers block cross-origin requests by default (security). CORS middleware tells the browser "it's okay, allow this."

Without it: your React app's `fetch('/chat')` silently fails.

---

## 9. Startup events = initialize once

```python
@app.on_event("startup")
def startup():
    """Runs ONCE before any requests are accepted."""
    print("🚀 Loading model...")
    load()                                 # load model + tokenizer into globals
    print("✅ Ready")

@app.on_event("shutdown")
def shutdown():
    """Runs ONCE when server stops."""
    print("👋 Cleaning up...")
```

The server won't serve requests until `startup` completes. Use it for:
- Loading ML models
- Connecting to databases
- Warming up caches

---

## 10. Error handling

```python
from fastapi import HTTPException

@app.post("/chat")
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")
```

`HTTPException` immediately stops the function and returns an error response. Common status codes:

| Code | Meaning | When |
|------|---------|------|
| 400 | Bad Request | Client sent bad data |
| 422 | Unprocessable | Validation failed (automatic) |
| 500 | Internal Error | Something crashed |
| 503 | Service Unavailable | Model not loaded yet |

---

## 11. Your full request lifecycle

```
Browser (localhost:3000)              FastAPI (localhost:8000)
    │                                      │
    │  POST /chat                          │
    │  Content-Type: application/json      │
    │  {"prompt": "thơ...",                │
    │   "temperature": 0.75}               │
    ├─────────────────────────────────────>│
    │                                      │  1. Parse JSON body
    │                                      │  2. Validate → ChatRequest
    │                                      │     prompt: str ✓
    │                                      │     temperature: 0.75 ✓
    │                                      │  3. Call chat(req)
    │                                      │  4. Run generate()
    │                                      │     (autoregressive loop,
    │                                      │      ~30 forward passes)
    │                                      │  5. Decode new tokens
    │                                      │  6. Build ChatResponse
    │                                      │  7. Validate response ✓
    │                                      │  8. Convert to JSON
    │                                      │
    │  {"response": "cho nên...",          │
    │   "prompt": "thơ..."}                │
    │<─────────────────────────────────────┤
```

---

## 12. Async vs Sync

```python
@app.get("/sync")
def sync_route():                     # blocks the worker thread
    time.sleep(5)                     # other requests WAIT
    return {"done": True}

@app.get("/async")
async def async_route():              # doesn't block
    await asyncio.sleep(5)            # other requests proceed
    return {"done": True}
```

| Use case | sync `def` | async `async def` |
|----------|-----------|-------------------|
| CPU-bound (ML inference) | ✅ Fine | ❌ No benefit |
| I/O (DB, HTTP, files) | ❌ Blocks | ✅ Efficient |
| Fast return (<10ms) | ✅ Either | ✅ Either |

Your model inference is CPU-bound — `def` is correct. FastAPI runs it in a thread pool automatically.

---

## 13. Dependency injection (for scaling beyond globals)

Your current code uses module-level globals:

```python
model = None          # ← global, set in load()
tokenizer = None      # ← global, set in load()

def generate(...):
    end_id = tokenizer.token_to_id(...)  # ← uses global
```

This works for a single server but doesn't scale. FastAPI's solution:

```python
from fastapi import Depends

def get_model():
    """Could load from pool, Redis, or just return the global."""
    return model

def get_tokenizer():
    return tokenizer

@app.post("/chat")
def chat(
    req: ChatRequest,
    mdl = Depends(get_model),          # ← FastAPI injects this
    tok = Depends(get_tokenizer),      # ← And this
):
    end_id = tok.token_to_id("<|end|>")
    logits, _ = mdl(idx)
```

**Why this matters later:**
- Swap `get_model()` to load a *different* model for A/B testing
- Return from a connection pool for concurrent requests
- Mock for unit tests by overriding the dependency

No global variables. No import spaghetti. Just "give me what I need."

---

## 14. Auto-generated docs

Start your server and open:

```
http://localhost:8000/docs           ← Swagger UI (interactive)
http://localhost:8000/redoc          ← ReDoc (cleaner reading)
```

Every route appears with:
- HTTP method and path
- Request body schema (from `ChatRequest`)
- Response schema (from `ChatResponse`)
- **"Try it out" button** — sends real requests from the browser
- All HTTP status codes with descriptions

Zero documentation code written — all generated from type hints.

---

## 15. Common patterns comparison

| What | Flask | FastAPI | Django REST |
|------|-------|---------|-------------|
| Define route | `@app.route("/")` | `@app.get("/")` | `urls.py` + ViewSet |
| Validate input | Manual or Marshmallow | Type hints (Pydantic) | Serializer |
| API docs | Manual or Flask-RESTX | Auto `/docs` | drf-spectacular |
| Async support | Limited | Native (async/await) | Django 4.2+ |
| Performance | Good | Very good (Starlette) | Good |
| Learning curve | Low | Low | High |

---

## 16. Minimal working API (your server distilled)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EchoRequest(BaseModel):
    message: str

class EchoResponse(BaseModel):
    echo: str

@app.post("/echo", response_model=EchoResponse)
def echo(req: EchoRequest):
    return EchoResponse(echo=f"You said: {req.message}")

# That's it. Run: uvicorn main:app --port 8000
# Test:  curl -X POST http://localhost:8000/echo -H "Content-Type: application/json" -d '{"message": "hello"}'
# Docs:  http://localhost:8000/docs
```

40 lines. Automatic validation, error handling, JSON parsing, and interactive docs.

---

## 17. Key takeaways

1. **Type hints drive everything.** You write them once, FastAPI uses them for validation, serialization, and documentation.

2. **No magic strings.** `req.prompt` is typed, auto-completed by your IDE, and refactor-safe.

3. **Errors are informative.** Bad input → 422 with exact field + reason. Never a generic "something went wrong."

4. **Docs are free.** Every route, every model, every status code — generated from code, always in sync.

5. **It's still Python.** No decorators that change control flow. No metaclasses. Just functions returning dicts or Pydantic models.
