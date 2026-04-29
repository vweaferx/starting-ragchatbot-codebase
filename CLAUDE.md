# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (from repo root)
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Environment setup:** copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

This is a full-stack RAG chatbot. The backend is a FastAPI server in `backend/`; the frontend is vanilla HTML/CSS/JS in `frontend/` and is served as static files by FastAPI. The `docs/` folder holds the 4 course transcript `.txt` files that form the knowledge base.

### Request flow

1. `POST /api/query` in `app.py` receives `{query, session_id}` and delegates to `RAGSystem.query()`.
2. `RAGSystem` (`rag_system.py`) pulls conversation history from `SessionManager`, then calls `AIGenerator.generate_response()` with the query, history, and a `search_course_content` tool definition.
3. `AIGenerator` (`ai_generator.py`) makes a first Claude API call. If Claude responds with `stop_reason="tool_use"`, `_handle_tool_execution()` runs the tool and makes a second call (without tools) to synthesize the final answer.
4. Tool execution goes through `ToolManager` → `CourseSearchTool.execute()` → `VectorStore.search()` → ChromaDB.
5. Sources (course + lesson tags) are tracked as a side-effect on `CourseSearchTool.last_sources` and retrieved via `ToolManager.get_last_sources()` after the call.
6. `RAGSystem` updates session history and returns `(answer, sources)`.

### Data layer

`VectorStore` (`vector_store.py`) maintains two ChromaDB collections:
- **`course_catalog`** — one document per course, used for fuzzy course-name resolution via semantic search.
- **`course_content`** — sentence-chunked lesson text, filtered by `course_title` and/or `lesson_number` metadata.

On startup, `app.py` ingests all `.txt` files from `../docs/` into ChromaDB, skipping courses whose title already exists in the catalog.

### Document format

Course `.txt` files must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content…>

Lesson 1: <title>
…
```

`DocumentProcessor.process_course_document()` parses this format strictly. The course title doubles as the unique ID in ChromaDB.

### Session state

`SessionManager` holds sessions in-memory (lost on restart). It keeps a rolling window of the last `MAX_HISTORY` (default: 2) exchanges. History is injected into Claude's **system prompt**, not the messages array.

### Key config (`backend/config.py`)

| Setting | Default |
|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | 800 chars |
| `CHUNK_OVERLAP` | 100 chars |
| `MAX_RESULTS` | 5 |
| `MAX_HISTORY` | 2 exchanges |
| `CHROMA_PATH` | `./chroma_db` (relative to `backend/`) |
