# Multi-Tool AI Agent — Web Search + Live Weather + PDF RAG

A single **LangGraph** ReAct agent that reads a user's question and automatically picks the right tool to answer it: live web search, current weather, or question-answering over an uploaded PDF. Built as a mini-project for the Generative AI / Agentic AI coursework.

The agent reasons over the question, calls at most one tool per step, and returns a plain-language answer. Every step is optionally traced in **LangSmith**.

## Features

- **Three tools, one agent** — the LLM decides which to use based on tool docstrings and a system prompt.
  | Tool | Built with | Handles |
  |---|---|---|
  | `web_search` | SerpAPI (Google) | latest news, current events, live facts |
  | `get_current_weather` | OpenWeatherMap API | current temperature, humidity, wind for a city |
  | `search_uploaded_pdf` | FAISS + HuggingFace embeddings | anything written inside the uploaded PDF |
- **PDF RAG pipeline** — load → split (`RecursiveCharacterTextSplitter`, 500/50) → embed (`all-MiniLM-L6-v2`, local, no key) → store in FAISS → retrieve top-3 chunks with a relevance-score cutoff.
- **Bounded memory** — only the latest 10 message objects are kept via `trim_messages`, keeping token cost low without breaking tool-call/tool-result pairs.
- **Safe key handling** — all keys are entered at runtime with `getpass`; nothing is hardcoded.
- **Robust error handling** — clear messages for a missing key, an invalid city (HTTP 401/404/429), network failures, or empty results.
- **LangSmith tracing** — every LLM call and tool call is recorded when a LangSmith key is provided (runs fine without one).

## How it works

```
User question
     |
     v
  Agent (LLM reads question + system prompt)
     |
     +-- needs live internet info? --> web_search          --> SerpAPI
     +-- needs current weather?    --> get_current_weather --> OpenWeather API
     +-- about the PDF?            --> search_uploaded_pdf --> FAISS vector DB
     |
     v
Tool result returns to the LLM  -->  final natural-language answer
     (whole path traced in LangSmith)
```

The loop (`create_react_agent`) repeats until the LLM answers without requesting a tool.

## Tech stack

- **Agent framework:** LangChain + LangGraph (`create_react_agent`)
- **LLM:** `openai/gpt-4o-mini` via **OpenRouter** (OpenAI-compatible endpoint, `temperature=0`)
- **Embeddings / vector store:** `sentence-transformers/all-MiniLM-L6-v2` + `faiss-cpu` (runs locally)
- **Tools:** SerpAPI (search), OpenWeatherMap (weather), FAISS retriever (PDF)
- **Observability:** LangSmith
- **PDF parsing:** `pypdf`

## Getting started

### 1. Open the notebook

Designed for **Google Colab** (uses `google.colab.files` for the PDF upload), but runs anywhere Jupyter does — without Colab it falls back to a small built-in sample document.

### 2. Install dependencies

The first cell installs everything:

```bash
pip install -U \
    langchain langgraph langchain-openai langchain-community \
    langchain-huggingface sentence-transformers faiss-cpu \
    pypdf google-search-results langsmith requests
```

### 3. Provide API keys

When you run the key cell, you'll be prompted for four keys (`getpass` hides input). Only the first three affect answers; LangSmith is optional.

| Environment variable | Get it from | Used for | Required |
|---|---|---|---|
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | the LLM brain of the agent | Yes |
| `SERPAPI_API_KEY` | https://serpapi.com/manage-api-key | web search tool | For search |
| `OPENWEATHER_API_KEY` | https://home.openweathermap.org/api_keys | weather tool | For weather |
| `LANGSMITH_API_KEY` | https://smith.langchain.com → Settings → API Keys | tracing | Optional |

> Missing a key doesn't crash the notebook — the affected tool returns a clear error message and the rest keeps working.

### 4. Run the cells in order

Install → keys → LLM init → build each tool → upload a PDF (optional) → build the agent → run the test questions. Ask your own questions with `ask_agent("...")`, or use the interactive chat loop in the final cell (`exit`/`quit` to stop, `clear` to reset memory).

## Example questions

```python
ask_agent("What are the latest news headlines about AI in India?")  # -> web_search
ask_agent("What is the current weather in Jammu?")                  # -> get_current_weather
ask_agent("What is the humidity there?")                            # -> uses remembered city
ask_agent("According to the uploaded PDF, what is it about?")       # -> search_uploaded_pdf
ask_agent("In one line, what three tools can you use?")             # -> answered with no tool
```

## Project structure

```
.
├── Multi_Tool_Agent_Search_Weather_PDF_RAG_1.ipynb   # the agent notebook
├── README.md
└── .gitignore
```

## Notes & limitations

- `get_current_weather` returns **current** conditions only — no forecasts or history.
- Web results are truncated to ~2000 characters to keep token usage low.
- The PDF retriever drops weak matches (relevance score ≤ 0.15) and will say so rather than invent an answer.
- Memory is capped at 10 messages, so very long multi-turn context can fall out of the window.

## License

Add a license of your choice (e.g. MIT) if you intend others to reuse this.
