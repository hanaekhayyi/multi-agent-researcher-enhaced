## 📰 Multi-Agent AI Researcher

A Streamlit app where a small [Agno](https://docs.agno.com) team researches a topic on Hacker News and the web, then writes a report, a blog post or a social media post, with sources.

It runs on **free models**: [Groq](https://console.groq.com) (free API key, no credit card) or a local [Ollama](https://ollama.com) model.

### Features
- Searches Hacker News **by topic** and reads what the community says in the top comments
- Reads the linked articles and adds recent context from the web
- Output format of your choice: report, blog post, or LinkedIn / X post
- Live view of which agent is working on what, plus token usage and duration
- Follow-up questions in the same conversation (history kept in RAM)

### How to get started

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/advanced_ai_agents/multi_agent_apps/multi_agent_researcher
pip install -r requirements.txt
cp .env.example .env        # then paste your free Groq key in .env (or in the sidebar)
streamlit run research_agent.py
```

To use Ollama instead: install it, run `ollama pull llama3.2`, and pick **Ollama** in the sidebar.

### How it works

```
             ┌──────── HackerNews Team (leader, gpt-oss-120b) ────────┐
 query ────▶ │ 1. HackerNews Researcher ─ search_hackernews_stories│
             │                            get_top_comments (@tool) │
             │ 2. Article Reader ──────── Newspaper4kTools          │ ───▶ markdown + sources
             │ 3. Web Searcher ────────── WebSearchTools            │
             └──────────────── InMemoryDb (session history) ───────┘
```

| Agno feature | Where |
|---|---|
| `Team` (coordinate mode) | the leader plans, delegates to members and writes the final answer |
| `Agent` with `role`, `instructions`, `tool_call_limit` | three specialised members on a lighter, faster model |
| Custom `@tool` | `hn_tools.py`: topic search (HN Algolia API, optional `recent_only` filter) and top comments, returning compact JSON |
| Built-in toolkits | `Newspaper4kTools(article_length=...)`, `WebSearchTools(backend="auto")`: falls back to another engine when DuckDuckGo blocks automated queries |
| Model factory | `build_model()` switches between Groq and Ollama; Groq retries 429s with exponential backoff (5s → 40s) and gpt-oss runs with `reasoning_effort="low"` via `request_params` |
| `InMemoryDb` + `add_history_to_context` | follow-up questions, no database needed |
| Streaming events | `team.run(stream=True, stream_events=True)` drives the live status panel |
| Source check | URLs from the members' tool calls (streamed events) are collected; any other URL in the answer is flagged as possibly invented |

**Staying inside the free Groq quota.** Articles are truncated, comments are short, members must answer in under 200 words, each agent has a `tool_call_limit`, and members use the smaller gpt-oss-20b. A typical run uses about 5,000 tokens and takes 10 to 30 seconds. If the quota is still reached, the app shows a clear message: wait a minute and retry.

**In-memory storage.** The conversation history lives in the Streamlit process memory. It is lost when the app restarts, which is intentional to keep the project database-free.

**Note on "top comments".** The Hacker News API does not expose comment scores. `get_top_comments` returns the first top-level comments in the order HN ranks them, which is the closest available signal.

### Tests

The tests run offline: no LLM call, and HTTP calls are mocked.

```bash
pip install -r requirements-dev.txt
pytest
```
