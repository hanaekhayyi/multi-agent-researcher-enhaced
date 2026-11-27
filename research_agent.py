"""Multi-Agent AI Researcher: a small Agno team that researches Hacker News topics.

Run with:  streamlit run research_agent.py
"""

import os
import re
import time
import uuid

import streamlit as st
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.run.agent import RunEvent
from agno.run.team import TeamRunEvent
from agno.team import Team
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.websearch import WebSearchTools
from dotenv import load_dotenv

from hn_tools import get_top_comments, search_hackernews_stories

load_dotenv()  # reads GROQ_API_KEY and model names from a local .env, if present

# --- Configuration -----------------------------------------------------------

# Free Groq models: a strong one for the leader, a fast one for members that only call tools.
DEFAULT_LEADER_MODEL = os.getenv("LEADER_MODEL", "openai/gpt-oss-120b")
DEFAULT_MEMBER_MODEL = os.getenv("MEMBER_MODEL", "openai/gpt-oss-20b")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Limits that keep a run inside the free Groq quota (8,000 tokens per minute per model).
MAX_ARTICLE_CHARS = 2000
MAX_WEB_RESULTS = 3
SOURCES_RULE = "Only cite URLs copied exactly from your tool results. Never write a URL from memory."

# Member events are streamed through the team; their tool results are the only trusted sources.
MEMBER_TOOL_CALL_STARTED = RunEvent.tool_call_started
MEMBER_TOOL_CALL_COMPLETED = RunEvent.tool_call_completed

OUTPUT_FORMATS = {
    "📄 Rapport": "a structured research report",
    "✍️ Blog": "an engaging blog post",
    "📣 Post social": "a short social media post (under 250 words) with a few hashtags",
}

# Display name and icon of each member, keyed by the member_id Agno derives from its name.
MEMBERS_UI = {
    "hackernews-researcher": ("🔎", "HackerNews Researcher", "Cherche les stories HN et lit les meilleurs commentaires"),
    "article-reader": ("📖", "Article Reader", "Lit les articles liés et en extrait l'essentiel"),
    "web-searcher": ("🌐", "Web Searcher", "Ajoute du contexte récent depuis le web"),
}

EXAMPLE_TOPICS = ["Rust en production", "Agents IA autonomes", "Bases de données vectorielles"]

# --- Agno: models and team ---------------------------------------------------


def build_model(provider: str, model_id: str, api_key: str | None = None) -> Model:
    """Return the Agno model for the chosen provider."""
    if provider == "groq":
        if not api_key:
            raise ValueError("A Groq API key is required.")
        # gpt-oss models reason before answering: "low" effort is faster and uses fewer tokens.
        request_params = {"reasoning_effort": "low"} if "gpt-oss" in model_id else None
        # Groq asks to wait ~20s after a 429: back off 5s, 10s, 20s, 40s.
        return Groq(
            id=model_id,
            api_key=api_key,
            request_params=request_params,
            retries=4,
            delay_between_retries=5,
            exponential_backoff=True,
        )
    if provider == "ollama":
        return Ollama(id=model_id, options={"num_predict": 1024})
    raise ValueError(f"Unknown provider: {provider!r}")


def build_team(leader_model: Model, member_model: Model, db: InMemoryDb | None = None) -> Team:
    """Assemble the research team. Pure function: no LLM call is made here."""
    hn_researcher = Agent(
        name="HackerNews Researcher",
        role="Finds Hacker News stories about a topic and what the community says about them.",
        model=member_model,
        tools=[search_hackernews_stories, get_top_comments],
        instructions=[
            "Call search_hackernews_stories exactly once. Use recent_only=True if recent stories are asked for.",
            "Then fetch the top comments of the 2 most relevant stories.",
            "Return each story's title, url, hn_url, points and a one-line summary of the comments.",
            "Be concise: under 200 words in total.",
            SOURCES_RULE,
        ],
        tool_call_limit=3,
    )

    article_reader = Agent(
        name="Article Reader",
        role="Reads articles from URLs and extracts their key points.",
        model=member_model,
        tools=[Newspaper4kTools(article_length=MAX_ARTICLE_CHARS)],
        instructions=[
            "Read each URL you are given (2 at most).",
            "Return 3 key points per article with its exact URL, under 150 words in total.",
            "If a page cannot be read, say so. Never invent content for it.",
            SOURCES_RULE,
        ],
        tool_call_limit=2,
    )

    web_searcher = Agent(
        name="Web Searcher",
        role="Searches the web for recent context on a topic.",
        model=member_model,
        # backend="auto" falls back to another engine when DuckDuckGo blocks automated queries.
        tools=[WebSearchTools(backend="auto", fixed_max_results=MAX_WEB_RESULTS, enable_news=False)],
        instructions=[
            "Run one short web search (a few keywords) on the topic.",
            "Return the 3 most useful facts with their source URL, under 150 words.",
            SOURCES_RULE,
        ],
        add_datetime_to_context=True,
        tool_call_limit=2,
    )

    return Team(
        name="HackerNews Team",
        model=leader_model,
        members=[hn_researcher, article_reader, web_searcher],
        description="You lead a research team that turns Hacker News discussions into well-sourced content.",
        instructions=[
            "1. Ask the HackerNews Researcher for stories about the user's topic, once.",
            "2. Send the URLs of the 2 best stories to the Article Reader. Always give it the links.",
            "3. Ask the Web Searcher for extra context on the topic.",
            "4. Write the final answer in the format the user asked for, using only facts your members found.",
            "Never delegate the same task twice: work with what the members returned.",
            "In '## Sources', list only URLs that your members returned. Never add a URL they did not give you.",
            "For a follow-up question, reuse the previous research and only delegate if new information is needed.",
            "Answer in the language of the user's question.",
        ],
        expected_output=(
            "Markdown with a title, the content in the requested format, "
            "and a final '## Sources' list of the Hacker News and article links used."
        ),
        db=db,
        add_history_to_context=db is not None,
        num_history_runs=3,
        markdown=True,
    )


@st.cache_resource
def get_team(provider: str, leader_id: str, member_id: str, api_key: str | None) -> Team:
    """Build the team once per configuration instead of on every Streamlit rerun."""
    return build_team(
        leader_model=build_model(provider, leader_id, api_key),
        member_model=build_model(provider, member_id, api_key),
        db=InMemoryDb(),  # sessions live in RAM and are lost when the app restarts
    )


# --- Helpers -----------------------------------------------------------------


def friendly_error(message: str) -> str:
    """Translate a raw provider error into a message the user can act on."""
    text = message.lower()
    if "invalid_api_key" in text or "401" in text or "authentication" in text:
        return "Clé API invalide. Vérifie ta clé Groq."
    if "model_not_found" in text or "decommissioned" in text:
        return "Ce modèle n'est plus disponible sur Groq. Choisis-en un autre (LEADER_MODEL / MEMBER_MODEL dans .env)."
    if "rate_limit" in text or "429" in text or "too many requests" in text:
        return "Quota gratuit de Groq atteint pour le moment. Réessaie dans une minute."
    if "connect" in text or "connection" in text:
        return "Impossible de joindre le modèle. Si tu utilises Ollama, vérifie qu'il est lancé (ollama serve)."
    return f"Le modèle a renvoyé une erreur : {message}"


URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]\"'`]+")


def extract_urls(text: str) -> set[str]:
    """Find the URLs in a text, without trailing punctuation or slash."""
    return {url.rstrip(".,;:!?*/") for url in URL_PATTERN.findall(text or "")}


def unverified_urls(answer: str, known_urls: set[str]) -> list[str]:
    """URLs cited in the answer that no tool ever returned: likely invented by the model."""
    return sorted(extract_urls(answer) - known_urls)


def member_ui(member_id: str) -> tuple[str, str]:
    """Icon and display name of a member, with a fallback for unknown ids."""
    icon, name, _ = MEMBERS_UI.get(member_id, ("🤖", member_id, ""))
    return icon, name


# --- Streamlit UI ------------------------------------------------------------

CSS = """
<style>
.block-container {padding-top: 2.5rem; max-width: 820px;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0.2rem;
  background: linear-gradient(90deg, #ff6600, #ffb347); -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {opacity: 0.75; margin-top: 0;}
.agent-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;}
.agent-card b {display: block; font-size: 0.9rem;}
.agent-card span {font-size: 0.78rem; opacity: 0.7;}
.run-meta {font-size: 0.8rem; opacity: 0.7; margin-top: 0.4rem;}
</style>
"""


def render_sidebar() -> tuple[str, str | None, str, str, str]:
    """Settings panel. Returns provider, api_key, leader_id, member_id, output_format."""
    with st.sidebar:
        st.markdown("### ⚙️ Réglages")
        provider = st.segmented_control(
            "Modèle", ["groq", "ollama"], default="groq", width="stretch",
            format_func=lambda p: {"groq": "⚡ Groq (gratuit)", "ollama": "💻 Ollama (local)"}[p],
        ) or "groq"
        if provider == "groq":
            api_key = st.text_input("Clé API Groq", value=os.getenv("GROQ_API_KEY", ""), type="password")
            st.caption("Clé gratuite sur [console.groq.com](https://console.groq.com/keys)")
            leader_id, member_id = DEFAULT_LEADER_MODEL, DEFAULT_MEMBER_MODEL
        else:
            api_key, leader_id = None, st.text_input("Modèle Ollama", value=DEFAULT_OLLAMA_MODEL)
            member_id = leader_id

        output_format = st.segmented_control("Format de sortie", list(OUTPUT_FORMATS), default="📄 Rapport", width="stretch")
        output_format = output_format or "📄 Rapport"

        if st.button("🗑️ Nouvelle conversation", width="stretch"):
            for key in ("session_id", "messages", "known_urls"):
                st.session_state.pop(key, None)

        st.divider()
        st.markdown("### 🤝 L'équipe")
        st.markdown(
            f"<div class='agent-card'><b>🧭 Team Leader</b><span>{leader_id} · planifie et rédige</span></div>",
            unsafe_allow_html=True,
        )
        for icon, name, desc in MEMBERS_UI.values():
            st.markdown(f"<div class='agent-card'><b>{icon} {name}</b><span>{desc}</span></div>", unsafe_allow_html=True)
        st.caption(f"Membres : {member_id}")
    return provider, api_key, leader_id, member_id, output_format


def render_welcome() -> str | None:
    """Empty state with example topics. Returns a topic if one was clicked."""
    st.markdown("#### 👋 Que veux-tu explorer ?")
    st.markdown(
        "Donne un sujet : l'équipe cherche les discussions Hacker News, lit les articles, "
        "vérifie le contexte sur le web, puis rédige une synthèse sourcée."
    )
    columns = st.columns(len(EXAMPLE_TOPICS))
    for column, topic in zip(columns, EXAMPLE_TOPICS):
        if column.button(topic, width="stretch"):
            return topic
    return None


def stream_answer(team: Team, prompt: str, session_id: str, known_urls: set[str]) -> tuple[str, dict]:
    """Run the team, showing each agent's work live. Returns the markdown and run stats.

    URLs seen in the members' tool calls are added to ``known_urls`` so the
    answer's sources can be checked against them.
    """
    answer, stats, started = "", {"tokens": None}, time.time()
    status = st.status("🧭 Le leader planifie la recherche…", expanded=True)
    answer_box = st.empty()  # the answer streams here, below the progress panel
    for event in team.run(prompt, stream=True, stream_events=True, session_id=session_id):
        if event.event == MEMBER_TOOL_CALL_STARTED and event.tool:
            args = ", ".join(f"{v}" for v in (event.tool.tool_args or {}).values())[:80]
            status.caption(f"↳ `{event.tool.tool_name}` {args}")
        elif event.event == MEMBER_TOOL_CALL_COMPLETED and event.tool:
            known_urls |= extract_urls(f"{event.tool.tool_args} {event.tool.result}")
        elif event.event == TeamRunEvent.tool_call_started and event.tool:
            args = event.tool.tool_args or {}
            icon, name = member_ui(args.get("member_id", ""))
            status.update(label=f"{icon} {name} travaille…")
            status.markdown(f"**{icon} {name}** · {args.get('task', '')[:160]}")
        elif event.event == TeamRunEvent.run_content and isinstance(event.content, str):
            if not answer:
                status.update(label="✍️ Rédaction de la réponse…")
            answer += event.content
            answer_box.markdown(answer)
        elif event.event == TeamRunEvent.run_error:
            status.update(label="Échec de la recherche", state="error")
            raise RuntimeError(friendly_error(event.content or ""))
        elif event.event == TeamRunEvent.run_completed and getattr(event, "metrics", None):
            stats["tokens"] = event.metrics.total_tokens
    stats["seconds"] = time.time() - started
    status.update(label=f"✅ Recherche terminée en {stats['seconds']:.0f} s", state="complete", expanded=False)
    return answer, stats


def render_meta(meta: dict) -> None:
    """Small footer under an answer: duration, tokens and source check."""
    parts = [f"⏱️ {meta['seconds']:.0f} s"]
    if meta.get("tokens"):
        parts.append(f"🔢 {meta['tokens']:,} tokens".replace(",", " "))
    parts.append(f"⚠️ {len(meta['unverified'])} lien(s) non vérifié(s)" if meta["unverified"] else "✅ Sources vérifiées")
    if meta["unverified"]:
        links = "\n".join(f"- {url}" for url in meta["unverified"])
        st.warning(f"Ces liens n'ont été renvoyés par aucun outil, ils sont peut-être inventés :\n\n{links}")
    st.markdown(f"<div class='run-meta'>{' · '.join(parts)}</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🔍", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><h1>🔍 Multi-Agent Researcher</h1>"
        "<p>Une équipe d'agents Agno qui explore Hacker News et rédige rapports, articles et posts sourcés.</p></div>",
        unsafe_allow_html=True,
    )

    provider, api_key, leader_id, member_id, output_format = render_sidebar()
    if provider == "groq" and not api_key:
        st.info("🔑 Entre ta clé Groq gratuite dans la barre latérale pour commencer.")
        return

    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("known_urls", set())

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message.get("meta"):
                render_meta(message["meta"])

    example = render_welcome() if not st.session_state.messages else None
    query = st.chat_input("Un sujet à rechercher, ou une question de suivi…") or example
    if not query:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    prompt = f"{query}\n\nWrite the answer as {OUTPUT_FORMATS[output_format]}."

    with st.chat_message("assistant", avatar="🤖"):
        try:
            team = get_team(provider, leader_id, member_id, api_key)
            answer, stats = stream_answer(team, prompt, st.session_state.session_id, st.session_state.known_urls)
        except Exception as exc:  # run errors, network down, Ollama not running, ...
            st.error(str(exc) if isinstance(exc, RuntimeError) else friendly_error(str(exc)))
            st.session_state.messages.pop()  # let the user retry the same question
            return
        meta = {**stats, "unverified": unverified_urls(answer, st.session_state.known_urls)}
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.rerun()  # redraw from history: hides the welcome block and the progress panel


if __name__ == "__main__":
    main()
