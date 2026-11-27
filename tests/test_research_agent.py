"""Offline tests: no LLM call and no real network access."""

import json

import httpx
import pytest
import respx
from agno.models.groq import Groq
from agno.models.ollama import Ollama

from hn_tools import ALGOLIA_SEARCH_URL, HN_ITEM_URL, clean_hn_text, get_top_comments, search_hackernews_stories
from research_agent import build_model, build_team, extract_urls, friendly_error, unverified_urls

# --- Model factory -----------------------------------------------------------


def test_build_model_groq_uses_key_and_retries():
    model = build_model("groq", "llama-3.3-70b-versatile", api_key="gsk_test")
    assert isinstance(model, Groq)
    assert model.id == "llama-3.3-70b-versatile"
    assert model.api_key == "gsk_test"
    assert model.retries > 0


def test_build_model_groq_without_key_fails():
    with pytest.raises(ValueError):
        build_model("groq", "llama-3.3-70b-versatile", api_key="")


def test_build_model_ollama_limits_output():
    model = build_model("ollama", "llama3.2")
    assert isinstance(model, Ollama)
    assert model.options == {"num_predict": 1024}


def test_build_model_unknown_provider():
    with pytest.raises(ValueError):
        build_model("openai", "gpt-4o")


# --- Team factory ------------------------------------------------------------


def test_build_team_structure():
    leader = build_model("groq", "leader", api_key="gsk_test")
    member = build_model("groq", "member", api_key="gsk_test")
    team = build_team(leader, member)

    assert team.model is leader
    assert [m.name for m in team.members] == ["HackerNews Researcher", "Article Reader", "Web Searcher"]
    assert all(m.model is member for m in team.members)
    assert all(m.tool_call_limit for m in team.members)

    hn_tool_names = {t.name for t in team.members[0].tools}
    assert hn_tool_names == {"search_hackernews_stories", "get_top_comments"}


# --- Error messages ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"error":{"code":"invalid_api_key"}}', "Clé API invalide"),
        ('{"error":{"code":"rate_limit_exceeded"}}', "Quota gratuit"),
        ('{"error":{"code":"model_not_found"}}', "plus disponible"),
        ("Failed to connect to Ollama", "Ollama"),
        ("something odd", "something odd"),
    ],
)
def test_friendly_error(raw, expected):
    assert expected in friendly_error(raw)


# --- Source verification -----------------------------------------------------


def test_extract_urls_strips_punctuation_and_markdown():
    text = "See [post](https://a.dev/x/), https://news.ycombinator.com/item?id=1. Done"
    assert extract_urls(text) == {"https://a.dev/x", "https://news.ycombinator.com/item?id=1"}


def test_unverified_urls_flags_only_unknown_links():
    answer = "## Sources\n- https://a.dev/x\n- https://invented.example/post"
    assert unverified_urls(answer, known_urls={"https://a.dev/x"}) == ["https://invented.example/post"]


# --- Custom Hacker News tools ------------------------------------------------


def test_clean_hn_text_strips_html_and_truncates():
    assert clean_hn_text("Hello<p>it&#x27;s <i>great</i>") == "Hello\nit's great"
    assert clean_hn_text("a" * 20, max_chars=5) == "aaaaa…"
    assert clean_hn_text(None) == ""


@respx.mock
def test_search_hackernews_stories_returns_compact_json():
    respx.get(ALGOLIA_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"hits": [{"objectID": "42", "title": "Agno", "url": "https://x.dev", "points": 10, "num_comments": 3, "author": "bob", "created_at": "2026-03-01T10:00:00Z"}]},
        )
    )
    stories = json.loads(search_hackernews_stories.entrypoint(query="agents", max_results=50))

    assert stories == [
        {"id": 42, "title": "Agno", "url": "https://x.dev", "hn_url": "https://news.ycombinator.com/item?id=42", "points": 10, "num_comments": 3, "created_at": "2026-03-01"}
    ]
    assert respx.calls.last.request.url.params["hitsPerPage"] == "5"  # clamped
    assert "numericFilters" not in respx.calls.last.request.url.params


@respx.mock
def test_search_hackernews_stories_recent_only_filters_by_date():
    respx.get(ALGOLIA_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))
    search_hackernews_stories.entrypoint(query="rust", recent_only=True)
    assert respx.calls.last.request.url.params["numericFilters"].startswith("created_at_i>")


@respx.mock
def test_get_top_comments_skips_deleted_and_respects_limit():
    respx.get(HN_ITEM_URL.format(item_id=1)).mock(return_value=httpx.Response(200, json={"id": 1, "kids": [2, 3, 4, 5]}))
    respx.get(HN_ITEM_URL.format(item_id=2)).mock(return_value=httpx.Response(200, json={"by": "alice", "text": "First<p>point"}))
    respx.get(HN_ITEM_URL.format(item_id=3)).mock(return_value=httpx.Response(200, json={"deleted": True}))
    respx.get(HN_ITEM_URL.format(item_id=4)).mock(return_value=httpx.Response(200, json={"by": "carol", "text": "Second"}))

    comments = json.loads(get_top_comments.entrypoint(story_id=1, limit=2))

    assert comments == [{"author": "alice", "text": "First\npoint"}, {"author": "carol", "text": "Second"}]


@respx.mock
def test_hn_tools_return_error_json_on_http_failure():
    respx.get(ALGOLIA_SEARCH_URL).mock(return_value=httpx.Response(503))
    respx.get(HN_ITEM_URL.format(item_id=1)).mock(side_effect=httpx.ConnectError("down"))

    assert "error" in json.loads(search_hackernews_stories.entrypoint(query="x"))
    assert "error" in json.loads(get_top_comments.entrypoint(story_id=1))
