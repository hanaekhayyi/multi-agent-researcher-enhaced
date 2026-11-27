"""Custom Hacker News tools for the research team.

The built-in ``HackerNewsTools`` only returns the current front page, it cannot
search by topic. These two tools fill that gap and return compact JSON so the
free Groq tier is not flooded with tokens.
"""

import html
import json
import re
import time

import httpx
from agno.tools import tool

ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
HN_STORY_URL = "https://news.ycombinator.com/item?id={item_id}"
TIMEOUT_SECONDS = 10
MAX_COMMENT_CHARS = 300
ONE_YEAR_SECONDS = 365 * 24 * 3600


def clean_hn_text(raw: str | None, max_chars: int = MAX_COMMENT_CHARS) -> str:
    """Turn HN's HTML comment body into short plain text."""
    if not raw:
        return ""
    text = re.sub(r"<p>", "\n", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


@tool
def search_hackernews_stories(query: str, max_results: int = 3, recent_only: bool = False) -> str:
    """Search Hacker News stories about a topic, most relevant first. One call is enough.

    Args:
        query (str): Topic or keywords to search for. Do not add years: use recent_only instead.
        max_results (int): Number of stories to return (1 to 5). Defaults to 3.
        recent_only (bool): Only return stories from the last 12 months. Defaults to False.

    Returns:
        str: JSON list of stories with id, title, url, hn_url, points, num_comments, created_at.
    """
    params = {"query": query, "tags": "story", "hitsPerPage": max(1, min(max_results, 5))}
    if recent_only:
        params["numericFilters"] = f"created_at_i>{int(time.time()) - ONE_YEAR_SECONDS}"
    try:
        response = httpx.get(ALGOLIA_SEARCH_URL, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        hits = response.json().get("hits", [])
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Hacker News search failed: {exc}"})

    stories = [
        {
            "id": int(hit["objectID"]),
            "title": hit.get("title"),
            "url": hit.get("url"),
            "hn_url": HN_STORY_URL.format(item_id=hit["objectID"]),
            "points": hit.get("points"),
            "num_comments": hit.get("num_comments"),
            "created_at": (hit.get("created_at") or "")[:10],
        }
        for hit in hits
    ]
    return json.dumps(stories)


@tool
def get_top_comments(story_id: int, limit: int = 3) -> str:
    """Get the top-level comments of a Hacker News story, in HN's own ranking order.

    The HN API does not expose comment scores, but the order of a story's
    comments already reflects HN's ranking, so the first ones are the best rated.

    Args:
        story_id (int): The Hacker News story id.
        limit (int): Number of comments to return (1 to 5). Defaults to 3.

    Returns:
        str: JSON list of comments with author and text.
    """
    limit = max(1, min(limit, 5))
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            story = client.get(HN_ITEM_URL.format(item_id=story_id)).raise_for_status().json()
            if not story:
                return json.dumps({"error": f"Story {story_id} not found"})

            comments = []
            for comment_id in story.get("kids", []):
                if len(comments) >= limit:
                    break
                comment = client.get(HN_ITEM_URL.format(item_id=comment_id)).raise_for_status().json()
                if not comment or comment.get("deleted") or comment.get("dead"):
                    continue
                comments.append({"author": comment.get("by"), "text": clean_hn_text(comment.get("text"))})
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Fetching comments failed: {exc}"})

    return json.dumps(comments)
