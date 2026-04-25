"""All WikEM API helpers — title harvesting, page fetching, junk filtering."""
import requests
from bs4 import BeautifulSoup

WIKEM_API = "https://wikem.org/w/api.php"
HEADERS = {"User-Agent": "CrisisAppDataPackBuilder/1.0"}

# Patterns that mark a page as academic / non-actionable.
FORBIDDEN_WORDS = [
    "EBQ:", "Policy", "High altitude", "Journal Club",
    "Harbor:", "Maine:", "kg (", "calculator", "score", "protocol",
]


def is_junk(title: str) -> bool:
    """True if the page title looks like academic/irrelevant content."""
    lower = title.lower()
    return any(w.lower() in lower for w in FORBIDDEN_WORDS)


def list_pages_in_category(category_name: str) -> list[str]:
    """All page titles under a WikEM category, e.g. 'Category:Toxicology'."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name,
        "cmlimit": "500",
        "format": "json",
    }
    resp = requests.get(WIKEM_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"WikEM API error: {data['error']['info']}")
    return [item["title"] for item in data["query"]["categorymembers"]]


def fetch_wikem_page(page_title: str) -> str:
    """Clean text content of a single WikEM page (HTML stripped)."""
    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "text",
        "redirects": "true",
    }
    resp = requests.get(WIKEM_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"WikEM error for '{page_title}': {data['error']['info']}")

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    # Drop nav / edit / reference cruft
    for selector in [".toc", ".mw-editsection", ".reference", ".navbox"]:
        for el in soup.select(selector):
            el.decompose()
    return soup.get_text(separator="\n", strip=True)


def page_url(title: str) -> str:
    return f"https://wikem.org/wiki/{title.replace(' ', '_')}"
