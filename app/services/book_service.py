"""
Book Knowledge Service: Open Library & Wikipedia Search Client.
Single-purpose design ready for LangChain Tool conversion in Phase 4.
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional
from app.core.logging import logger


def fetch_web_knowledge(query: str) -> Optional[str]:
    search_term = re.sub(r"^(who is|what is|search|tell me about)\s+", "", query, flags=re.IGNORECASE).strip()
    search_term = re.sub(r"[^\w\s]", "", search_term)
    if not search_term:
        return None

    try:
        encoded_term = urllib.parse.quote(search_term.title().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_term}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            extract = data.get("extract")
            title = data.get("title", search_term.title())
            if extract and len(extract) > 20:
                return f"[WEB SEARCH RESULT: {title}]\n{extract}"
    except Exception as e:
        logger.error(f"Web knowledge search error: {e}")
    return None


def fetch_book_knowledge(query: str) -> Optional[str]:
    """
    Queries Open Library & Wikipedia for book title summaries.
    
    Signature: (query: str) -> Optional[str]
    Single-purpose contract ready for Phase 4 @tool wrapping.
    """
    book_title = re.sub(r"^(book title|summary of book|summary of|about book|read book|book)\s+", "", query, flags=re.IGNORECASE).strip()
    book_title = re.sub(r"[^\w\s]", "", book_title)

    if not book_title or len(book_title) < 2:
        return None

    try:
        encoded_title = urllib.parse.quote(book_title)
        ol_url = f"https://openlibrary.org/search.json?title={encoded_title}&limit=1"
        req = urllib.request.Request(ol_url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            docs = data.get("docs")

            if docs:
                book = docs[0]
                title = book.get("title", book_title.title())
                authors = ", ".join(book.get("author_name", ["Unknown Author"]))
                first_year = book.get("first_publish_year", "N/A")
                subjects = ", ".join(book.get("subject", [])[:4]) if book.get("subject") else "General Reading"

                wiki_summary = fetch_web_knowledge(f"Book {title}") or fetch_web_knowledge(title)

                report = (
                    f"[BOOK KNOWLEDGE SUMMARY]\n"
                    f"Title: {title}\n"
                    f"Author(s): {authors}\n"
                    f"First Published: {first_year}\n"
                    f"Genre / Topics: {subjects}\n"
                    f"--------------------------------------------------\n"
                    f"EXECUTIVE OVERVIEW:\n"
                    f"'{title}' by {authors} covers {subjects}.\n"
                )

                if wiki_summary:
                    clean_text = wiki_summary.split("\n", 1)[-1] if "\n" in wiki_summary else wiki_summary
                    report += f"\nBOOK SYNOPSIS:\n{clean_text}\n"

                report += "--------------------------------------------------"
                return report
    except Exception as e:
        logger.error(f"Book tool error: {e}")

    return fetch_web_knowledge(f"Book {book_title}") or fetch_web_knowledge(book_title)
