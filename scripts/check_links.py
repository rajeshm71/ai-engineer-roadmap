"""Local link checker for README.md.

Run before opening a PR that touches links: python scripts/check_links.py

CI uses lychee (see .github/workflows/link-check.yml); this script is a
faster local pre-check that needs only the stdlib and requests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

README = Path(__file__).resolve().parent.parent / "README.md"
LINK_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
TIMEOUT = 10


def extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in LINK_PATTERN.finditer(text):
        url = match.group(1)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def check(url: str) -> tuple[str, int | str]:
    request = Request(url, headers={"User-Agent": "ai-engineer-roadmap-link-check"})
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            return url, response.status
    except HTTPError as exc:
        return url, exc.code
    except URLError as exc:
        return url, str(exc.reason)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    urls = extract_urls(text)
    print(f"Checking {len(urls)} links in {README.name}...")

    failures = []
    for url in urls:
        _, result = check(url)
        ok = isinstance(result, int) and result < 400
        print(f"  [{'OK' if ok else 'FAIL'}] {result} {url}")
        if not ok:
            failures.append((url, result))

    if failures:
        print(f"\n{len(failures)} link(s) failed:")
        for url, result in failures:
            print(f"  {result}: {url}")
        return 1

    print("\nAll links resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
