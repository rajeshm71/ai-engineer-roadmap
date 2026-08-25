"""Local link checker for README.md.

Run before opening a PR that touches links: python scripts/check_links.py

CI uses lychee (see .github/workflows/link-check.yml); this script is a
faster local pre-check that needs only the stdlib.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

README = Path(__file__).resolve().parent.parent / "README.md"
LINK_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
TIMEOUT = 10
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_BACKOFF_SECONDS = 5

# Hosts confirmed (via curl with a real browser TLS stack) to serve real
# 200s to humans while blocking Python's urllib on fingerprint, not on
# User-Agent header alone. Treated as warnings, not failures, so a
# contributor isn't sent chasing a link that isn't actually dead.
KNOWN_BOT_BLOCKED_HOSTS = ("oreilly.com",)


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
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with urlopen(request, timeout=TIMEOUT) as response:
                return url, response.status
        except HTTPError as exc:
            if exc.code == 429 and attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            return url, exc.code
        except URLError as exc:
            return url, str(exc.reason)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    urls = extract_urls(text)
    print(f"Checking {len(urls)} links in {README.name}...")

    failures = []
    warnings = []
    for url in urls:
        _, result = check(url)
        ok = isinstance(result, int) and result < 400
        known_flaky = any(host in url for host in KNOWN_BOT_BLOCKED_HOSTS)
        if ok:
            label = "OK"
        elif known_flaky:
            label = "WARN"
        else:
            label = "FAIL"
        print(f"  [{label}] {result} {url}")
        if not ok and known_flaky:
            warnings.append((url, result))
        elif not ok:
            failures.append((url, result))

    if warnings:
        print(f"\n{len(warnings)} link(s) blocked known bot-detection hosts (not treated as failures):")
        for url, result in warnings:
            print(f"  {result}: {url}")

    if failures:
        print(f"\n{len(failures)} link(s) failed:")
        for url, result in failures:
            print(f"  {result}: {url}")
        return 1

    print("\nAll links resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
