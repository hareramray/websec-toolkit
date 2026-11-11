"""Check one page for clickjacking and CSRF weaknesses.

Usage:
  python src/scan_site.py https://your-site.example/account
  python src/scan_site.py https://your-site.example/account --cookie "sessionid=abc123"

Sends a single GET request, the same as opening the page in a browser, and reads the
response headers, cookies and forms. Nothing is submitted. Only scan sites you own
or have permission to test.

These two weaknesses are configuration mistakes with exact definitions, so they are
checked with rules, not with a trained model:
  clickjacking  the page may be framed by another site (no X-Frame-Options header
                and no CSP frame-ancestors directive)
  CSRF          a state-changing form has no anti-CSRF token, or cookies are sent
                on cross-site requests (SameSite missing or None)

Exit code: 0 no issues, 1 issues found, 2 the page could not be fetched.
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

TOKEN_NAME = re.compile(r"csrf|xsrf|token|authenticity|nonce|__requestverification", re.I)
FAIL, WARN, PASS, INFO = "FAIL", "WARN", "PASS", "INFO"


class PageParser(HTMLParser):
    """Collects forms (method, action, field names) and CSRF <meta> tokens."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict] = []
        self.meta_token = False
        self._open: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = {name: value or "" for name, value in attrs}
        if tag == "form":
            self._open = {"method": attr.get("method", "").strip().lower() or "get", "action": attr.get("action", ""), "fields": []}
            self.forms.append(self._open)
        elif tag in ("input", "select", "textarea", "button") and self._open is not None:
            if attr.get("name"):
                self._open["fields"].append(attr["name"])
        elif tag == "meta" and TOKEN_NAME.search(attr.get("name", "")):
            self.meta_token = True

    def handle_endtag(self, tag):
        if tag == "form":
            self._open = None


def frame_ancestors(csp: str) -> list[str] | None:
    """Sources of the frame-ancestors directive, or None when it is absent."""
    for policy in csp.split(","):
        for directive in policy.split(";"):
            parts = directive.split()
            if parts and parts[0].lower() == "frame-ancestors":
                return [p.lower() for p in parts[1:]]
    return None


def check_clickjacking(headers) -> list[tuple[str, str]]:
    findings = []
    ancestors = frame_ancestors(headers.get("Content-Security-Policy", ""))
    xfo = {v.strip().lower() for v in headers.get("X-Frame-Options", "").split(",") if v.strip()}
    fix = "Fix: send  Content-Security-Policy: frame-ancestors 'self'  plus  X-Frame-Options: SAMEORIGIN  for old browsers."

    if ancestors is not None:
        open_sources = [s for s in ancestors if s in ("*", "http:", "https:")]
        if open_sources:
            findings.append((FAIL, f"CSP frame-ancestors allows {' '.join(open_sources)}, so any site can still frame this page. {fix}"))
        else:
            findings.append((PASS, f"CSP frame-ancestors {' '.join(ancestors) or chr(39) + 'none' + chr(39)} restricts who can frame this page."))
    elif xfo and xfo <= {"deny", "sameorigin"}:
        findings.append((PASS, f"X-Frame-Options: {', '.join(sorted(xfo)).upper()} blocks framing by other sites."))
        findings.append((INFO, "Consider also sending CSP frame-ancestors, the current standard."))
    elif xfo:
        findings.append((FAIL, f"X-Frame-Options: {', '.join(sorted(xfo))} is not honoured by current browsers (only DENY and SAMEORIGIN are). {fix}"))
    else:
        findings.append((FAIL, f"No X-Frame-Options header and no CSP frame-ancestors directive: any site can embed this page in an invisible frame. {fix}"))

    if ancestors is None and frame_ancestors(headers.get("Content-Security-Policy-Report-Only", "")) is not None:
        findings.append((INFO, "frame-ancestors is present only in a Report-Only policy, which does not block anything."))
    return findings


def set_cookie_headers(response) -> list[str]:
    """Every Set-Cookie header from the response and the redirects that led to it."""
    out = []
    for r in [*response.history, response]:
        out += r.raw.headers.getlist("Set-Cookie") if hasattr(r.raw.headers, "getlist") else []
    return out


def check_csrf(response, page: PageParser) -> list[tuple[str, str]]:
    findings = []

    unsafe_forms = [f for f in page.forms if f["method"] != "get"]
    for form in unsafe_forms:
        target = urljoin(response.url, form["action"]) if form["action"] else response.url
        if any(TOKEN_NAME.search(name) for name in form["fields"]):
            findings.append((PASS, f"Form {form['method'].upper()} {target} carries an anti-CSRF token field."))
        elif page.meta_token:
            findings.append((INFO, f"Form {form['method'].upper()} {target} has no token field, but the page has a CSRF <meta> token; check that the server rejects the request without it."))
        else:
            fields = ", ".join(form["fields"]) or "none"
            findings.append((WARN, f"Form {form['method'].upper()} {target} has no anti-CSRF token field (fields: {fields}). "
                                   "Fix: add a per-session random token to the form and verify it on the server."))
    if not unsafe_forms:
        findings.append((INFO, f"No POST forms on this page ({len(page.forms)} form(s) in total). Forms built by JavaScript are not visible to this check."))

    for header in set_cookie_headers(response):
        name = header.split("=", 1)[0].strip()
        attributes = {a.strip().lower() for a in header.split(";")[1:]}
        same_site = next((a.split("=", 1)[1].strip() for a in attributes if a.startswith("samesite=")), None)
        if same_site == "none":
            findings.append((WARN, f'Cookie "{name}" has SameSite=None: it is sent with requests from other sites, so every state-changing request needs a CSRF token.'))
        elif same_site is None:
            findings.append((WARN, f'Cookie "{name}" has no SameSite attribute: protection depends on the browser default. Fix: set SameSite=Lax or Strict.'))
        else:
            findings.append((PASS, f'Cookie "{name}" has SameSite={same_site.capitalize()}.'))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="page to check, starting with http:// or https://")
    parser.add_argument("--cookie", help='Cookie header for a logged-in check, e.g. "sessionid=abc123"')
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        parser.error("the URL must start with http:// or https://")

    headers = {"User-Agent": "websec-check/1.0"}
    if args.cookie:
        headers["Cookie"] = args.cookie
    try:
        response = requests.get(args.url, headers=headers, timeout=args.timeout)
    except requests.RequestException as error:
        print(f"could not fetch {args.url}: {error}")
        return 2

    page = PageParser()
    if "html" in response.headers.get("Content-Type", "").lower():
        page.feed(response.text)

    print(f"Checked {response.url} (HTTP {response.status_code})")
    sections = {"Clickjacking": check_clickjacking(response.headers), "CSRF": check_csrf(response, page)}
    for title, findings in sections.items():
        print(f"\n{title}")
        for level, message in findings:
            print(f"  [{level}] {message}")

    issues = [level for findings in sections.values() for level, _ in findings if level in (FAIL, WARN)]
    print(f"\n{len(issues)} issue(s) found." if issues else "\nNo issues found by these checks.")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
