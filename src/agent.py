"""Security agent: works towards a goal by choosing its own next step.

Usage:
  python src/agent.py "audit https://your-site.example/search?q=shoes"
  python src/agent.py "audit https://your-site.example/" --active
  python src/agent.py "triage logs/access.log"
  python src/agent.py "is this xss: <img src=x onerror=alert(1)>"

How it works. The agent keeps an agenda of possible next steps, each with a priority
and a reason. It runs the most useful one, reads the result, and lets that result add,
drop or re-order steps: a page with parameters leads to probes, a link the detector
flags is visited first, a log line just under the threshold is re-examined value by
value. It stops when nothing useful is left or a budget is used up, then reports
every step it took and every finding with its evidence.

The planner is rule-based and the detector is the trained XSS model. No language
model decides anything, so the agent only pursues the three goals above.

Passive by default: pages are fetched with GET, like a browser would. With --active
the agent also sends each query parameter one harmless marker value and checks how
it comes back. It never sends a working payload, never submits POST forms, and stays
on the starting site. Only audit sites you own or have permission to test.

Exit code: 0 nothing above informational, 1 findings, 2 the goal could not be run.
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_site import FAIL, WARN, analyze, set_cookie_headers  # noqa: E402
from xss_model import default_model_path, explain, load_model, normalize, score  # noqa: E402

MAX_BODY = 2_000_000  # characters of a page that are inspected
MAX_LINES = 200_000
MAX_DEPTH = 2
PROBE_CHARS = "<'\">"
SEVERITIES = ["high", "medium", "low", "info"]

_URL = re.compile(r"https?://\S+", re.I)
_RISKY = re.compile(r"logout|log-out|signout|sign-out|delete|remove|destroy|unsubscribe|deactivate", re.I)
_STATIC = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|ico|css|js|pdf|zip|gz|mp[34]|webm|woff2?|ttf|xml|json|txt)$", re.I)
_REQUEST = re.compile(r'"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) (\S+) HTTP/[\d.]+"')
_TOKEN_FIELD = re.compile(r"csrf|xsrf|token|nonce|authenticity", re.I)
_DOM_SOURCE = re.compile(r"location\.(?:hash|search|href)|document\.(?:url|documenturi|referrer)|window\.name", re.I)
_DOM_SINK = re.compile(r"\.innerhtml\s*=|\.outerhtml\s*=|document\.write(?:ln)?\(|\beval\(|insertadjacenthtml\(|settimeout\(\s*['\"]", re.I)
_URL_ATTRIBUTE = re.compile(r"(?:href|src|action|formaction|data)\s*=\s*[\"']?$", re.I)
_HANDLER_ATTRIBUTE = re.compile(r"\bon[a-z]+\s*=\s*[\"']?[^\"'>]*$", re.I)
_ENCODED = {
    "<": ("&lt;", "&#60;", "&#x3c;", "%3c", "\\u003c", "\\x3c"),
    "'": ("&#39;", "&#x27;", "&apos;", "%27", "\\'", "\\u0027", "\\x27"),
    '"': ("&quot;", "&#34;", "&#x22;", "%22", '\\"', "\\u0022", "\\x22"),
    ">": ("&gt;", "&#62;", "&#x3e;", "%3e", "\\u003e", "\\x3e"),
}
_FAMILIES = [("script tag", re.compile(r"<\s*script")), ("javascript: URL", re.compile(r"javascript\s*:")),
             ("event handler", re.compile(r"\bon[a-z]+\s*=")), ("data: document", re.compile(r"data:text/html"))]


class Surface(HTMLParser):
    """What a page offers an attacker: links, forms and fields, inline scripts, inline handlers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.forms: list[dict] = []
        self.scripts: list[str] = []
        self.handlers = 0
        self._form: dict | None = None
        self._script: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attr = {name: value or "" for name, value in attrs}
        self.handlers += sum(name.startswith("on") for name in attr)
        if tag in ("a", "area") and attr.get("href"):
            self.links.append(attr["href"])
        elif tag == "form":
            self._form = {"method": attr.get("method", "").strip().lower() or "get", "action": attr.get("action", ""), "fields": []}
            self.forms.append(self._form)
        elif tag in ("input", "textarea", "select") and self._form is not None and attr.get("name"):
            self._form["fields"].append({"name": attr["name"], "type": attr.get("type", "text").lower(), "value": attr.get("value", "")})
        elif tag == "script" and not attr.get("src"):
            self._script = []

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        elif tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)


def origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def with_param(url: str, param: str, value: str) -> str:
    """`url` with one query parameter set to `value`; the others are kept."""
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if param not in [key for key, _ in pairs]:
        pairs.append((param, value))
    return urlunsplit(parts._replace(query=urlencode([(key, value if key == param else old) for key, old in pairs]), fragment=""))


def _context(body: str, index: int) -> tuple[str, str]:
    """Where position `index` of an HTML document sits: (context, quote character in force)."""
    before = body[:index]
    lower = before.lower()
    script_open = lower.rfind("<script")
    if script_open > lower.rfind("</script") and ">" in before[script_open:]:
        line = before[before.rfind("\n") + 1 :]
        for quote in ("'", '"', "`"):
            if (line.count(quote) - line.count("\\" + quote)) % 2:
                return "a script string", quote
        return "script code", ""
    if lower.rfind("<!--") > lower.rfind("-->"):
        return "an HTML comment", ""
    tag_open = before.rfind("<")
    if tag_open > before.rfind(">"):
        tag = before[tag_open:]
        quote = '"' if tag.count('"') % 2 else "'" if tag.count("'") % 2 else ""
        if _HANDLER_ATTRIBUTE.search(tag):
            return "an event-handler attribute", quote
        if _URL_ATTRIBUTE.search(tag):
            return "a URL attribute", quote
        return ("a quoted attribute" if quote else "an unquoted attribute"), quote
    return "HTML text", ""


def reflections(body: str, marker: str) -> list[dict]:
    """Each place the probe value came back: its context, which special characters survived, and whether that is exploitable."""
    found, start = [], 0
    while len(found) < 20:
        index = body.find(marker, start)
        if index < 0:
            break
        start = index + len(marker)
        tail = body[start : start + 60]
        lowered, raw = tail.lower(), []
        for char in PROBE_CHARS:
            if lowered.startswith(char):
                raw.append(char)
                lowered = lowered[1:]
                continue
            for form in _ENCODED[char]:
                if lowered.startswith(form):
                    lowered = lowered[len(form) :]
                    break
        context, quote = _context(body, index)
        if context in ("HTML text", "an HTML comment"):
            exploitable = "<" in raw
        elif context == "a script string":
            exploitable = quote in raw or "<" in raw
        elif context in ("script code", "an event-handler attribute", "an unquoted attribute"):
            exploitable = True
        elif context == "a URL attribute":
            exploitable = True  # the value may start a javascript: URL even when quotes are encoded
        else:
            exploitable = quote in raw
        found.append({"context": context, "raw": "".join(raw), "exploitable": exploitable,
                      "snippet": body[max(0, index - 60) : start + 30].replace("\n", " ").strip()})
    return found


class Agent:
    """Runs one goal to completion. `bundle` is a loaded XSS model: {"pipeline", "threshold"}."""

    def __init__(self, bundle: dict, *, active: bool = False, max_pages: int = 8, max_probes: int = 12, max_steps: int = 60,
                 delay: float = 0.3, timeout: float = 15, cookie: str | None = None, log=None):
        self.pipeline, self.threshold = bundle["pipeline"], float(bundle["threshold"])
        self.active, self.max_pages, self.max_probes, self.max_steps = active, max_pages, max_probes, max_steps
        self.delay, self.timeout, self.log = delay, timeout, log
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "websec-check/1.0"
        if cookie:
            self.http.headers["Cookie"] = cookie
        self.agenda: list[dict] = []
        self.trace: list[dict] = []
        self.findings: dict[tuple, dict] = {}
        self.pages: dict[str, dict] = {}
        self.queued: set[str] = set()
        self.endpoints: dict[tuple[str, str], str] = {}
        self.site = ""
        self.requests_sent = self.probes = self.post_forms = self._order = 0
        self.notes: list[str] = []
        self.entries: list[tuple[int, str]] = []
        self.hits: list[tuple[float, int, str]] = []
        self.kind = ""

    # ---- agenda -----------------------------------------------------------------

    def add(self, priority: int, action: str, why: str, **args) -> None:
        self._order += 1
        self.agenda.append({"priority": priority, "order": self._order, "action": action, "why": why, "args": args})

    def finding(self, severity: str, title: str, evidence: str, fix: str = "", where: str = "") -> None:
        key = (severity, title, evidence)
        if key in self.findings:
            self.findings[key]["count"] += 1
        else:
            self.findings[key] = {"severity": severity, "title": title, "evidence": evidence, "fix": fix, "where": where, "count": 1}

    def run(self, goal: str) -> dict:
        self._plan(goal.strip())
        while self.agenda and len(self.trace) < self.max_steps:
            task = max(self.agenda, key=lambda t: (t["priority"], -t["order"]))
            self.agenda.remove(task)
            try:
                observation = getattr(self, "_do_" + task["action"])(**task["args"])
            except (requests.RequestException, OSError, ValueError) as error:
                observation = f"failed: {error}"
            target = next((str(value) for value in task["args"].values() if isinstance(value, str)), "")
            step = {"step": len(self.trace) + 1, "action": task["action"], "target": target[:160], "why": task["why"], "observation": observation}
            self.trace.append(step)
            if self.log:
                self.log(f"{step['step']:>3}. {step['action']} {step['target'][:90]}\n       why: {step['why']}\n       saw: {observation}")
        if self.agenda:
            self.notes.append(f"Stopped at the step budget ({self.max_steps}); {len(self.agenda)} planned step(s) were not run.")
        return self._report(goal)

    def _plan(self, goal: str) -> None:
        url = _URL.search(goal)
        if url:
            start = url.group(0).rstrip(".,)\"'")
            self.kind, self.site = "audit", origin(start)
            self.add(100, "fetch", "the goal names this page, so it is the starting point", url=start, depth=0)
            return
        for word in [goal, *goal.split()]:
            candidate = Path(word.strip("\"'"))
            try:
                found = len(word) < 260 and candidate.is_file()
            except (OSError, ValueError):  # text that is not a legal path
                found = False
            if found:
                self.kind = "triage"
                self.add(100, "read_file", "the goal names this file", path=str(candidate))
                return
        self.kind = "classify"
        text = re.sub(r"^(?:is (?:this|it) (?:an? )?xss\??|check|classify|analy[sz]e)\s*:?\s*", "", goal, flags=re.I) or goal
        self.add(100, "classify_text", "no URL or file in the goal, so the text itself is the thing to judge", text=text)

    # ---- auditing a site --------------------------------------------------------

    def _get(self, url: str):
        """GET `url`, following redirects only while they stay on the audited site.

        Returns (response, other site or None). A redirect that leaves the site is
        not requested at all.
        """
        for _ in range(6):
            if self.requests_sent:
                time.sleep(self.delay)
            self.requests_sent += 1
            response = self.http.get(url, timeout=self.timeout, allow_redirects=False)
            location = response.headers.get("Location")
            if not (response.is_redirect and location):
                break
            url = urljoin(response.url, location)
            if origin(url) != self.site:
                return response, origin(url)
        return response, None

    def _do_fetch(self, url: str, depth: int) -> str:
        if len(self.pages) >= self.max_pages:
            if not any("Page budget" in note for note in self.notes):
                self.notes.append(f"Page budget of {self.max_pages} reached; some links were not visited.")
            return "skipped: page budget used"
        response, elsewhere = self._get(url)
        if elsewhere:
            return f"redirects to another site ({elsewhere}); not followed"
        if response.url in self.pages:
            return "already seen after redirect"
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower() or "unknown type"
        html = response.text[:MAX_BODY] if "html" in content_type else ""
        self.pages[response.url] = {"response": response, "html": html, "depth": depth}
        self.add(90, "check_headers", "response headers decide who may frame the page, how cookies travel and which scripts run", url=response.url)
        if html:
            self.add(85, "map_surface", "links, forms and parameters are where untrusted input enters", url=response.url)
        return f"HTTP {response.status_code}, {content_type}, {len(response.content):,} bytes"

    def _do_check_headers(self, url: str) -> str:
        page = self.pages[url]
        response = page["response"]
        before = len(self.findings)
        for section, items in analyze(response)["sections"].items():
            for level, message in items:
                if level in (FAIL, WARN):
                    problem, _, fix = message.partition(" Fix: ")
                    self.finding("medium" if level == FAIL else "low", f"{section} weakness", problem, fix, url)
        if page["html"]:
            policy = response.headers.get("Content-Security-Policy", "").lower()
            if "script-src" not in policy and "default-src" not in policy:
                self.finding("low", "No script policy", "The page sends no Content-Security-Policy that restricts scripts, so an injected script would run.",
                             "Send a Content-Security-Policy with a strict script-src (nonces or hashes).", url)
            elif "'unsafe-inline'" in policy:
                self.finding("low", "Script policy allows inline scripts", "The Content-Security-Policy contains 'unsafe-inline', so injected inline scripts still run.",
                             "Replace 'unsafe-inline' with nonces or hashes.", url)
        for header in set_cookie_headers(response):
            if "httponly" not in {part.strip().lower() for part in header.split(";")[1:]}:
                self.finding("low", "Cookie readable by scripts", f'Cookie "{header.split("=", 1)[0].strip()}" has no HttpOnly attribute, so an XSS payload can read it.',
                             "Set HttpOnly on session cookies.", url)
        return f"{len(self.findings) - before} new issue(s)"

    def _do_map_surface(self, url: str) -> str:
        page = self.pages[url]
        surface = Surface()
        surface.feed(page["html"])
        inputs: list[tuple[str, str]] = [(f"parameter {key} of this page", value) for key, value in parse_qsl(urlsplit(url).query)]
        targets = {url} if urlsplit(url).query else set()
        links = set()
        for href in surface.links:
            link = urljoin(url, href).split("#")[0]
            if link.startswith(("http://", "https://")) and origin(link) == self.site:
                links.add(link)
                if urlsplit(link).query:
                    targets.add(link)
                    inputs += [(f"parameter {key} of link {link[:80]}", value) for key, value in parse_qsl(urlsplit(link).query)]
        for form in surface.forms:
            action = urljoin(url, form["action"]) if form["action"] else url.split("?")[0]
            fields = [f for f in form["fields"] if f["type"] not in ("submit", "button", "file", "image", "reset", "checkbox", "radio", "password")]
            inputs += [(f"field {f['name']} of a form", f["value"]) for f in fields if f["value"]]
            if form["method"] != "get":
                self.post_forms += 1
            elif origin(action) == self.site and fields:
                defaults = [(f["name"], f["value"] or "test") for f in fields if not _TOKEN_FIELD.search(f["name"])]
                if defaults:
                    targets.add(action.split("?")[0] + "?" + urlencode(defaults))

        new_params = 0
        for target in sorted(targets):
            for key, value in parse_qsl(urlsplit(target).query, keep_blank_values=True):
                endpoint = (target.split("?")[0], key)
                if endpoint in self.endpoints or _RISKY.search(target):
                    continue
                self.endpoints[endpoint] = target
                new_params += 1
                if self.active:
                    echoed = len(value) >= 4 and value in page["html"]
                    self.add(78 if echoed else 70, "probe",
                             f"parameter {key} already echoes its value into the page, the first sign of reflection" if echoed
                             else f"parameter {key} takes input; a marker value shows whether it is reflected unencoded", url=target, param=key)
        page["inputs"] = [(where, value) for where, value in inputs if len(value) >= 3]
        if page["inputs"]:
            self.add(80, "classify_inputs", "values already inside links and forms may carry a payload; the detector can tell", url=url)

        for link in sorted(links):
            shape = link.split("?")[0] + "?" + ",".join(sorted(key for key, _ in parse_qsl(urlsplit(link).query)))
            if page["depth"] < MAX_DEPTH and shape not in self.queued and link not in self.pages and not _RISKY.search(link) and not _STATIC.search(urlsplit(link).path):
                self.queued.add(shape)
                has_query = bool(urlsplit(link).query)
                self.add(40 - 10 * page["depth"] + (10 if has_query else 0), "fetch",
                         "same-site link with parameters: more input to examine" if has_query else "same-site link: more of the site to examine",
                         url=link, depth=page["depth"] + 1)
        for script in surface.scripts:
            source, sink = _DOM_SOURCE.search(script), _DOM_SINK.search(script)
            if source and sink:
                self.finding("low", "Possible DOM XSS",
                             f"An inline script reads {source.group(0)} and uses {sink.group(0).strip()}. If the value flows from one to the other unescaped, the URL can inject script.",
                             "Trace the data flow; use textContent or a sanitiser instead of HTML sinks.", url)
        return (f"{len(links)} same-site link(s), {len(surface.forms)} form(s), {new_params} new parameter(s), "
                f"{len(surface.scripts)} inline script(s), {surface.handlers} inline handler(s)")

    def _do_classify_inputs(self, url: str) -> str:
        inputs = self.pages[url]["inputs"]
        scores = score(self.pipeline, [value for _, value in inputs])
        flagged = [(float(s), where, value) for (where, value), s in zip(inputs, scores) if s > self.threshold]
        for s, where, value in sorted(flagged, reverse=True)[:10]:
            evidence = ", ".join(repr(piece) for piece in explain(self.pipeline, value)) or "no single fragment stands out"
            self.finding("medium", "Payload-like value already in the page",
                         f"{where}: {value[:160]!r} scored {s:.3f} (threshold {self.threshold:.3f}). Strongest fragments: {evidence}.",
                         "Find out who placed this value. If the page reflects it, it is an attack link.", url)
        if flagged:  # a decision: what the detector flagged is examined before anything else
            for task in self.agenda:
                if task["action"] in ("probe", "fetch") and any(value[:40] in task["args"].get("url", "") or
                                                                normalize(value)[:40] in normalize(task["args"].get("url", "")) for _, _, value in flagged):
                    task["priority"] += 20
                    task["why"] += "; moved up because the detector flagged a value in this link"
        return f"{len(inputs)} value(s) scored, {len(flagged)} flagged (highest {float(max(scores)):.3f})"

    def _do_probe(self, url: str, param: str) -> str:
        if self.probes >= self.max_probes:
            if not any("Probe budget" in note for note in self.notes):
                self.notes.append(f"Probe budget of {self.max_probes} reached; some parameters were not tested.")
            return "skipped: probe budget used"
        marker = "zqx" + secrets.token_hex(4)
        self.probes += 1
        response, elsewhere = self._get(with_param(url, param, marker + PROBE_CHARS))
        if elsewhere:
            return f"redirects to another site ({elsewhere}); not followed"
        hits = reflections(response.text[:MAX_BODY], marker)
        if not hits:
            return "the marker did not come back: not reflected"
        dangerous = [hit for hit in hits if hit["exploitable"]]
        content_type = response.headers.get("Content-Type", "").lower()
        endpoint = url.split("?")[0]
        if dangerous and "html" in content_type:
            hit = dangerous[0]
            raw = f"with {' '.join(hit['raw'])} unencoded" if hit["raw"] else "inside a position where plain text is enough"
            self.finding("high", "Reflected XSS likely",
                         f'Parameter "{param}" of {endpoint} is written into {hit["context"]} {raw}. Returned as: ...{hit["snippet"]}...',
                         "Encode output for its context (HTML, attribute, JavaScript or URL) and add a strict Content-Security-Policy. "
                         "Confirm in a browser; this check sends no working payload.", endpoint)
            return f"reflected {len(hits)} time(s); exploitable in {hit['context']}"
        if dangerous:
            self.finding("info", "Unencoded reflection in a non-HTML response",
                         f'Parameter "{param}" of {endpoint} is echoed unencoded, but the response type is {content_type or "unknown"}, which browsers do not render as HTML.',
                         "Keep the Content-Type exact and send X-Content-Type-Options: nosniff.", endpoint)
            return "reflected unencoded, but not in an HTML response"
        return f"reflected {len(hits)} time(s) in {hits[0]['context']}, special characters encoded: handled safely"

    # ---- triaging a file --------------------------------------------------------

    def _do_read_file(self, path: str) -> str:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[:MAX_LINES]
        requests_found = 0
        for number, line in enumerate(lines, 1):
            if line.strip():
                request = _REQUEST.search(line)
                requests_found += bool(request)
                self.entries.append((number, request.group(1) if request else line))
        if not self.entries:
            return "the file has no content"
        self.add(90, "classify_lines", "every entry gets a detector score; only the suspicious ones need a closer look")
        return f"{len(self.entries):,} entries" + (f", {requests_found:,} of them access-log requests (request target extracted)" if requests_found else "")

    def _do_classify_lines(self) -> str:
        scores = score(self.pipeline, [text for _, text in self.entries])
        self.hits = sorted(((float(s), number, text) for (number, text), s in zip(self.entries, scores) if s > self.threshold), reverse=True)
        borderline = [(number, text) for (number, text), s in zip(self.entries, scores) if 0.25 * self.threshold < s <= self.threshold]
        if borderline:
            self.add(80, "recheck_borderline", f"{len(borderline)} entries scored just under the threshold; a payload can be diluted by a long harmless line, "
                                               "so each value is scored on its own", entries=borderline[:500])
        if self.hits:
            self.add(70, "explain_hits", "a score alone is not evidence; show which fragments made the strongest hits suspicious")
        return f"{len(self.hits)} of {len(self.entries):,} entries flagged, {len(borderline)} borderline"

    def _do_recheck_borderline(self, entries: list[tuple[int, str]]) -> str:
        parts = [(number, value) for number, text in entries for value in re.split(r"[?&;=\s]+", text) if len(value) >= 8]
        if not parts:
            return "no separate values to score"
        scores = score(self.pipeline, [value for _, value in parts])
        found: dict[int, tuple[float, str]] = {}
        for (number, value), s in zip(parts, scores):
            if s > self.threshold and s > found.get(number, (0, ""))[0]:
                found[number] = (float(s), value)
        self.hits = sorted([*self.hits, *((s, number, value) for number, (s, value) in found.items())], reverse=True)
        if found and not any(task["action"] == "explain_hits" for task in self.agenda):
            self.add(70, "explain_hits", "the re-check turned up hits that need evidence")
        return f"{len(found)} more entr{'y' if len(found) == 1 else 'ies'} flagged once values were scored separately"

    def _do_explain_hits(self) -> str:
        families: dict[str, int] = {}
        for s, number, text in self.hits:
            family = next((name for name, pattern in _FAMILIES if pattern.search(normalize(text))), "other")
            families[family] = families.get(family, 0) + 1
        for s, number, text in self.hits[:10]:
            evidence = ", ".join(repr(piece) for piece in explain(self.pipeline, text)) or "no single fragment stands out"
            self.finding("medium", f"Line {number} looks like XSS", f"{text[:200]!r} scored {s:.3f} (threshold {self.threshold:.3f}). Strongest fragments: {evidence}.",
                         "Check whether the request succeeded and whether the response reflected the value.")
        summary = ", ".join(f"{count} {name}" for name, count in sorted(families.items(), key=lambda item: -item[1]))
        self.finding("info", "Flagged entries by kind", f"{len(self.hits)} flagged: {summary}." + (" Only the 10 highest-scoring are listed." if len(self.hits) > 10 else ""))
        return f"explained the top {min(10, len(self.hits))}; kinds: {summary}"

    # ---- judging one text -------------------------------------------------------

    def _do_classify_text(self, text: str) -> str:
        s = float(score(self.pipeline, [text])[0])
        if s > self.threshold:
            evidence = ", ".join(repr(piece) for piece in explain(self.pipeline, text)) or "no single fragment stands out"
            self.finding("medium", "The text looks like XSS", f"{text[:200]!r} scored {s:.3f} (threshold {self.threshold:.3f}). Strongest fragments: {evidence}.",
                         "Encode it for the context it is written into; never insert it into a page as it is.")
        elif s > 0.25 * self.threshold:
            self.add(80, "recheck_borderline", "the score is just under the threshold; score each value on its own", entries=[(1, text)])
        return f"score {s:.3f} against threshold {self.threshold:.3f}: {'flagged' if s > self.threshold else 'not flagged'}"

    # ---- report -----------------------------------------------------------------

    def _report(self, goal: str) -> dict:
        findings = sorted(self.findings.values(), key=lambda f: SEVERITIES.index(f["severity"]))
        not_covered = []
        if self.kind == "audit":
            untested = len(self.endpoints)
            if not self.active and untested:
                not_covered.append(f"{untested} parameter(s) were found but not tested for reflection. Re-run with --active on a site you are authorised to test.")
            if self.post_forms:
                not_covered.append(f"{self.post_forms} POST form(s) were not submitted, so their fields are untested.")
            not_covered.append("Stored XSS, pages behind a login without --cookie, and content built by JavaScript are outside this audit.")
        elif self.kind == "classify" and not findings:
            not_covered.append("A low score does not prove the text is safe. Keep output encoding in place.")
        return {"goal": goal, "kind": self.kind, "mode": "active" if self.active else "passive", "steps": self.trace, "findings": findings,
                "counts": {severity: sum(f["severity"] == severity for f in findings) for severity in SEVERITIES},
                "stats": {"pages": len(self.pages), "requests": self.requests_sent, "probes": self.probes, "entries": len(self.entries)},
                "notes": self.notes, "not_covered": not_covered}


def format_report(report: dict, steps: bool = True) -> str:
    lines = [f"Goal: {report['goal']}"]
    stats = report["stats"]
    if report["kind"] == "audit":
        lines.append(f"Mode: {report['mode']} | {stats['pages']} page(s) fetched, {stats['requests']} request(s) sent, {stats['probes']} parameter probe(s)")
    if steps:
        lines.append(f"\nSteps taken ({len(report['steps'])})")
    for step in report["steps"] if steps else []:
        lines.append(f"  {step['step']:>2}. {step['action']} {step['target'][:80]}".rstrip())
        lines.append(f"      why: {step['why']}")
        lines.append(f"      saw: {step['observation']}")
    lines.append(f"\nFindings ({len(report['findings'])})" if report["findings"] else "\nNo findings.")
    for finding in report["findings"]:
        repeat = f" (on {finding['count']} pages)" if finding["count"] > 1 else ""
        lines.append(f"  [{finding['severity'].upper()}] {finding['title']}{repeat}")
        lines.append(f"      {finding['evidence']}")
        if finding["fix"]:
            lines.append(f"      Fix: {finding['fix']}")
    for title, items in (("Notes", report["notes"]), ("Not covered", report["not_covered"])):
        if items:
            lines.append(f"\n{title}")
            lines.extend(f"  - {item}" for item in items)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("goal", help='what to do, e.g. "audit https://your-site.example/"')
    parser.add_argument("--active", action="store_true", help="also send harmless marker values to query parameters (authorised sites only)")
    parser.add_argument("--model", type=Path, help="trusted XSS model pickle (default: the neural model if trained, else the linear one)")
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--max-probes", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--cookie", help='Cookie header for a logged-in audit, e.g. "sessionid=abc123"')
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()
    path = args.model or default_model_path()
    if not path.exists():
        print(f"no model at {path}: run  python src/train_xss.py  or  python src/train_xss_neural.py  first")
        return 2
    if args.active:
        print("Active mode sends test requests. Only use it on sites you own or are authorised to test.\n")
    agent = Agent(load_model(path), active=args.active, max_pages=args.max_pages, max_probes=args.max_probes,
                  delay=args.delay, timeout=args.timeout, cookie=args.cookie, log=print)
    report = agent.run(args.goal)
    print("\n" + format_report(report, steps=False))  # the steps were printed as they happened
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.json}")
    if any(step["observation"].startswith("failed") for step in report["steps"][:1]):
        return 2
    return 1 if any(report["counts"][severity] for severity in ("high", "medium", "low")) else 0


if __name__ == "__main__":
    sys.exit(main())
