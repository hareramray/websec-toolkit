"""Generated benign samples that the public datasets lack.

The datasets contain almost no harmless HTML and only a narrow set of harmless URLs,
so a model trained on them alone flags `<b>hello</b>` and ordinary search links as
attacks. These generators fill that gap with everyday HTML, URLs and short strings
that contain angle brackets, quotes and ampersands for innocent reasons.

Nothing here is built from the datasets, so it can be used in every training run.
"""
from __future__ import annotations

import random
import re

WORDS = """time year people way day thing world life hand part child eye woman place work week case point
government company number group problem fact music garden kitchen travel photo recipe review order account
profile settings search product category summer winter coffee book movie phone laptop camera ticket hotel
flight school teacher student doctor market price offer sale news sports weather health science history
art design color blue green red black white small large new old great good best free easy quick simple
happy sunny quiet fresh local daily weekly monthly online mobile digital modern classic family friend
team project report meeting holiday weekend morning evening dinner lunch breakfast city country river
mountain beach forest street house room table chair window door bike car train bus road map guide
the a of and to in is for on with this that from your our more about please thanks welcome today""".split()

TLDS = ["com", "com", "com", "org", "net", "io", "co.uk", "edu", "gov", "in", "de", "fr", "app", "dev", "info", "co"]
SUBDOMAINS = ["www", "www", "www", "", "", "blog", "shop", "docs", "api", "m", "news", "support", "cdn"]
PARAMS = """q id page sort category lang ref utm_source utm_medium utm_campaign search query type view limit
offset date from to user item product tag year hl s p order filter tab session v name city format""".split()
PAGE_ENDINGS = ["", "", "", "", ".html", ".php", ".aspx", ".jsp", ".htm", "/"]
FIXED_VALUES = ["true", "false", "asc", "desc", "en", "en-US", "price_asc", "all", "1", "0", "json", "newest"]

INLINE_TAGS = """b i em strong u span code small mark sub sup abbr cite q s del ins kbd samp var time label
font big strike tt dfn""".split()
BLOCK_TAGS = """p div h1 h2 h3 h4 h5 h6 blockquote pre section article aside header footer main nav figure
figcaption details summary address center dd dt caption""".split()
STYLES = [
    "color:red", "color: #336699", "font-weight:bold", "text-align:center", "margin:0 auto", "font-size:14px",
    "background-color:#f5f5f5; padding:8px", "display:none", "width:100%", "border:1px solid #ccc", "float:left",
    "text-decoration:underline", "line-height:1.5", "max-width:600px",
]
IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "webp", "svg"]
VERBS = ["is", "was", "isn't", "wasn't"]


def phrase(rng: random.Random, low: int = 1, high: int = 6) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))


def sentence(rng: random.Random) -> str:
    text = phrase(rng, 3, 10)
    if rng.random() < 0.5:
        text = text.capitalize()
    return text + rng.choice([".", ".", "!", "?", "", ""])


def value(rng: random.Random) -> str:
    """A harmless query-string value."""
    kind = rng.randrange(9)
    if kind == 0:
        return str(rng.randint(1, 99999))
    if kind == 1:
        return "+".join(rng.choice(WORDS) for _ in range(rng.randint(2, 3)))
    if kind == 2:
        return "-".join(rng.choice(WORDS) for _ in range(2))
    if kind == 3:
        return f"{rng.randint(2015, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind == 4:
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.choice([8, 16, 24, 32])))
    if kind == 5:
        return rng.choice(FIXED_VALUES)
    if kind == 6:
        return "%20".join(rng.choice(WORDS) for _ in range(2))
    return rng.choice(WORDS)


def _slug(rng: random.Random) -> str:
    kind = rng.randrange(5)
    if kind == 0:
        return str(rng.randint(1, 99999))
    if kind == 1:
        return "-".join(rng.choice(WORDS) for _ in range(rng.randint(2, 4)))
    if kind == 2:
        return rng.choice(WORDS) + "_" + rng.choice(WORDS)
    return rng.choice(WORDS)


def _host(rng: random.Random) -> str:
    if rng.random() < 0.05:
        local = rng.choice(["localhost", "127.0.0.1", f"192.168.1.{rng.randint(2, 254)}"])
        return f"{local}:{rng.choice([3000, 5000, 8000, 8080])}"
    name = rng.choice(WORDS) + rng.choice(["", "", rng.choice(WORDS), str(rng.randint(1, 99))])
    return ".".join(part for part in (rng.choice(SUBDOMAINS), name, rng.choice(TLDS)) if part)


def _email(rng: random.Random) -> str:
    user = rng.choice(WORDS) + rng.choice(["", ".", "_"]) + rng.choice(WORDS + ["", str(rng.randint(1, 999))])
    return f"{user.strip('._')}@{rng.choice(WORDS)}.{rng.choice(TLDS)}"


def url(rng: random.Random) -> str:
    path = "".join("/" + _slug(rng) for _ in range(rng.randint(0, 3)))
    path += rng.choice(PAGE_ENDINGS) if path else rng.choice(["", "/"])
    count = rng.choice([0, 0, 1, 1, 2, 3, 4])
    query = "&".join(f"{rng.choice(PARAMS)}={value(rng)}" for _ in range(count))
    out = f"{rng.choice(['http', 'https', 'https'])}://{_host(rng)}{path}"
    if query:
        out += "?" + query
    if rng.random() < 0.1:
        out += "#" + rng.choice(WORDS)
    return out


def _relative_path(rng: random.Random, endings: list[str]) -> str:
    start = rng.choice(["/", "/", "", "../", "./"])
    return start + "/".join(_slug(rng) for _ in range(rng.randint(1, 3))) + rng.choice(endings)


def _attr(rng: random.Random, name: str, val: str) -> str:
    roll = rng.random()
    if roll < 0.15 and not re.search(r"[\s=<>'\"]", val):
        return f"{name}={val}"  # unquoted, as hand-written HTML often is
    quote = "'" if roll > 0.85 else '"'
    return f"{name}={quote}{val}{quote}"


def _common_attrs(rng: random.Random) -> str:
    attrs = []
    if rng.random() < 0.3:
        attrs.append(_attr(rng, "class", " ".join("-".join(rng.sample(WORDS, rng.randint(1, 2))) for _ in range(rng.randint(1, 2)))))
    if rng.random() < 0.12:
        attrs.append(_attr(rng, "id", rng.choice(WORDS) + str(rng.randint(1, 99))))
    if rng.random() < 0.12:
        attrs.append(_attr(rng, "style", rng.choice(STYLES)))
    if rng.random() < 0.06:
        attrs.append(_attr(rng, "title", phrase(rng, 1, 3)))
    return " " + " ".join(attrs) if attrs else ""


def _link(rng: random.Random) -> str:
    target = rng.choice(
        [url(rng), url(rng), _relative_path(rng, PAGE_ENDINGS), "#" + rng.choice(WORDS), "mailto:" + _email(rng),
         f"tel:+{rng.randint(1, 99)}{rng.randint(1000000000, 9999999999)}"]
    )
    attrs = [_attr(rng, "href", target)]
    if rng.random() < 0.25:
        attrs.append(_attr(rng, "target", "_blank"))
    if rng.random() < 0.2:
        attrs.append(_attr(rng, "rel", rng.choice(["noopener", "noopener noreferrer", "nofollow"])))
    rng.shuffle(attrs)
    return f"<a {' '.join(attrs)}{_common_attrs(rng)}>{phrase(rng, 1, 4)}</a>"


def _image(rng: random.Random) -> str:
    ending = ["." + rng.choice(IMAGE_TYPES)]
    source = rng.choice([_relative_path(rng, ending), f"https://{_host(rng)}/{_relative_path(rng, ending).lstrip('./')}"])
    attrs = [_attr(rng, "src", source)]
    if rng.random() < 0.8:
        attrs.append(_attr(rng, "alt", phrase(rng, 1, 4)))
    if rng.random() < 0.4:
        attrs += [_attr(rng, "width", str(rng.choice([32, 64, 120, 300, 640]))), _attr(rng, "height", str(rng.choice([32, 64, 90, 200, 480])))]
    if rng.random() < 0.15:
        attrs.append(_attr(rng, "loading", "lazy"))
    return f"<img {' '.join(attrs)}{_common_attrs(rng)}{rng.choice(['>', '>', ' />', '/>'])}"


def _media(rng: random.Random) -> str:
    tag, ending = rng.choice([("video", ".mp4"), ("video", ".webm"), ("audio", ".mp3"), ("audio", ".ogg")])
    source = _relative_path(rng, [ending])
    if rng.random() < 0.5:
        return f"<{tag} controls {_attr(rng, 'src', source)}></{tag}>"
    return f'<{tag} controls{_common_attrs(rng)}><source {_attr(rng, "src", source)} type="{tag}/{ending[1:]}"></{tag}>'


def _list(rng: random.Random) -> str:
    tag = rng.choice(["ul", "ul", "ol"])
    items = "".join(f"<li>{phrase(rng, 1, 5)}</li>" for _ in range(rng.randint(2, 4)))
    return f"<{tag}{_common_attrs(rng)}>{items}</{tag}>"


def _table(rng: random.Random) -> str:
    columns = rng.randint(2, 3)
    head = "".join(f"<th>{rng.choice(WORDS)}</th>" for _ in range(columns))
    rows = "".join(
        "<tr>" + "".join(f"<td>{rng.choice([rng.choice(WORDS), str(rng.randint(1, 999))])}</td>" for _ in range(columns)) + "</tr>"
        for _ in range(rng.randint(1, 2))
    )
    return f"<table{_common_attrs(rng)}><tr>{head}</tr>{rows}</table>"


def _element(rng: random.Random, depth: int) -> str:
    kind = rng.choices(
        ["inline", "block", "link", "image", "list", "table", "void", "mixed", "heading", "media", "comment"],
        weights=[16, 20, 14, 10, 6, 4, 3, 14, 6, 3, 2],
    )[0]
    if kind == "inline":
        tag = rng.choice(INLINE_TAGS)
        return f"<{tag}{_common_attrs(rng)}>{phrase(rng, 1, 4)}</{tag}>"
    if kind == "block":
        tag = rng.choice(BLOCK_TAGS)
        inner = html_fragment(rng, depth + 1) if depth < 2 and rng.random() < 0.4 else sentence(rng)
        return f"<{tag}{_common_attrs(rng)}>{inner}</{tag}>"
    if kind == "link":
        return _link(rng)
    if kind == "image":
        return _image(rng)
    if kind == "list":
        return _list(rng)
    if kind == "table":
        return _table(rng)
    if kind == "void":
        return rng.choice(["<br>", "<br/>", "<br />", "<hr>", "<hr/>"])
    if kind == "mixed":
        tag = rng.choice(INLINE_TAGS)
        return f"{phrase(rng, 1, 5)} <{tag}>{phrase(rng, 1, 3)}</{tag}> {sentence(rng)}"
    if kind == "heading":
        level = rng.randint(1, 4)
        return f"<h{level}{_common_attrs(rng)}>{phrase(rng, 1, 4)}</h{level}><p>{sentence(rng)}</p>"
    if kind == "media":
        return _media(rng)
    return f"<!-- {phrase(rng, 1, 4)} -->"


def html_fragment(rng: random.Random, depth: int = 0) -> str:
    """A piece of ordinary, script-free HTML such as a comment, a post body or a page section."""
    body = rng.choice(["", "", " ", "\n"]).join(_element(rng, depth) for _ in range(rng.randint(1, 3)))
    if depth == 0 and rng.random() < 0.04:
        head = f"<title>{phrase(rng, 1, 4)}</title>" + rng.choice(["", '<meta charset="utf-8">', '<link rel="stylesheet" href="/css/style.css">'])
        return f"<!doctype html><html><head>{head}</head><body>{body}</body></html>"
    return body


def misc(rng: random.Random) -> str:
    """Short everyday strings, many with <, >, quotes or & used innocently."""
    a, b, c = (rng.choice(WORDS) for _ in range(3))
    n, m = rng.randint(0, 500), rng.randint(0, 500)
    options = [
        _email(rng),
        f"{phrase(rng, 1, 2)} <{_email(rng)}>",
        f"{n} {rng.choice(['<', '>', '<=', '>=', '==', '!='])} {m}",
        f"{a} {rng.choice(['<', '>', '<=', '>='])} {n} {rng.choice(['and', 'or', '&&', '||'])} {b} {rng.choice(['<', '>'])} {m}",
        f"if ({a} {rng.choice(['<', '>', '<=', '>='])} {b}) {{ return {c}; }}",
        f"for (let i = 0; i < {a}.length; i++) {{ {b} += {a}[i]; }}",
        f"{a} = {b} > {n} ? {b} : {c};",
        f"function {a}({b}, {c}) {{ return {b} + {c}; }}",
        f"const {a} = ({b}) => {b} * {n};",
        f"{rng.choice(['List', 'Map', 'Vector', 'Array', 'Optional'])}<{rng.choice(['String', 'Integer', 'int', 'T'])}> {a} = new {b}<>();",
        f"#include <{a}.h>",
        f"{rng.choice(['cat', 'echo', 'sort', 'grep ' + a])} {b}.txt > {c}.{rng.choice(['log', 'txt', 'csv'])}",
        f"{a} -> {b} -> {c}",
        f"<{phrase(rng, 1, 2)}>",
        f"{a.capitalize()} & {b.capitalize()}: \"{phrase(rng, 1, 3)}\" ({rng.randint(1950, 2026)})",
        f"{a}'s {b} {rng.choice(VERBS)} {c}",
        f"{rng.choice(['C:', 'D:'])}\\{a}\\{b}\\{c}.{rng.choice(['docx', 'pdf', 'xlsx', 'png', 'txt'])}",
        f"/{rng.choice(['home', 'usr', 'var', 'opt'])}/{a}/{b}/{c}.{rng.choice(['js', 'py', 'conf', 'log'])}",
        f"+{rng.randint(1, 99)} {rng.randint(10000, 99999)} {rng.randint(10000, 99999)}",
        f"{rng.randint(2015, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00+05:30",
        f"{rng.choice(['$', 'Rs. ', 'EUR ', 'GBP '])}{rng.randint(1, 99999):,} ({rng.randint(5, 70)}% off!)",
        f"**{a}** and [{b}]({url(rng)})",
        f'{{"{a}": "{b}", "{c}": {n}, "tags": ["{b}", "{c}"]}}',
        f"SELECT {a}, {b} FROM {c} WHERE id = {n}",
        f"{sentence(rng)} {sentence(rng)}",
        f"The {rng.choice(['onclick', 'onerror', 'onload', 'alert()', 'script', 'iframe', 'javascript', 'cookie'])} {rng.choice(['handler', 'tag', 'function', 'element', 'policy', 'tutorial'])} {rng.choice(['is', 'was'])} {a} {b}",
        f"{a}@{n}!{b}#{m}?<>",
    ]
    return rng.choice(options)


def generate(html_count: int = 6000, url_count: int = 4000, misc_count: int = 2500, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    samples = [html_fragment(rng) for _ in range(html_count)]
    samples += [url(rng) for _ in range(url_count)]
    samples += [misc(rng) for _ in range(misc_count)]
    return samples
