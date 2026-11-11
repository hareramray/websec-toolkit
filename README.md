# Web security checks: XSS, CSRF, clickjacking

Two tools, because the three problems are different in kind:

| Problem | Tool | Why |
|---|---|---|
| XSS | Trained model (`train_xss.py`, `predict_xss.py`) | Payloads vary endlessly, so a model that learns their patterns helps. |
| CSRF, clickjacking | Rule-based scanner (`scan_site.py`) | These are missing headers, tokens and cookie flags. Each has an exact definition, so rules are more reliable than a model. |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## XSS model

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py     # download and clean the data
.\.venv\Scripts\python.exe src\train_xss.py        # train, evaluate, save models\xss_model.joblib
.\.venv\Scripts\python.exe src\predict_xss.py "<img src=x onerror=alert(1)>" "<b>hello</b>"
```

The model is character n-gram TF-IDF with logistic regression. Input is URL-decoded
and HTML-unescaped (up to three rounds) and lowercased before scoring, and long input
is scored in overlapping windows so a payload cannot be hidden by padding.

Training data (`source` column in `data/processed/xss.csv`):

| Source | Benign | XSS | What it is |
|---|---|---|---|
| `mereani` | 28,067 | 14,372 | Mereani & Howe 2018 (MIT): attack URLs, benign URLs, JavaScript, text |
| `webattacks` | 4,830 | 9,246 | shengqin/web-attacks-long on Hugging Face (no licence stated) |
| `twin_urls` | 13,429 | 0 | The mereani attack URLs rebuilt without their payload |
| `synthetic` | 12,397 | 0 | Generated everyday HTML, URLs and short strings (`synthetic.py`) |

Results of the last run (`models/xss_metrics.json`):

| Evaluation | Attacks caught | False alarms |
|---|---|---|
| Held-out test split (12,352 samples) | 98.2% | 0.08% |
| Trained on mereani, tested on webattacks | 98.3% | 0.00% |
| Trained on webattacks, tested on mereani | 92.6% | 3.02% |
| Challenge set (`data/challenge/`, hand-written, never trained on) | 60 of 60 | 0 of 80 |

Read these with care:

- The held-out number is flattering: test rows come from the same sources as training rows.
- The challenge set guided two rounds of fixes, so only its last 20 attacks and 25 benign
  lines (added afterwards and scored once) are an untouched check.
- The benign HTML and URLs are generated, not real traffic. Before using the model as a
  filter, score a sample of your own site's normal input and count the false alarms.
- A detector does not fix XSS. The fix is escaping output and a Content-Security-Policy.

To improve the model, add lines it gets wrong to `data/challenge/`, then add more
training data of that kind in `prepare_data.py` or `synthetic.py`.

## CSRF and clickjacking scanner

```powershell
.\.venv\Scripts\python.exe src\scan_site.py https://your-site.example/account
.\.venv\Scripts\python.exe src\scan_site.py https://your-site.example/account --cookie "sessionid=abc123"
```

It sends one GET request and reads the headers, cookies and forms. Only scan sites you
own or have permission to test.

- **Clickjacking** fails when the page has neither `X-Frame-Options: DENY/SAMEORIGIN`
  nor a restrictive CSP `frame-ancestors` directive.
- **CSRF** warns when a POST form has no token field, or a cookie has `SameSite=None`
  or no `SameSite` at all. These are leads to confirm by hand: forms built by JavaScript
  and tokens sent in request headers are not visible to this check.

Exit code is 0 for no issues, 1 for issues, 2 if the page could not be fetched.
