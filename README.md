# Local security assistant with models trained from scratch

Chat locally, classify possible XSS, let an agent audit a site or triage a log, and
inspect a page for CSRF and clickjacking indicators. No hosted model, API key,
pretrained weights, or pretrained tokenizer is used.

| Part | What it does | Saved artifact |
|---|---|---|
| Neural XSS classifier | Character convolutions + Transformer encoder, trained on a GPU from random weights. The strongest detector here; used by default once trained | `models/xss_neural.pkl` |
| Linear XSS classifier | Character TF-IDF + logistic regression. Fast, needs no PyTorch, and is the baseline the neural model is measured against | `models/xss_model.pkl` |
| Security agent | Given a goal, plans its own steps: fetches pages, maps inputs, runs the detector, probes parameters, follows up on what it finds, and reports each step and finding | `src/agent.py` |
| Text generator | Causal Transformer initialized with random weights; learns to predict UTF-8 bytes | `models/chat_model.pkl` |
| Assistant controller | Routes explicit requests, runs security tools and the agent, remembers the last classification, and generates other replies | `src/chat.py` |
| Page scanner | Checks response headers, cookies and HTML forms with rules | No trained weights |

The assistant is a **hybrid with rule-based tool routing**. For `/check`, it
normalizes input, scores it, explains influential character fragments, and selects
a next-step recommendation based on the verdict. Follow-up `/why` uses the saved
result. Ordinary conversation uses the Transformer. Generated text is never
interpreted as a command.

The **agent** chooses its own next step from what it has observed, but its planner
is rules plus the trained detector, not a language model. It pursues three kinds
of goal (audit a URL, triage a file, judge a text) and cannot take on open-ended
instructions. See [Security agent](#security-agent).

The bundled **64 conversation pairs are a training demonstration**. The starter
model can generate text, but sampled replies are not yet coherent. Its validation
loss worsens as it memorizes the small corpus. Useful open-ended chat requires much
more language/conversation data and compute; training on XSS strings alone will
not provide that. Security tool results are returned directly from code and are
labelled separately from experimental generated text.

## Setup and train

Run in PowerShell from the project root (Python 3.10+):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-chat.txt

# XSS: download, clean, train and evaluate the linear baseline
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\train_xss.py

# XSS: evaluation-only benchmark, then the neural model (GPU recommended, see below)
.\.venv\Scripts\python.exe src\prepare_external.py
.\.venv\Scripts\python.exe src\train_xss_neural.py

# Language model: randomly initialized Transformer, no downloaded weights
.\.venv\Scripts\python.exe src\train_chat.py --steps 600

# Start interactive chat
.\.venv\Scripts\python.exe src\chat.py
```

Training creates real Python pickle files and JSON metrics under `models/`.
These generated artifacts are ignored by Git. Only load trusted pickles;
unpickling can execute code.

For the linear classifier, agent and scanner without PyTorch, install
`requirements.txt` and run `src/chat.py --tools-only`.

### GPU

`requirements-chat.txt` installs the CPU build of PyTorch. For an NVIDIA GPU,
replace it with a CUDA build that your driver supports. This workspace uses an
RTX 5050 with driver 577, which supports CUDA 12.x:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`train_xss_neural.py` and `train_chat.py --device cuda` then use the GPU. Plug the
laptop in first: on battery Windows slows the GPU to a fraction of its speed.
The neural model also runs on a CPU, more slowly.

## Chat examples

```text
You> /check <img src=x onerror=alert(1)>
You> why?
You> check: <b>hello</b>
You> /scan https://your-authorized-site.example/account
You> /agent audit https://your-authorized-site.example/
You> /agent triage logs/access.log
You> /metrics
You> /datasets
You> /train
You> What is XSS?
You> /reset
You> /quit
```

`/train` prints the training commands. It does not silently retrain weights.
Chat history stays in memory and is not added to training data. `/reset` clears
history and the last analysis; restart chat after retraining to load new weights.
The neural model sees at most its configured context length (256 **bytes/markers**
by default), so long conversations lose earlier context.

One-shot use and custom models:

```powershell
.\.venv\Scripts\python.exe src\chat.py --tools-only --message "/check <b>hello</b>"
.\.venv\Scripts\python.exe src\chat.py --message "hello" --temperature 0 --max-new-tokens 160
.\.venv\Scripts\python.exe src\chat.py --model models\xss_expanded.pkl --chat-model models\chat_model.pkl
```

Without `--model`, chat, `predict_xss.py` and the agent use `models/xss_neural.pkl`
when it exists and PyTorch is installed, and `models/xss_model.pkl` otherwise.

## Publish the neural detector on Hugging Face

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hf.txt
.\.venv\Scripts\hf.exe auth login          # paste a token with write access
.\.venv\Scripts\python.exe src\export_hf.py --repo-id your-name/xss-detector            # build dist\hf only
.\.venv\Scripts\python.exe src\export_hf.py --repo-id your-name/xss-detector --push     # upload as a private repo
```

The package holds `model.safetensors` (no pickle), `config.json`, a standalone
`xss_detector.py` that needs only torch, numpy and safetensors, `metrics.json`
with local paths removed, and a model card generated from the saved metrics.
Add `--public` to publish openly and `--license` to change the license id. The
exporter refuses to build a card for a training source it has no license entry for.
The published model: https://huggingface.co/hareramray325/websec

## Security agent

```powershell
.\.venv\Scripts\python.exe src\agent.py "audit https://your-authorized-site.example/"
.\.venv\Scripts\python.exe src\agent.py "audit https://your-authorized-site.example/" --active
.\.venv\Scripts\python.exe src\agent.py "triage logs\access.log"
.\.venv\Scripts\python.exe src\agent.py "is this xss: <img src=x onerror=alert(1)>"
```

The agent keeps an agenda of possible next steps, each with a priority and a
reason. It runs the most useful one, reads the result, and lets the result add or
re-order steps. It prints every step with why it was chosen and what it saw.

| Goal | What the agent does |
|---|---|
| Audit a URL | Fetches the page, checks framing, CSRF, cookie and script-policy headers, maps links, forms and parameters, scores values already in the page with the detector, visits same-site links (those the detector flags first), and flags inline scripts that connect a URL source to an HTML sink |
| Audit with `--active` | Also sends each query parameter one marker value followed by `<'">`, then reports where it came back and whether those characters were encoded for that context |
| Triage a file | Scores every line (extracting the request from access-log lines), re-scores the separate values of lines just under the threshold, and explains the strongest hits |
| Judge a text | Scores it, explains it, and re-checks its parts when the score is borderline |

Limits you should know:

- **Only audit sites you own or are authorised to test.** Active mode sends test
  requests. It never sends a working payload, never submits POST forms, skips
  logout and delete links, stays on the starting site, and waits between requests.
- "Reflected XSS likely" means the marker came back unencoded in a position where
  script can be injected. Confirm it in a browser before treating it as proven.
- Stored XSS, pages behind a login (unless you pass `--cookie`), and content built
  by JavaScript are not covered. Budgets: `--max-pages 8`, `--max-probes 12`.
- The planner is rule-based. There is no language model behind it, so it does not
  understand goals outside the table above.

## Add datasets and train larger models

See [the dataset and training guide](docs/DATASETS.md) for source links, licenses,
CSV/JSONL formats, import commands, evaluation guidance, and larger training runs.
It includes additional XSS sources (Kaggle, SecLists, FuzzDB) and separate language
sources (WikiText, OpenAssistant, SmolTalk).
See [the local training report](docs/TRAINING_RESULTS.md) for the commands and
measured results from this workspace.

Import a labelled CSV and retrain the classifier:

```powershell
.\.venv\Scripts\python.exe src\import_dataset.py data\raw\extra.csv --text-column Sentence --label-column Label --source extra_xss --base data\processed\xss.csv --out data\processed\xss_expanded.csv
.\.venv\Scripts\python.exe src\train_xss.py --data data\processed\xss_expanded.csv --out models\xss_expanded.pkl
```

Generate rows in the three shapes the agent scores (parameter/field values, request
targets, log lines) and merge them in, to cut false alarms on code, security talk and
other-language text without weakening attack detection:

```powershell
.\.venv\Scripts\python.exe src\prepare_agent_data.py --merge
.\.venv\Scripts\python.exe src\train_xss_neural.py --data data\processed\xss_expanded.csv --out models\xss_neural_expanded.pkl --baseline models\xss_expanded.pkl
```

Attack rows reuse existing payloads (Mereani and `xss_augment.synthetic_attack`)
placed into those shapes; the benign side generates ordinary text, plus non-XSS
attacks (SQLi, traversal, SSRF, scanner probes) as benign so the XSS detector does
not fire on them. See [the dataset guide](docs/DATASETS.md#agent-shaped-rows-for-the-detector).

### Continue (resume) training the same neural model

`train_xss_neural.py --resume` continues your own saved checkpoint instead of starting
from random weights: same architecture, a fresh optimizer, and the threshold and
calibration re-tuned on validation data afterwards. Use it to fine-tune on added data
with a smaller learning rate and fewer epochs:

```powershell
.\.venv\Scripts\python.exe src\train_xss_neural.py --resume models\xss_neural_expanded.pkl --data data\processed\xss_expanded.csv --out models\xss_neural_resumed.pkl --epochs 4 --learning-rate 2e-4
```

The saved metrics record the parent checkpoint's path, SHA-256 and initialization, so a
continued model's lineage is traceable. `--width/--layers/--heads` are ignored when
resuming (the checkpoint's architecture is reused). The linear baseline has no resume
step: it is cheap to retrain in full. To keep the published `models/xss_neural.pkl`,
resume to a different `--out`.

Train your language model on your own JSONL conversations:

```powershell
.\.venv\Scripts\python.exe src\train_chat.py --data data\raw\my_dialogues.jsonl --steps 5000
```

Use `--init models\language_base.pkl` to continue your own earlier weights with a
fresh optimizer. Without `--init`, training always starts from random weights.
The defaults use 2 Transformer blocks, 96-dimensional embeddings, 4 attention
heads, and 298,368 parameters. The trainer keeps the checkpoint with lowest
validation next-byte loss. This loss is not factual accuracy.

## XSS evaluation

```powershell
.\.venv\Scripts\python.exe src\predict_xss.py "<img src=x onerror=alert(1)>" "<b>hello</b>"
.\.venv\Scripts\python.exe src\predict_xss.py --file inputs.txt
```

Text is URL-decoded, HTML-unescaped, lowercased, and scored in overlapping windows.
Training normalizes and deduplicates rows, removes conflicting labels, and excludes
exact normalized matches to `data/challenge/*.txt` before splitting. The threshold
is tuned on validation data. The saved classifier is the evaluated training-split
model, not a subsequent refit on test rows.

Read current results in `models/xss_metrics.json`: held-out detection rate, false
alarms, per-source results, challenge mistakes and the external benchmark. Custom
output names get corresponding metrics files, such as `xss_expanded_metrics.json`.
`--exclude-source NAME` drops one source before splitting, for example data whose
license does not allow your use.

The neural model uses the same rows, split, threshold rule and evaluations, and
writes `models/xss_neural_metrics.json`, which also holds the linear model's
results on the same rows. Its training set is varied every epoch: proof-of-concept
calls are swapped for real actions, attacks are placed inside benign text and URLs,
and template-built attacks and benign look-alikes are added (`src/xss_augment.py`).
Only training rows are varied.

`prepare_external.py` builds an **evaluation-only benchmark**: SecLists payload
lists as attacks, WikiText and OpenAssistant text as benign. Nothing is fitted or
tuned on it. Both trainers report it, with the attack lines that match no project
row counted separately. It is the fairest comparison here, with limits: the lists
overlap in style with the training sources, and the benign text is mostly prose.

The earlier challenge set had 14 exact overlaps with the prepared training data;
the trainer now removes those. Challenge examples also guided earlier project
changes, so this set is a regression check rather than an untouched benchmark.
Exact deduplication does not prevent related payload families from crossing splits.
The synthetic benign examples do not replace testing on realistic normal traffic.
A detector score does not prove that a payload is exploitable or safe.

## Page scanner

```powershell
.\.venv\Scripts\python.exe src\scan_site.py https://your-authorized-site.example/account
.\.venv\Scripts\python.exe src\scan_site.py https://your-authorized-site.example/account --cookie "sessionid=abc123"
```

Only scan sites you own or have permission to test. The scanner uses GET (following
redirects), examines framing policies, token fields, and SameSite cookies, and does
not submit forms or execute JavaScript. Findings are indicators to verify with the
application's actual behavior; JavaScript forms and header-based CSRF tokens may
not be visible. Standalone exit codes: 0 no issues, 1 issues, 2 fetch failure.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover causal attention, assistant-only training targets, real weight updates,
pickle round trips, conversation split boundaries, XSS data cleaning and challenge
exclusion, chat follow-ups, and scanner integration with a mocked response. For the
neural detector they cover the pickle round trip through the shared loader, padding
invariance, a real training run and the per-epoch variation. For the agent they run
audits against a local test site with one unsafe and one safe reflection, check that
passive mode sends no probe, and check log triage and the `/agent` chat command.
The Hugging Face export is checked to reproduce the project model's scores exactly.

## License and data

The code is released under the [MIT License](LICENSE). Training uses only openly
licensed or generated data:

| Data | Used for | License |
|---|---|---|
| [Mereani & Howe XSS dataset](https://github.com/fmereani/Cross-Site-Scripting-XSS) | XSS detector training | MIT |
| Generated samples (`src/synthetic.py`, `src/xss_augment.py`, `src/agent_data.py`) | XSS detector training | Part of this project |
| [SecLists XSS lists](https://github.com/danielmiessler/SecLists/tree/master/Fuzzing/XSS) | XSS evaluation only | MIT |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | Language model; XSS false-alarm evaluation | CC BY-SA 3.0 / GFDL |
| [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Language model; XSS false-alarm evaluation | Apache 2.0 |

Datasets are downloaded by the preparation scripts and are not stored in this
repository. The hand-written challenge set in `data/challenge/` and the starter
dialogues in `data/chat/` were written for this project.
