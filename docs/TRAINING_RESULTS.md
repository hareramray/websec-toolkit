# Local training and verification

## XSS detectors

Both detectors are trained only on openly licensed data: the MIT-licensed Mereani
dataset plus generated benign samples. After excluding 7 normalised matches to the
challenge set, 68,258 rows remain, split into 47,780 training, 10,239 validation and
10,239 test rows. Commands:

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\prepare_external.py
.\.venv\Scripts\python.exe src\train_xss.py
.\.venv\Scripts\python.exe src\train_xss_neural.py --epochs 12
```

`models/xss_neural.pkl` is the default for chat, `predict_xss.py` and the agent, and
is the model published at https://huggingface.co/hareramray325/websec. The architecture
has 1,959,361 parameters: character embeddings, parallel convolutions over 1, 3, 5 and
7 characters, two stride-2 convolutions, 4 Transformer encoder blocks (width 192,
8 heads) and a pooled classification head.

**As of 2026-11-12 both the default and the published model are the one trained on the
merged agent-shaped data** (see "Agent-shaped training data" below), so the default
stays byte-for-byte identical to the Hugging Face model. The rest of this section
describes the earlier Mereani-only model it replaced, kept for comparison: trained from
random weights on a laptop RTX 5050, 12 epochs in 256 s on mains power, epoch 9 kept.

Both models were scored on the same rows, each at its own threshold tuned on
validation data for 0.1% false alarms (linear 0.649, neural 0.728).

| Measurement | Linear | Neural |
|---|---|---|
| Held-out attacks caught (of 2,155) | 2,126 (98.65%) | 2,131 (98.89%) |
| Held-out false alarms (of 8,084 benign) | 11 (0.14%) | 6 (0.07%) |
| Held-out detection at exactly 0.1% false alarms | 97.96% | 98.89% |
| Challenge attacks caught | 58 / 60 | 60 / 60 |
| Challenge benign false alarms | 0 / 80 | 0 / 80 |
| External attacks caught (SecLists, of 7,896) | 7,604 (96.3%) | 7,712 (97.7%) |
| External attacks matching no training row (of 7,460) | 7,176 (96.2%) | 7,282 (97.6%) |
| External benign false alarms (of 6,993) | 0 | 5 |

What these results do and do not show:

- **The neural model is stronger on attacks.** It misses fewer held-out attacks
  with half the false alarms, catches every challenge attack, and catches 108 more
  external payloads. The external benchmark is the most trustworthy row: nothing
  was fitted or tuned on it.
- **It is less clean on unfamiliar benign text.** Its 5 external false alarms come
  from 1,994 OpenAssistant messages: four answers containing code, and one
  Russian-language message (every non-Latin character is the same token to this
  model). None came from 4,999 Wikipedia paragraphs. Measure false alarms on your
  own traffic before blocking anything.
- **The challenge set is a regression check, not an independent test.** Its past
  mistakes shaped the training-time variation in `src/xss_augment.py`.
- A plain `<iframe src=...>`, `<script src=...>` or `<form>` in an input is scored as
  an attack. The datasets label injected markup that way, so this is intended for
  classifying inputs, not for scanning whole pages.
- On an informal spot check written separately (32 fresh benign strings, 12 fresh
  attacks), linear flagged 0 and caught 10, neural flagged 0 and caught 12. It is
  small and hand-written, so treat it as a sanity check.

Earlier versions also trained on a second public dataset that states no license.
It was removed from the project, together with every model trained on it. With it,
the neural model caught more external attacks (99.5%), but it cannot be shared as
open source.

Lessons from discarded runs, for anyone retraining:

1. One run flagged `john.doe@example.com` at 0.994. The network had learned site
   names and bare function calls as attack signals. Fixed by swapping site names
   in both classes and adding benign code and maths.
2. A 20-epoch run scored best on validation but flagged `<a href="/">Home</a>` at
   0.997 and 8 of 20 fresh benign HTML strings. Validation data cannot see this,
   because it contains no such markup. Fixed by adding plain links, navigation,
   buttons, form controls and lone words as benign training rows.
3. More epochs are not automatically better: check fresh benign inputs, not only
   the validation number. On battery the same training is about three times slower.

All details are in `models/xss_metrics.json` and `models/xss_neural_metrics.json`.

## Agent-shaped training data

`src/prepare_agent_data.py` (generators in `src/agent_data.py`) adds rows in the
three shapes the agent scores: parameter/field values, request targets from access
logs, and whole log lines. It was merged with the Mereani data and trained to
**separate** files, so the published `models/xss_neural.pkl` is left untouched:

```powershell
.\.venv\Scripts\python.exe src\prepare_agent_data.py --merge
.\.venv\Scripts\python.exe src\train_xss.py --data data\processed\xss_expanded.csv --out models\xss_expanded.pkl
.\.venv\Scripts\python.exe src\train_xss_neural.py --data data\processed\xss_expanded.csv --out models\xss_neural_expanded.pkl --baseline models\xss_expanded.pkl
```

The merged file is 96,835 rows: 37,941 generated agent rows (benign values, request
targets and log lines; other-language text; and non-XSS attacks as benign) plus the
Mereani, synthetic and twin_urls rows. The neural model trained on it, each model at
its own validation-tuned threshold, on the shared test split, challenge set and the
evaluation-only external benchmark:

| Measurement | Published neural (Mereani only) | Neural on merged data |
|---|---|---|
| Held-out detection rate | 98.89% | 99.53% |
| Held-out false-alarm rate | 0.07% | 0.01% |
| Held-out detection at exactly 0.1% false alarms | 98.89% | 99.71% |
| Challenge attacks caught | 60 / 60 | 60 / 60 |
| Challenge benign false alarms | 0 / 80 | 0 / 80 |
| External attacks caught (SecLists, of 7,896) | 7,712 (97.7%) | 7,734 (97.9%) |
| External benign false alarms (of 6,993) | 5 | 5 |

The held-out split is not the same rows as the published model's, because the merged
data adds the generated rows, so read these as "both strong" rather than a controlled
A/B. The controlled signal is below.

**Where the generated data clearly helps.** On the held-out test split, every
generated benign source had **0 false alarms**: 723 parameter values, 832 request
targets, 693 log lines, 450 other-language rows, and 333 non-XSS attacks (SQLi,
traversal, command injection, SSRF, scanner probes). The linear model on the same
data behaves the same way. These are exactly the shapes `agent.py` feeds the detector,
and they were where the first model raised false alarms.

**Where it helps less, and why.** On a separate hand-written probe of 31 deliberately
tricky strings (framework code such as `(click)="save()"`, and sentences about XSS
that contain real markup), the merged neural model flagged 10 against the published
model's 12. Scores dropped (for example `<code>print('hello')</code>` fell from 0.994
to 0.794), but several still cross the threshold. Some of those are **intended**: a
literal `<script src=...>` tag, or an embedded live `<script>alert(1)</script>`, in an
input is scored as an attack — the datasets label injected markup that way, and the
detector classifies inputs, not whole trusted pages (see the note above). Treat the
code cases as a reminder to measure on your own traffic before blocking, not as a
solved problem.

Nothing was trained on the challenge set or the external benchmark. The generated
templates were written with the first model's SecLists mistakes in view, so the
external benchmark no longer measures those families fully independently.

## Security agent

`src/agent.py` was run end to end against a local test site with seven query
parameters. In active mode it reported the three unsafe reflections (HTML text, a
quoted attribute, a script string), reported the four safely encoded parameters as
handled, visited the link whose parameter the detector flagged before the others,
and reported the missing framing, cookie and script-policy protections and one
possible DOM XSS. The automated tests assert that it never requests a logout link
or a POST form and does not follow a redirect to another site. On a ten-line access
log it flagged the four attack requests and none of the six ordinary ones. The
planner is rule-based; see the README for its limits.

## Language model

The model has 298,368 parameters, 2 causal attention blocks, 4 attention heads,
96-dimensional embeddings, and a 256-byte/marker context. All initial weights are
random. Training uses next-byte prediction; conversation training supervises only
assistant replies. No pretrained model or tokenizer was downloaded.

The 64-pair starter run is preserved as `models/chat_demo.pkl`. After 600 steps,
training loss fell to 0.107 while validation loss rose to 5.774. The saved best
checkpoint is step 100 (validation loss 2.687). Sampled replies were incoherent;
memorizing the demonstration corpus does not produce a useful general chatbot.

The public-data pipeline was also exercised successfully:

- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext): 629 articles exported
  from the upstream training split; 624 distinct documents remain after deduplication,
  split into 562 training and 62 validation documents.
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1): 1,000 reviewed
  English rank-0 reply pairs, grouped into 511 conversation trees; 902 pairs for
  training and 98 for validation.

Commands for this public-data experiment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe src\prepare_chat_data.py --source wikitext --limit 1000 --out data\raw\wikitext_train.jsonl
.\.venv\Scripts\python.exe src\train_chat.py --data data\raw\wikitext_train.jsonl --steps 1500 --eval-every 250 --out models\language_base.pkl
.\.venv\Scripts\python.exe src\prepare_chat_data.py --source oasst1 --limit 1000 --out data\raw\oasst_train.jsonl
.\.venv\Scripts\python.exe src\train_chat.py --data data\raw\oasst_train.jsonl --init models\language_base.pkl --steps 1000 --eval-every 200 --learning-rate 0.0003 --out models\chat_model.pkl
```

The parent checkpoint's hash and provenance are saved inside the continued model.
Metrics and text samples must be inspected separately: a lower next-byte loss
does not establish that replies are coherent, relevant, or factually correct.

WikiText pretraining completed 1,500 steps on CPU in about 9 minutes. Validation
loss fell from 2.476 at step 250 to 1.955 at step 1,500; the final checkpoint was
selected. Full history is in `models/language_base_metrics.json`.

OpenAssistant fine-tuning then ran 1,000 steps from that checkpoint in about 5
minutes, producing `models/chat_model.pkl`. Validation loss fell from 2.082 at
step 200 to 1.931 at step 1,000; the final checkpoint was selected and the loss
was still falling, so the run stopped before converging. Training loss (1.909)
stayed close to validation loss, so unlike the starter run there is no sign of
memorization. The recorded parent hash matches `models/language_base.pkl`. The
two validation losses come from different datasets (WikiText prose, OpenAssistant
replies) and cannot be compared with each other. Full history is in
`models/chat_metrics.json`.

Sampled replies at temperature 0.7 are still not usable. The model produces
English-like letter patterns, spacing, some real short words, and occasional
list formatting, but most longer words are invented and no reply addressed its
prompt:

| Prompt | Start of reply |
|---|---|
| Hello, how are you? | Where scriptate a project that the intent with is specient the from the to the persent... |
| What is cross-site scripting? | The varierstain doculations, designal follocess and to contenting and the Some providuator... |
| How do I make a cup of tea? | The mell chare subring the difficient with such as and resultion. 1. Is and Jung definaing... |

A loss near 1.93 per byte is far above what coherent text needs. This is expected
for 298,368 parameters trained for 2,500 steps on a few megabytes; the security
tools in `src/chat.py` remain the only reliable part of the assistant.

## Verification

All 32 automated tests passed (`python -m unittest discover -s tests`), including
pickle round trips, causal attention, assistant-only target masks, real weight
updates, checkpoint continuation, conversation split isolation, challenge exclusion,
scanner integration, chat follow-ups, the neural detector (shared loader, padding
invariance, a real training run, and resuming a checkpoint's weights to continue
training), the agent (active and passive audits of a local test site, off-site
redirects, log triage, the `/agent` chat command) and the agent-shaped data generators
(benign shapes never carry injectable markup, `injected` flags real payloads but not
look-alikes, payload shaping preserves the payload, and the builder yields clean rows
of both classes). PyTorch is
now `2.11.0+cu128`; the language-model results above were produced with the CPU
build and were not re-run. The custom CSV importer and interactive tools-only chat
were also run end to end. The chat scanner was exercised against a local HTTP server with known
headers, a cookie, and a token-bearing form. `pip check` reported no broken requirements.

The public downloads and models live in Git-ignored directories. The source code,
starter corpus, tests, and instructions are available in the working tree.
