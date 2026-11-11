# Data and training guide

There are **two different learned models**. The XSS detector needs labelled payloads.
The text-generating Transformer needs language and conversation examples. An XSS
CSV alone cannot teach a model to have a useful conversation.

## XSS datasets

| Source | How to use it | Important limits |
|---|---|---|
| [Mereani / Howe XSS repository](https://github.com/fmereani/Cross-Site-Scripting-XSS) | Already supported by `prepare_data.py`: XSS and benign examples. | Repository lists MIT; preserve attribution and review the dataset documentation. |
| [shengqin/web-attacks-long](https://huggingface.co/datasets/shengqin/web-attacks-long) | **Deliberately not used.** Earlier versions trained on it; it was removed so that the project and its models stay open source. | No license is stated on its dataset card (checked 2026-11-12). |
| [Kaggle: Cross site scripting XSS dataset for Deep learning](https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning) | Candidate additional labelled corpus. Download it and inspect the actual column names and label meanings before importing. | Check its current license and duplicates against your existing corpus; do not assume it is an independent test source. |
| [SecLists](https://github.com/danielmiessler/SecLists/tree/master/Fuzzing/XSS) | **Used for evaluation only.** `prepare_external.py` downloads nine payload lists into `data/processed/xss_external.csv`, and both trainers report results on it. | MIT licence. Do not import these lists for training, or the external benchmark stops being independent. Payload collection, not a balanced labelled set: a few lines are inert fragments, and many payloads also appear in the training sources, so lines matching no project row are reported separately. |
| [FuzzDB](https://github.com/fuzzdb-project/fuzzdb/tree/master/attack/xss) | Additional reviewed XSS patterns for training or a separate evaluation set. | Attack-only material needs benign examples alongside it. The project describes New BSD / CC Attribution licensing; preserve attribution and check file-specific terms. |
| Your application's reviewed traffic | Add realistic benign text, allowed HTML, URLs, and confirmed attack cases. | Remove secrets and personal data. Keep a later time period or separate application for final testing. |

Do not train this binary detector on a generic "attack" label from a mixed HTTP or
network dataset. SQLi, CSRF, and clickjacking are not interchangeable with XSS.
The existing CSRF/clickjacking checks remain rules over the response and forms.

Canonical CSV format:

```csv
text,label,source
hello world,0,my_app
<b>welcome</b>,0,my_app
<script>alert(1)</script>,1,reviewed_xss
```

`text` and `label` are required. `source` defaults to `custom` if absent. Training
normalizes text, removes exact normalized duplicates and conflicting labels, and
rejects invalid labels. This is exact deduplication, not payload-family deduplication.
At least 10 distinct examples of each class are required for splitting; that is only
a mechanical minimum, not enough for a reliable detector.

Import a downloaded CSV (adjust column names to the file you inspected):

```powershell
.\.venv\Scripts\python.exe src\import_dataset.py data\raw\extra.csv --text-column Sentence --label-column Label --source extra_xss --base data\processed\xss.csv --out data\processed\xss_expanded.csv
.\.venv\Scripts\python.exe src\train_xss.py --data data\processed\xss_expanded.csv --out models\xss_expanded.pkl
.\.venv\Scripts\python.exe src\predict_xss.py --model models\xss_expanded.pkl "<b>hello</b>"
```

For string labels, add `--label-map '{"normal":0,"XSS":1}'`. For a reviewed text
file of XSS payloads, use:

```powershell
.\.venv\Scripts\python.exe src\import_dataset.py data\raw\reviewed_xss.txt --label 1 --source reviewed_patterns --base data\processed\xss.csv --out data\processed\xss_expanded.csv
```

Keep the challenge set and final test examples out of imported training files.
Do not repeatedly tune against the challenge set and then call it untouched.
New sources appear in the per-source held-out report but do not automatically gain
independent leave-one-source-out evaluation. Split related payload families before
training if you need a stronger estimate of generalization. Only import data whose
license allows your use: the exporter in `src/export_hf.py` refuses to build a model
card for a training source it has no license entry for.

## Language and conversation datasets

| Source | Purpose | License / preparation |
|---|---|---|
| `data/chat/security_dialogues.jsonl` | 64 project-authored demonstration pairs to exercise training and chat. | Tiny starter corpus, not a security benchmark or adequate general-language corpus. |
| [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | Learn general next-byte language patterns from articles before conversation training. | Card lists CC BY-SA 3.0 and GFDL. Preserve attribution and meet applicable terms. This is prose, not dialogue. |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Human conversation data for learning user/assistant turns. | Apache 2.0 on the card. The helper selects reviewed, English, rank-0 replies and groups pairs by conversation tree. |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Larger synthetic instruction/conversation collection for a later experiment. | Review the terms of the selected subset. Export complete alternating user/assistant messages; tool/system messages need a deliberate format conversion. No automatic importer is included. |
| Your reviewed security Q&A | Teach your application's terminology and accurate explanations. | Use content you can train on. Put paraphrases of one question in the same `group`. |

Source cards checked on 2026-11-11. No pretrained model weights are required by any
of the commands below. Training on synthetic conversations is still random-weight
training, although those conversations may themselves have been generated by another model.

Supported JSONL records (one JSON object per line):

```jsonl
{"user":"What is XSS?","assistant":"Cross-site scripting is unsafe execution of untrusted browser content.","group":"xss_definition"}
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"What can you check?"},{"role":"assistant","content":"Use /check for an XSS classification."}],"group":"intro"}
{"text":"A complete article or document for language pretraining.","group":"document_001"}
```

The loader deduplicates identical records and splits **groups before making training
windows**. About 10% of groups are reserved for validation. Without an explicit
`group`, each unique record is its own group; paraphrases are not detected
automatically. Dialogue loss is applied to assistant replies only; plain-text loss
is applied to all next-byte targets. Reserve an additional unseen test set yourself.

Run the small demonstration:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-chat.txt
.\.venv\Scripts\python.exe src\train_chat.py --steps 600
.\.venv\Scripts\python.exe src\chat.py
```

Try a larger data pipeline (optional downloads; these are starting settings, not
a recipe for production-quality conversation):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe src\prepare_chat_data.py --source wikitext --limit 1000 --out data\raw\wikitext_train.jsonl
.\.venv\Scripts\python.exe src\train_chat.py --data data\raw\wikitext_train.jsonl --steps 5000 --out models\language_base.pkl
.\.venv\Scripts\python.exe src\prepare_chat_data.py --source oasst1 --limit 5000 --out data\raw\oasst_train.jsonl
.\.venv\Scripts\python.exe src\train_chat.py --data data\raw\oasst_train.jsonl --init models\language_base.pkl --steps 5000 --out models\chat_model.pkl
```

The helper downloads the upstream **train** split only. `--init` continues your
own checkpoint's weights and architecture with a fresh optimizer; without it every
run starts from random weights. For larger experiments, set architecture flags on
the first run, e.g. `--embed-dim 256 --layers 4 --heads 4 --context-length 512`.
These increase compute and memory demand; this implementation keeps prepared data
in memory and is intended for small experiments. A compatible CUDA PyTorch install
can use `--device cuda`; CPU works with the defaults. Install the appropriate build
using [PyTorch's official instructions](https://pytorch.org/get-started/locally/).

Inspect `models/chat_metrics.json`, compare unseen prompts, and review generated
answers. Validation next-byte loss is not an accuracy score. More steps on 64 pairs
mainly encourage memorization. Byte tokenization, a 256-token context, and about
298K parameters sharply limit the demonstration model. Useful broad conversation
requires much larger and more diverse data, suitable model capacity, and compute.

Both model files are real Python pickles. Only load trusted files you trained
yourself: unpickling can execute code. The neural pickle contains NumPy arrays and
configuration, and `load_chat_model()` rebuilds the PyTorch module on the chosen device.
