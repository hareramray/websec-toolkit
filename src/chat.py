"""Local security assistant: deterministic tool routing plus a language model trained from scratch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

from agent import Agent
from agent import format_report as format_agent_report
from scan_site import format_report, scan
from xss_model import ROOT, default_model_path, explain, load_model, normalize, score

HELP = """Security tools (results come directly from code):
  /check <text>       classify XSS, explain evidence, suggest next action
  /why               explain the last classification
  /scan <http(s) URL> check headers, cookies and forms on an authorized site
  /agent <goal>      let the agent plan and run the steps itself, e.g.
                       /agent audit https://your-site.example/
                       /agent --active audit https://your-site.example/   (sends harmless probes)
                       /agent triage logs/access.log
  /metrics           show saved model evaluation results
  /datasets          show training data guidance and source links
  /train             show commands to train both models
  /reset             clear conversation and last result
  /quit              leave chat
Other messages go to your locally trained text-generating model.
Natural requests such as 'check: <b>hello</b>' also work."""


class SecurityAssistant:
    def __init__(self, xss_path: Path | None = None, chat_path: Path | None = None,
                 tools_only: bool = False, temperature: float = 0.7, max_new_tokens: int = 240):
        self.xss_path = xss_path or default_model_path()
        self.chat_path = chat_path or ROOT / "models" / "chat_model.pkl"
        self.tools_only = tools_only
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.bundle = None
        self.language_model = None
        self.history: list[dict] = []
        self.last_analysis: dict | None = None

    def analyze(self, text: str) -> str:
        if not text.strip():
            return "Usage: /check <text to classify>"
        if len(text) > 100_000:
            return "Please keep each input under 100,000 characters."
        if self.bundle is None:
            self.bundle = load_model(self.xss_path)
        pipeline = self.bundle["pipeline"]
        probability = float(score(pipeline, [text])[0])
        threshold = float(self.bundle["threshold"])
        flagged = probability > threshold
        self.last_analysis = {"text": text, "normalized": normalize(text), "score": probability,
                              "threshold": threshold, "flagged": flagged, "evidence": explain(pipeline, text)}
        verdict = "XSS flagged" if flagged else "Not flagged as XSS"
        evidence = ", ".join(repr(x) for x in self.last_analysis["evidence"]) or "No strong positive character fragments."
        action = ("Review the input's output context; use context-aware encoding and sanitization where needed."
                  if flagged else "Keep normal output encoding in place; a low score does not prove safety.")
        return (f"[Tools: normalize -> classify -> explain]\n{verdict}. Model score: {probability:.4f}; threshold: {threshold:.4f}.\n"
                f"Fragments increasing the score: {evidence}\n{action}")

    def respond(self, message: str) -> str:
        message = message.strip()
        lower = message.lower()
        if not message or lower in ("/help", "help"):
            return HELP
        if lower == "/reset":
            self.history.clear()
            self.last_analysis = None
            return "Conversation and last classification cleared. Model weights are unchanged."
        if lower in ("/why", "/explain", "why", "why?", "explain that", "explain the last result"):
            if self.last_analysis is None:
                return "Use /check <text> first so there is a result to explain."
            last = self.last_analysis
            return (f"The score {last['score']:.4f} was compared with threshold {last['threshold']:.4f}.\n"
                    f"Positive model evidence: {', '.join(repr(x) for x in last['evidence']) or 'none'}.\n"
                    "These are learned character patterns, not proof that code would execute in a browser.")
        if lower in ("/datasets", "datasets"):
            return (ROOT / "docs" / "DATASETS.md").read_text(encoding="utf-8")
        if lower == "/train":
            return ("Run from the project root:\n"
                    ".\\.venv\\Scripts\\python.exe src\\train_xss.py\n"
                    ".\\.venv\\Scripts\\python.exe src\\train_xss_neural.py\n"
                    ".\\.venv\\Scripts\\python.exe src\\train_chat.py --steps 600\n"
                    "All write .pkl files under models/. See docs/DATASETS.md for custom-data commands. "
                    "Restart chat after retraining to load the new weights.")
        if lower == "/metrics":
            results = {}
            for name, model_path in (("xss", self.xss_path), ("chat", self.chat_path)):
                path = model_path.with_name(model_path.stem.replace("_model", "") + "_metrics.json")
                results[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else "No metrics yet; train this model first."
            return json.dumps(results, indent=2)
        if lower == "/check":
            return "Usage: /check <text to classify>"
        if lower.startswith("/check "):
            return self.analyze(message[7:])
        request = re.match(r"^(?:check|classify|analyze|analyse)(?:\s*:\s*|\s+)(.+)$", message, re.I | re.S)
        if request:
            return self.analyze(request.group(1))
        if lower == "/scan":
            return "Usage: /scan https://your-authorized-site.example/page"
        if lower.startswith("/scan "):
            url = message[6:].strip()
            if any(character.isspace() for character in url):
                return "Supply one http(s) URL after /scan."
            return "[Tool: page scanner]\n" + format_report(scan(url))
        if lower == "/agent":
            return "Usage: /agent audit https://your-authorized-site.example/   (add --active after /agent to send harmless probes)"
        if lower.startswith("/agent "):
            goal = message[7:].strip()
            active = goal.lower().startswith("--active ")
            if self.bundle is None:
                self.bundle = load_model(self.xss_path)
            return "[Agent]\n" + format_agent_report(Agent(self.bundle, active=active).run(goal[9:].strip() if active else goal))
        if message.startswith("/"):
            return "Unknown command. Use /help for the available tools."
        if self.tools_only:
            return "Tools-only mode does not generate text. Use /help, or train chat_model.pkl and restart without --tools-only."
        from chat_model import generate, load_chat_model
        if self.language_model is None:
            self.language_model, _ = load_chat_model(self.chat_path)
        pending = [*self.history[-8:], {"role": "user", "content": message}]
        reply = generate(self.language_model, pending, self.max_new_tokens, self.temperature)
        if not reply:
            return "[Scratch model] No text generated. Try a different prompt or train on more conversation data."
        self.history = [*pending, {"role": "assistant", "content": reply}][-8:]
        return "[Scratch model; experimental text]\n" + reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="trusted XSS pickle (default: the neural model if trained, else the linear one)")
    parser.add_argument("--chat-model", type=Path, default=ROOT / "models" / "chat_model.pkl")
    parser.add_argument("--tools-only", action="store_true", help="run security tools without PyTorch or chat weights")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=240, help="maximum generated bytes")
    parser.add_argument("--message", help="answer one message and exit")
    args = parser.parse_args()
    if not args.tools_only:
        try:
            import torch
            torch.set_num_threads(4)
        except ImportError:
            parser.error("Install requirements-chat.txt, or use --tools-only.")
    assistant = SecurityAssistant(args.model, args.chat_model, args.tools_only, args.temperature, args.max_new_tokens)
    if args.message is None:
        print("Local security chat. /help lists tools; /quit exits. Generated replies are experimental.")
    while True:
        try:
            message = args.message if args.message is not None else input("\nYou> ")
            if message.strip().lower() in ("/quit", "/exit", "quit", "exit"):
                return 0
            print("\nAssistant> " + assistant.respond(message))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        except FileNotFoundError as error:
            print(f"Missing file: {error.filename}. Run src/train_xss.py and src/train_chat.py first, or use /help.")
            if args.message is not None:
                return 2
        except (OSError, ValueError, requests.RequestException) as error:
            print(f"Could not complete the request: {error}")
            if args.message is not None:
                return 2
        if args.message is not None:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
