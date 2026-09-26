"""A careful OpenRouter chat client for data generation.

Safety rails, because generation jobs loop:
- max_requests per client (per run) and max_per_day across runs (counted on disk) are hard
  limits on HTTP attempts, retries included: exceeding either raises Budget.
- max_consecutive_errors failed calls in a row raise Budget too (a dead provider or a bad
  prompt stops the run instead of burning the budget).
- every response is cached on disk by (model, messages, params), so re-running a job only
  pays for what is new.
- requests are paced (min_interval) and retried with backoff on 429/5xx, whether the error
  arrives as an HTTP status or inside the response body (providers do both), honoring
  Retry-After; other errors raise immediately.
- every call is logged: data/llm/live.log (one readable line per call, for `tail -f`) and
  data/llm/calls.jsonl (full prompt and reply). Cache hits are logged too.

Credentials come from the environment or ./.env: OPENROUTER_KEY, OPENROUTER_MODEL.
Client.groq() pins gpt-oss-120b to the Groq provider (fast; paid, so a dollar cap applies:
max_spend_per_day, counted on disk from each reply's reported cost).

Prompt caching: providers cache the longest repeated *prefix* of a request, and cached input
tokens cost about half. So callers put everything that stays the same (instructions, rules,
output format) in the system message, first, and only the part that changes (the task, the
persona) in the final user message. The live log shows cached/total input tokens per call.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"


class Budget(RuntimeError):
    pass


def load_env(path=".env"):
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Client:
    def __init__(self, model=None, max_requests=50, max_per_day=500, min_interval=3.5,
                 cache_dir="data/llm_cache", timeout=180, max_consecutive_errors=5,
                 log_dir="data/llm", provider=None, max_spend_per_day=5.0):
        load_env()
        self.key = os.environ.get("OPENROUTER_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL")
        self.provider = provider                # e.g. "Groq": route only there, no fallbacks
        self.max_spend_per_day = max_spend_per_day
        self.tokens = {"in": 0, "cached": 0, "out": 0, "cost": 0.0}
        if not self.key or not self.model:
            raise RuntimeError("set OPENROUTER_KEY and OPENROUTER_MODEL (environment or .env)")
        self.max_requests, self.max_per_day = max_requests, max_per_day
        self.min_interval, self.timeout = min_interval, timeout
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.cache / "usage.json"
        self.requests = 0
        self.cache_hits = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.errors_in_a_row = 0
        self._last = 0.0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def groq(cls, **kw):
        """gpt-oss-120b served by Groq: ~350 tokens/s, $0.15/M in ($0.075 cached), $0.60/M out."""
        kw.setdefault("min_interval", 0.3)
        kw.setdefault("max_per_day", 5000)
        return cls(model="openai/gpt-oss-120b", provider="Groq", **kw)

    # ---- logging
    def _log(self, status, body, text="", secs=0.0, usage=None):
        prompt = body["messages"][-1]["content"]
        u = usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        tok = f" in {cached}/{u['prompt_tokens']} cached, out {u.get('completion_tokens')}" if u else ""
        with open(self.log_dir / "calls.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status,
                                "model": body["model"], "secs": round(secs, 1), "usage": usage,
                                "prompt": prompt, "reply": text}) + "\n")
        one = lambda x, n: " ".join(str(x).split())[:n]
        with open(self.log_dir / "live.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {status:7s} {secs:5.1f}s "
                    f"req {self.requests:3d}{tok}  prompt: {one(prompt, 60)}"
                    f"  ->  {one(text, 160)}\n")

    # ---- accounting
    def _usage(self):
        try:
            u = json.loads(self.usage_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            u = {}
        return u

    def _count(self, cost=0.0):
        """One request (cost=0) or the dollars a finished request cost."""
        u = self._usage()
        today = date.today().isoformat()
        if cost:
            u["$" + today] = round(u.get("$" + today, 0.0) + cost, 6)
        else:
            u[today] = u.get(today, 0) + 1
        self.usage_file.write_text(json.dumps(u, indent=1))

    def _check_budget(self):
        if self.requests >= self.max_requests:
            raise Budget(f"run limit reached: {self.requests} requests (max_requests={self.max_requests})")
        today = self._usage().get(date.today().isoformat(), 0)
        if today >= self.max_per_day:
            raise Budget(f"daily limit reached: {today} requests today (max_per_day={self.max_per_day})")
        spent = self._usage().get("$" + date.today().isoformat(), 0.0)
        if spent >= self.max_spend_per_day:
            raise Budget(f"daily spend limit reached: ${spent:.2f} (max_spend_per_day={self.max_spend_per_day})")

    # ---- the call
    def _cache_path(self, body):
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        return self.cache / f"{key}.json"

    def body(self, messages, temperature=0.9, max_tokens=800, reasoning_effort="low", **extra):
        b = {"model": self.model, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, **extra}
        if reasoning_effort:            # reasoning models: less hidden thinking, faster, cheaper
            b["reasoning"] = {"effort": reasoning_effort}
        if self.provider:
            b["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return b

    def forget(self, messages, **kw):
        """Drop a cached reply (e.g. one that turned out unusable) so the next run asks again."""
        self._cache_path(self.body(messages, **kw)).unlink(missing_ok=True)

    def chat(self, messages, **kw):
        body = self.body(messages, **kw)
        hit = self._cache_path(body)
        if hit.exists():
            self.cache_hits += 1
            text = json.loads(hit.read_text())["text"]
            self._log("cached", body, text)
            return text
        t0 = time.time()
        for attempt in range(6):
            self._check_budget()
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "X-Title": "dink-model data generation"})
            self.requests += 1
            self._count()
            retry_after, detail = None, ""
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                err = resp.get("error")
                if not err:
                    self.errors_in_a_row = 0
                    break
                code = int(err.get("code") or 0) if str(err.get("code") or "").isdigit() else 0
                detail = json.dumps(err)[:300]
                if not (code == 429 or code >= 500):
                    self._fail(body, detail, time.time() - t0)
                    raise RuntimeError(f"openrouter error: {detail}")
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if not (e.code == 429 or e.code >= 500):
                    self._fail(body, detail, time.time() - t0)
                    raise RuntimeError(f"openrouter HTTP {e.code}: {detail}") from None
                retry_after = e.headers.get("Retry-After")
            except (urllib.error.URLError, TimeoutError) as e:
                detail = f"network error: {e}"
            delay = float(retry_after or min(60, 5 * 2 ** attempt))
            self._log("retry", body, detail, time.time() - t0)
            print(f"  openrouter busy/unavailable, retrying in {delay:.0f}s: {detail[:140]}")
            time.sleep(delay)
        else:
            self._fail(body, "gave up after 6 attempts", time.time() - t0)
            raise RuntimeError("openrouter: gave up after 6 attempts")
        text = resp["choices"][0]["message"].get("content") or ""
        u = resp.get("usage") or {}
        cost = float(u.get("cost") or 0.0)
        self.tokens["in"] += u.get("prompt_tokens", 0) or 0
        self.tokens["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        self.tokens["out"] += u.get("completion_tokens", 0) or 0
        self.tokens["cost"] += cost
        if cost:
            self._count(cost)
        hit.write_text(json.dumps({"body": body, "text": text, "usage": resp.get("usage")}))
        self._log("ok", body, text, time.time() - t0, resp.get("usage"))
        return text

    def _fail(self, body, detail, secs):
        self._log("FAILED", body, detail, secs)
        self.errors_in_a_row += 1
        if self.errors_in_a_row >= self.max_consecutive_errors:
            raise Budget(f"{self.errors_in_a_row} failed calls in a row; stopping")

    def summary(self):
        t = self.tokens
        share = t["cached"] / t["in"] if t["in"] else 0.0
        return (f"{self.requests} requests, {self.cache_hits} from disk cache; tokens in {t['in']} "
                f"({share:.0%} provider-cached), out {t['out']}; ${t['cost']:.4f}")

    def json_list(self, prompt, system=None, **kw):
        """Ask for a JSON array; returns the parsed list, or [] if the reply isn't one."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.chat(msgs, **kw).strip()
        if "```" in text:
            text = text.split("```")[1].removeprefix("json").strip()
        start, end = text.find("["), text.rfind("]")
        out = None
        if 0 <= start < end:
            try:
                out = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        if not isinstance(out, list):
            self.forget(msgs, **kw)             # never keep an unusable reply in the cache
            return []
        return out
