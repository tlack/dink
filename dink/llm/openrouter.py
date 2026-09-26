"""A careful OpenRouter chat client for data generation.

Safety rails, because generation jobs loop:
- max_requests per client (per run) and max_per_day across runs (counted on disk) are hard
  limits: exceeding either raises Budget, it never silently continues.
- every response is cached on disk by (model, messages, params), so re-running a job only
  pays for what is new.
- requests are paced (min_interval) and retried with backoff on 429/5xx, honoring
  Retry-After; other errors raise immediately.

Credentials come from the environment or ./.env: OPENROUTER_KEY, OPENROUTER_MODEL.
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
                 cache_dir="data/llm_cache", timeout=180):
        load_env()
        self.key = os.environ.get("OPENROUTER_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL")
        if not self.key or not self.model:
            raise RuntimeError("set OPENROUTER_KEY and OPENROUTER_MODEL (environment or .env)")
        self.max_requests, self.max_per_day = max_requests, max_per_day
        self.min_interval, self.timeout = min_interval, timeout
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.cache / "usage.json"
        self.requests = 0
        self.cache_hits = 0
        self._last = 0.0

    # ---- accounting
    def _usage(self):
        try:
            u = json.loads(self.usage_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            u = {}
        return u

    def _count(self):
        u = self._usage()
        today = date.today().isoformat()
        u[today] = u.get(today, 0) + 1
        self.usage_file.write_text(json.dumps(u, indent=1))

    def _check_budget(self):
        if self.requests >= self.max_requests:
            raise Budget(f"run limit reached: {self.requests} requests (max_requests={self.max_requests})")
        today = self._usage().get(date.today().isoformat(), 0)
        if today >= self.max_per_day:
            raise Budget(f"daily limit reached: {today} requests today (max_per_day={self.max_per_day})")

    # ---- the call
    def chat(self, messages, temperature=0.9, max_tokens=800, **extra):
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **extra}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        hit = self.cache / f"{key}.json"
        if hit.exists():
            self.cache_hits += 1
            return json.loads(hit.read_text())["text"]
        self._check_budget()
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "X-Title": "dink-model data generation"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 429 or e.code >= 500:
                    delay = float(e.headers.get("Retry-After") or min(60, 4 * 2 ** attempt))
                    print(f"  openrouter {e.code}, retrying in {delay:.0f}s: {detail[:120]}")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"openrouter HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                delay = min(60, 4 * 2 ** attempt)
                print(f"  network error ({e}), retrying in {delay}s")
                time.sleep(delay)
        else:
            raise RuntimeError("openrouter: gave up after 6 attempts")
        self.requests += 1
        self._count()
        if "error" in resp:
            raise RuntimeError(f"openrouter error: {json.dumps(resp['error'])[:300]}")
        text = resp["choices"][0]["message"].get("content") or ""
        hit.write_text(json.dumps({"body": body, "text": text, "usage": resp.get("usage")}))
        return text

    def json_list(self, prompt, system=None, **kw):
        """Ask for a JSON array; returns the parsed list, or [] if the reply isn't one."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.chat(msgs, **kw).strip()
        if "```" in text:
            text = text.split("```")[1].removeprefix("json").strip()
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            out = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []
        return out if isinstance(out, list) else []
