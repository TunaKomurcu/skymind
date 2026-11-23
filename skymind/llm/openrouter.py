"""OpenRouter istemcisi: sert bütçe tavanı, defter, disk önbelleği, 429 yeniden deneme.

Tasarım ilkeleri
* Tavan (varsayılan 4 USD) anahtarın kendi limitinin altındadır; aşılacaksa istek **gönderilmez**.
* Her istekten önce en kötü durum maliyeti (prompt tahmini + max_tokens) hesaplanıp kalan bütçeyle
  karşılaştırılır; gerçek maliyet yanıttaki ``usage.cost`` ile deftere yazılır.
* ``:free`` modeller (fiyat 0) bütçeden düşmez ama günlük istek kotasına tabidir.
* Aynı (model, mesajlar, parametreler) isteği diskten döner; tekrar ücretlenmez.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_BUDGET_USD = 4.0
CHARS_PER_TOKEN = 3.0  # Türkçe için tutucu (gerçekte ~3.5-4); fazla tahmin = güvenli taraf


class BudgetExceeded(RuntimeError):
    """Bir istek bütçe tavanını aşacağı için gönderilmedi."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    cost_usd: float
    cached: bool


def load_dotenv(path: Path = Path(".env")) -> None:
    """Çok basit .env yükleyici (mevcut ortam değişkenlerini ezmez)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Ledger:
    """Harcamayı diskte tutar (jsonl); süreçler arası toplam harcama kaybolmaz."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line)["cost_usd"])
        return total

    def record(self, model: str, cost_usd: float) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "model": model, "cost_usd": cost_usd}) + "\n")


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        budget_usd: float | None = None,
        state_dir: Path = Path("outputs/openrouter"),
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.budget = (
            budget_usd
            if budget_usd is not None
            else float(os.environ.get("SKYMIND_BUDGET_USD", DEFAULT_BUDGET_USD))
        )
        self.ledger = Ledger(state_dir / "ledger.jsonl")
        self.cache_dir = state_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
            transport=transport,
        )
        self._sleep = sleep
        self._prices: dict[str, tuple[float, float]] | None = None

    # -- fiyatlar / bütçe -------------------------------------------------------------------
    def prices(self) -> dict[str, tuple[float, float]]:
        """model -> (USD/prompt-token, USD/completion-token)."""
        if self._prices is None:
            data = self._http.get("/models").raise_for_status().json()["data"]
            self._prices = {
                m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
                for m in data
            }
        return self._prices

    def remaining(self) -> float:
        return self.budget - self.ledger.spent()

    def worst_case_cost(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> float:
        p_prompt, p_completion = self.prices().get(model, (float("inf"), float("inf")))
        prompt_tokens = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN + 50
        return prompt_tokens * p_prompt + max_tokens * p_completion

    # -- istek ------------------------------------------------------------------------------
    def _key(self, model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> Path:
        blob = json.dumps([model, messages, params], sort_keys=True, ensure_ascii=False)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_retries: int = 5,
        **extra: Any,
    ) -> Completion:
        params = {"max_tokens": max_tokens, "temperature": temperature, **extra}
        cache_path = self._key(model, messages, params)
        if cache_path.exists():
            hit = json.loads(cache_path.read_text(encoding="utf-8"))
            return Completion(hit["text"], model, 0.0, cached=True)

        worst = self.worst_case_cost(model, messages, max_tokens)
        if worst > self.remaining():
            raise BudgetExceeded(
                f"{model}: en kötü durum {worst:.4f} USD > kalan {self.remaining():.4f} USD "
                f"(tavan {self.budget:.2f} USD). İstek gönderilmedi."
            )

        body = {"model": model, "messages": messages, "usage": {"include": True}, **params}
        for attempt in range(max_retries):
            r = self._http.post("/chat/completions", json=body)
            if r.status_code in (429, 502, 503):
                self._sleep(min(2**attempt * 3, 60))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(f"OpenRouter hatası: {data['error']}")
            cost = float(data.get("usage", {}).get("cost") or 0.0)
            self.ledger.record(model, cost)
            text = data["choices"][0]["message"]["content"] or ""
            cache_path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            return Completion(text, model, cost, cached=False)
        raise RuntimeError(f"{model}: {max_retries} denemeden sonra hâlâ 429/5xx")
