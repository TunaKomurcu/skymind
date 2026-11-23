# SkyMind — Türkçe İHA Operatör Asistanı

Operatörün Türkçe komutunu doğrulanmış bir **JSON görev planına** çeviren, eksik/belirsiz komutta uydurmak
yerine **netleştirme sorusu** soran bir NLP/LLM araştırma prototipi. Yalnızca açık kaynak ve sentetik veri
kullanır; gerçek operasyonel sistem iddiası yoktur.

> Durum: Faz 0 (iskele). Plan ve ilerleme: [`docs/ROADMAP.md`](docs/ROADMAP.md).
> Kararlar: [`docs/adr/`](docs/adr/).

## Hızlı başlangıç

```bash
uv sync --extra dev
uv run pytest
```

## Yapı

```
skymind/    Python paketi (schemas, data, eval, nlp, planner, rag, ...)
tests/      pytest testleri
docs/       yol haritası, ADR'ler
```

Mimari diyagram, sonuç tabloları ve tekrar üretme adımları fazlar ilerledikçe buraya eklenecek.
