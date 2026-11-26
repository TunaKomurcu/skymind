# Depo rehberi: hangi dosya ne işe yarıyor

Aşağıdaki tüm komutlar depo kökünden çalıştırılır. Python tarafı `uv` ile yönetilir; ağır kütüphaneler "extra" olarak
ayrılmıştır: `dev` (test/lint), `serve` (FastAPI), `train` (torch/transformers/peft/trl), `optimize` (ONNX).

## Kurulum

```bash
uv sync --extra dev --extra serve                 # test, lint, servisler (torch gerekmez)
uv sync --extra dev --extra serve --extra train   # + eğitim/değerlendirme (CUDA'lı torch indirir, büyük)
uv run pytest                                     # testler
```

## Dizin haritası

| Yol | İçerik |
|---|---|
| `skymind/` | Python paketi (aşağıda modül modül) |
| `cpp/` | C++ çıkarım servisi + GoogleTest (`cpp/README.md`: derleme, uçlar) |
| `deploy/` | Konteyner, compose, Kubernetes, Prometheus, Grafana (`deploy/README.md`) |
| `data/gold/` | Altın küme (300) ve NER elle seti (40) + inceleme kayıtları (`data/gold/README.md` veri kartı) |
| `data/synth/` | Gemma-3 paraphrase çıktıları (`paraphrases.jsonl`) |
| `docs/` | Rapor, sonuçlar, teknolojiler, veri kartı, yol haritası, ADR |
| `scripts/` | Yardımcı betikler (GGUF üretimi, demo, C++ ortam/fixture) |
| `tests/` | pytest testleri |
| `.github/workflows/ci.yml` | CI: lint + tip + test + C++ derleme/test |

Git'e girmeyenler (`.gitignore`): `outputs/` (modeller, adaptörler, GGUF, ham sonuçlar, MLflow), `data/processed/` (yeniden üretilir), `.env`.

## `skymind/` modülleri

| Modül | Ne yapar | Ana komut |
|---|---|---|
| `schemas/plan.py` | Plan, netleştirme ve ret şeması (Pydantic); tüm doğrulamanın tek kaynağı | (kütüphane) |
| `data/generator.py` | Şablon/gramer motoru: plan → Türkçe komut | `python -m skymind.data.generate` |
| `data/paraphrase.py`, `run_paraphrase.py` | Gemma-3 ile yeniden yazım + kural tabanlı doğrulama | `python -m skymind.data.run_paraphrase` (OpenRouter anahtarı gerekir) |
| `data/build.py` | Sızıntısız train/val/test bölme + altın aday seçimi | `python -m skymind.data.build` |
| `data/repair.py` | Bozuk paraphrase satırlarını onarma (bir kerelik) | |
| `data/review.py`, `review_web.py`, `review_web.html` | Altın küme insanı doğrulama (terminal ve web) | `python -m skymind.data.review --web` |
| `data/ner_review.py`, `ner_review_web.html` | NER elle seti doğrulama + `--apply` | `python -m skymind.data.ner_review --web` |
| `eval/metrics.py` | JSON geçerlilik, alan doğruluğu, uydurma ve gereksiz soru oranları | (kütüphane) |
| `planner/` | `baselines.py` (zero-shot), `sft.py` (QLoRA SFT), `dpo.py`, `preferences.py`, `evaluate.py`, `infer.py`, `prompt.py` | `python -m skymind.planner.sft --name sft-v1 ...` |
| `nlp/` | `train.py` (BERTurk niyet/NER), `baselines.py` (TF-IDF, kural), `ner_data.py` (zayıf etiket), `compress.py`/`compress_run.py` (damıtma/budama) | `python -m skymind.nlp.train --task intent --name intent-berturk` |
| `rag/` | `corpus.py` (kurgusal kılavuz), `retrieval.py` (BM25, e5, RRF), `answer.py` (atıf), `evaluate.py`, `generate.py` (LLM yanıtlayıcı) | `python -m skymind.rag.evaluate --seeds 1,2,3,4,5,6,7,8` |
| `aviation/` | `metar.py`, `notam.py` (ayrıştırıcı + Türkçe özet), `faithfulness.py` (sayı denetimi), `generate.py` (sentetik rapor), `llm_summary.py` (LLM karşılaştırması) | `python -m skymind.aviation.llm_summary` |
| `optimize/` | `merge.py` (adaptör birleştirme), `eval_gguf.py`, `eval_onnx.py` | `python -m skymind.optimize.eval_gguf --gguf ...` |
| `serve/` | `app.py` (Python servis), `engines.py` (HF ve llama-cpp-python motorları), `gateway.py` (A/B), `loadtest.py`, `parity.py` | `python -m skymind.serve.app --model ...` |
| `llm/openrouter.py` | OpenRouter istemcisi: sert bütçe tavanı, defter, önbellek | (kütüphane) |
| `mlops/tracking.py` | MLflow `log_run` yardımcısı | (kütüphane) |
| `demo.py` | Uçtan uca demo (planlayıcı + METAR/NOTAM + atıflı RAG) | `python -m skymind.demo` |

## `cpp/` (C++ çıkarım servisi)

| Dosya | Ne yapar |
|---|---|
| `include/skymind/engine.hpp`, `llama_engine.hpp`, `src/llama_engine.cpp` | Motor arayüzü ve llama.cpp uygulaması (greedy, KV sıfırlama) |
| `validate.hpp`, `src/validate.cpp` | Plan şeması doğrulayıcı (Pydantic ile eşlik testli) |
| `prompt.hpp`, `src/prompt.cpp` | ChatML istem kurucu |
| `metrics.hpp`, `src/metrics.cpp` | Prometheus metin biçimi metrikleri |
| `service.hpp`, `src/service.cpp` | HTTP'den bağımsız servis mantığı (400/500 yolları, sayaçlar) |
| `src/server_main.cpp` | HTTP sunucu (cpp-httplib), komut satırı |
| `tests/` | GoogleTest: doğrulayıcı (263 vaka), istem/metrik, servis |

Uçlar: `POST /v1/plan {"command": "..."}` · `GET /healthz` · `GET /metrics`.

## `deploy/`
`compose.yaml` (tek komutla yığın) · `containers/Containerfile.cpp` ve `Containerfile.gateway` · `k8s/*.yaml` + `kustomization.yaml`
(namespace, PVC, iki arka uç, gateway+HPA, izleme) · `prometheus/` (yapılandırma ve 4 alarm) · `grafana/` (veri kaynağı, 10 panelli pano).

## `docs/`
| Belge | İçerik |
|---|---|
| `PROJECT_REPORT.md` | Proje raporu: özet, hedef eşlemesi, bulgular, hatalar/dersler, sınırlamalar |
| `RESULTS.md` | Tüm ölçüm tabloları, uyarılar ve **tekrar üretme komutları** |
| `TECH_STACK.md` | Kullanılan teknolojiler ve ne işe yaradıkları; kullanılmayanlar |
| `DATA_CARD.md` | Komut→plan veri setinin kartı |
| `GOLD_REVIEW.md` | Altın küme inceleme rehberi (araç kullanımı, karar ölçütleri) |
| `ROADMAP.md` | Fazlar ve sprint durumu |
| `adr/` | Mimari karar kayıtları |

## Sık kullanılan akışlar

```bash
# Veri → eğitim → değerlendirme (GPU gerekir; adım adım komutlar: docs/RESULTS.md "Tekrar üretme")
uv run --extra train python -m skymind.planner.sft --name sft-v1 --subset 5000 --epochs 1
uv run --extra train python -m skymind.planner.evaluate --name sft-v1 --adapter outputs/sft-v1/adapter --limit 400

# Demo (Windows): C++ sunucu + 5 komut + METAR/NOTAM + atıflı RAG
.\scripts\demo.ps1

# Tüm yığın (Podman/Docker): gateway :8000, Prometheus :9090, Grafana :3000
podman compose -f deploy/compose.yaml up --build
```
