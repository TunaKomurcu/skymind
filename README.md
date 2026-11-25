# SkyMind — Türkçe İHA Operatör Asistanı

Operatörün Türkçe komutunu doğrulanmış bir **JSON görev planına** çeviren, eksik/belirsiz komutta uydurmak
yerine **netleştirme sorusu** soran bir NLP/LLM araştırma prototipi. Yalnızca açık kaynak ve sentetik veri
kullanır; gerçek operasyonel sistem iddiası yoktur.

> Durum: Faz 0–4 tamam (veri, SFT, DPO), Faz 6 nicemleme tamam. Plan ve ilerleme: [`docs/ROADMAP.md`](docs/ROADMAP.md).
> Kararlar: [`docs/adr/`](docs/adr/). Tüm sonuçlar ve uyarılar: [`docs/RESULTS.md`](docs/RESULTS.md).

## Planlayıcı sonuçları (400 test örneği)

| | Zero-shot | SFT (QLoRA) | DPO v2 |
|---|---|---|---|
| Geçerli JSON | 0.367 | **0.988** | 0.953 |
| Alan doğruluğu | 0.014 | 0.842 | **0.848** |
| Uydurma oranı ↓ | 0.043 | 0.016 | **0.011** |
| Gereksiz soru ↓ | 0.236 | 0.108 | **0.057** |

SFT dağıtım adayı. Saf DPO şemayı bozdu, SFT kaybı eklenen v2 yapıyı korudu ama geçerlilikte SFT'nin
gerisinde kaldı. Test seti sentetik/dağılım içi, fark küçük örneklemde: ayrıntı ve uyarılar
[`docs/RESULTS.md`](docs/RESULTS.md)'de.

## Hızlı başlangıç

```bash
uv sync --extra dev
uv run pytest
```

## Yapı

```
skymind/    Python paketi (schemas, data, eval, planner, optimize, serve, rag, aviation)
cpp/        C++ çıkarım servisi (llama.cpp) + GoogleTest, bkz. cpp/README.md
deploy/     A/B ağ geçidi, Prometheus/Grafana, Containerfile, compose, K8s, bkz. deploy/README.md
tests/      pytest testleri
docs/       yol haritası, ADR'ler, sonuçlar
```

Mimari diyagram, sonuç tabloları ve tekrar üretme adımları fazlar ilerledikçe buraya eklenecek.

## Nicemleme (SFT modeli, llama.cpp, RTX 4060)

| Varyant | Boyut | Alan doğruluğu | p50 gecikme |
|---|---|---|---|
| F16 | 2950 MB | 0.842 | 0.50 s |
| **Q5_K_M** | 1073 MB | 0.843 | 0.26 s |
| Q4_K_M | 940 MB | 0.758 | 0.21 s |
| Q3_K_M | 786 MB | 0.180 | 0.24 s |

Q5_K_M doğruluğu koruyup boyutu %64 küçültüyor; Q4'ten itibaren model gereksiz soruya kayıyor.
Ayrıntı ve uyarılar: [`docs/RESULTS.md`](docs/RESULTS.md).

CPU'da ONNX Runtime int4 (937 MB, 25 tok/s, alan doğruluğu 0.854) ile llama.cpp Q4_K_M (940 MB, 31 tok/s, 0.703)
karşılaştırıldı; ONNX int4 gereksiz soruyu azaltıyor ama daha sık uyduruyor (0.037). Ayrıntı: `docs/RESULTS.md`.
