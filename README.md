# SkyMind — Türkçe İHA Operatör Asistanı

Operatörün Türkçe komutunu doğrulanmış bir **JSON görev planına** çeviren, eksik/belirsiz komutta uydurmak
yerine **netleştirme sorusu** soran bir NLP/LLM araştırma prototipi. Yalnızca açık kaynak ve sentetik veri
kullanır; gerçek operasyonel sistem iddiası yoktur.

> Durum: Faz 0–7 büyük ölçüde tamam (veri, SFT, DPO, niyet/NER, RAG, METAR/NOTAM, nicemleme, damıtma/budama, C++ servis, A/B). Kalan: gerçek küme denemesi, altın set gözden geçirme, final demo. Plan ve ilerleme: [`docs/ROADMAP.md`](docs/ROADMAP.md).
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

## Diğer sonuçlar (özet; ayrıntı ve uyarılar `docs/RESULTS.md`'de)

* **C++ vs Python servis** (aynı GPU, batch 1): tek istekte p50 258 ms (llama.cpp Q5_K_M) vs 1243 ms (HF bf16),
  ~4.8×; fark dil + nicemleme + yığın karışımı, saf dil farkı ayrıştırılmadı.
* **A/B** (SFT vs DPO v2, 400 istek): geçerlilik 0.9725 vs 0.9776, p = 0.74, anlamlı fark yok.
* **Niyet** (BERTurk): doğruluk 0.986, makro-F1 0.982 (TF-IDF: 0.962 / 0.851). **NER**: zayıf-etiket testi
  döngüsel (kurallar da 0.994); elle yazılmış zor sette BERTurk 0.867 vs kurallar 0.853.
* **RAG** (kurgusal kılavuz): hibrit R@5 0.852 (BM25 0.819/0.822). Üretken yanıtlayıcıda doğru parça gelince
  doğruluk 0.94, cevapsız sorularda vazgeçme 24/24; atıf geçerliliği 1.0 ama desteklenme ölçülmedi.
* **Sıkıştırma** (niyet sınıflandırıcı): katman azaltan öğrenciler CPU'da 2.0×/3.1×/5.2× hızlı, doğruluk bu görevde
  tavana yakın (fark gürültü düzeyinde); kafa budama ≤1.3×, yapısal olmayan budama hız kazandırmaz.
* **METAR/NOTAM özeti:** kural tabanlı özetleyici sayısal olarak sadık (yapı gereği); LLM (gemma-3 27B) 80 METAR'ın
  %35'inde, 48 NOTAM'ın %31'inde kaynakta olmayan sayı üretti (NOTAM tarih kodunu sık yanlış okuyor); 4B daha kötü.
