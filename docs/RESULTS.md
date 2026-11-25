# Sonuçlar — Planlayıcı (Faz 3–4)

Model: Qwen2.5-1.5B-Instruct. Test: `data/processed/test.jsonl`'den eşit aralıklı **400** örnek
(212 plan, 123 netleştirme, 65 ret), tüm modellerde aynı küme, greedy üretim.
Metrikler katı (strict) JSON ayrıştırmasıyla; "lenient" sütunu kod bloğu vb. sarmalayıcıları ayıklar
(MLflow'da `lenient_*`).

| Metrik | Zero-shot | SFT | DPO v1 | DPO v2 |
|---|---|---|---|---|
| json_parsable | 0.848 | **1.000** | 0.945 | **1.000** |
| json_valid | 0.367 | **0.988** | 0.520 | 0.953 |
| type_accuracy | 0.147 | **0.920** | 0.468 | 0.907 |
| action_accuracy | 0.019 | 0.849 | 0.377 | **0.854** |
| field_accuracy | 0.014 | 0.842 | 0.330 | **0.848** |
| fabrication_rate ↓ | 0.043 | 0.016 | **0.011** | **0.011** |
| over_clarification_rate ↓ | 0.236 | 0.108 | **0.028** | 0.057 |
| invalid_action_rate ↓ | 0.480 | **0.013** | 0.425 | 0.048 |

Zero-shot: uzun sistem istemi; ince ayarlı modeller: kısa sistem istemi.

## Eğitim ayarları

| | SFT | DPO v1 | DPO v2 |
|---|---|---|---|
| Veri | 5000 örnek, 1 epoch | 3000 tercih çifti | 2000 tercih çifti |
| Yöntem | QLoRA (nf4), r=16, lr 2e-4 | LoRA r=16 (SFT birleştirilmiş), β=0.1, lr 2e-5 | aynı + SFT kaybı (ağırlık 1.0), lr 5e-6 |
| Etkin batch | 8×2 | 1×16 | 1×16 |

SFT: eval loss 0.0226, token doğruluğu %99.1.

## Okuma ve uyarılar

1. **SFT asıl kazanım.** Zero-shot'a göre geçerli JSON 0.37 → 0.99, alan doğruluğu 0.01 → 0.84.
2. **Saf DPO (v1) zarar verdi.** Uydurma ve gereksiz soru azaldı, ama model şemayı bozdu:
   `distance_m` yerine `distance`, eksik `kind`, var olmayan aksiyonlar (`fly`, `approach`).
   Tercih çiftleri tek bir sayıda ayrıştığı için ortak yapısal token'ların olasılığı da düşürüldü
   (likelihood displacement). Bu bir hata ayıklama bulgusu olarak tutuldu.
3. **DPO v2 (SFT kaybı + düşük lr) SFT'yi net geçmiyor.** Gereksiz soru yarıya indi (0.108 → 0.057,
   212 plandan ~23 → ~12), alan/aksiyon doğruluğu +0.5 puan; karşılığında geçerli JSON 0.988 → 0.953 ve
   geçersiz aksiyon 0.013 → 0.048. Operatör asistanında geçersiz plan, gereksiz soruya göre daha
   maliyetli olduğundan **dağıtım adayı SFT modelidir**; DPO v2 ek bir seçenek olarak raporlanır.
4. **Gürültü sınırı.** n=400; fabrication paydası 188 (netleştirme+ret), yani 0.016 ≈ 3 örnek,
   0.011 ≈ 2 örnek. SFT ile DPO v2 arasındaki fabrication farkı istatistiksel olarak anlamlı değil.
   Tek tohum, güven aralığı hesaplanmadı.
5. **Test dağılımı.** Test seti şablon + paraphrase ile üretildi (dağılım içi) ve etiketleri otomatik
   doğru. 300 aday elle gözden geçirilmedi; dağılım dışı ve insan yazımı komutlardaki performans
   ölçülmedi. Mutlak rakamlar gerçek kullanımı temsil etmez.
6. **Zero-shot sürümü.** İlk ölçüm 160 örnekte yapılmıştı (strict json_valid 0.362); eşit aralıklı alt
   küme farklı olduğu için tablo 400 örnekle yeniden çalıştırıldı.

## Tekrar üretme

```bash
uv run --extra train python -m skymind.planner.sft --name sft-v1 --subset 5000 --epochs 1
uv run --extra train python -m skymind.planner.dpo --name dpo-v2 --sft-adapter outputs/sft-v1/adapter \
    --subset 2000 --batch 1 --accum 16 --max-length 384 --lr 5e-6 --sft-weight 1.0
uv run --extra train python -m skymind.planner.evaluate --name dpo-v2 \
    --adapter outputs/dpo-v2/adapter --base-adapter outputs/sft-v1/adapter --limit 400
```

8 GB GPU'da DPO için `--batch 2` belleğe sığmadı (paylaşımlı belleğe taşma, adım başına dakikalar);
`--batch 1 --accum 16 --max-length 384` kullanıldı.

---

# Nicemleme (Faz 6a) — GGUF / llama.cpp

SFT modeli birleştirilip GGUF'a çevrildi (llama.cpp b12655), `llama-server` ile CUDA'da (RTX 4060 Laptop,
tüm katmanlar GPU'da) aynı 400 örnekte ölçüldü. İstekler ardışık, batch=1, greedy, kısa sistem istemi.
Gecikme = uçtan uca HTTP isteği (ortalama ~40 üretilen token).

| Varyant | Boyut | json_valid | type_acc | field_acc | uydurma ↓ | gereksiz soru ↓ | p50 | p95 | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| HF bf16 (referans) | ~2950 MB | 0.988 | 0.920 | 0.842 | 0.016 | 0.108 | — | — | — |
| GGUF F16 | 2950 MB | 0.988 | 0.922 | 0.842 | 0.011 | 0.108 | 0.50 s | 1.13 s | 67 |
| Q8_0 | 1570 MB | 0.988 | 0.920 | **0.846** | 0.016 | 0.108 | 0.29 s | 0.65 s | 113 |
| **Q5_K_M** | 1073 MB | 0.978 | 0.912 | 0.843 | 0.011 | 0.108 | 0.26 s | 0.53 s | 136 |
| Q4_K_M | 940 MB | 0.988 | 0.870 | 0.758 | 0.011 | 0.203 | 0.21 s | 0.47 s | 152 |
| Q3_K_M | 786 MB | 0.970 | 0.552 | 0.180 | 0.005 | 0.759 | 0.24 s | 0.36 s | 138 |

## Okuma ve uyarılar

1. **Q8_0 ve Q5_K_M doğruluğu koruyor** (alan doğruluğu 0.842 → 0.846 / 0.843; fark gürültü içinde),
   boyut %47 / %64 küçülüyor, gecikme p50 ~%42 / ~%48 düşüyor. **Dağıtım adayı: Q5_K_M.**
2. **Q4_K_M'de bozulma başlıyor:** alan doğruluğu 0.842 → 0.758, gereksiz soru 0.108 → 0.203. Model
   emin olmadığında netleştirme sorusuna kayıyor. Boyut kazancı (Q5'e göre 133 MB) buna değmiyor.
3. **Q3_K_M çöküyor:** alan doğruluğu 0.18, komutların %76'sında gereksiz soru. **Düşük uydurma oranı
   (0.005) burada başarı değil**, model neredeyse her şeye soru soruyor. Uydurma oranı tek başına
   okunmamalı; gereksiz soru oranıyla birlikte değerlendirilmeli.
4. **Gecikme azalması doğrusal değil:** Q4'ten Q3'e hız artmadı (138 vs 152 tok/s). Nedeni ölçülmedi; olası
   açıklamalar sabit istek başı ek yükü (HTTP, ayrıştırma) ve Q3'ün kısa yanıtları (gereksiz soru):
   kısa yanıtlarda tok/s karşılaştırması sapmalı.
5. **Gürültü:** n=400, tek çalıştırma, tek GPU. Gecikme p95'i arka plan yüküne duyarlı; ölçüm sırasında
   oyun/Godot kapalıydı. Q8/Q5/F16 arasındaki doğruluk farkları (±0.005 ≈ 1–2 örnek) anlamlı değil.
6. **GPU yükü uyarısı:** aynı model başka bir GPU yükü (oyun) varken 25 tok/s verdi, boşken 164 tok/s
   (`llama-bench`). Gecikme sayıları yalnızca boş GPU'da karşılaştırılabilir.
7. **HF bf16 ile GGUF F16 farkı** (uydurma 0.016 vs 0.011, 1 örnek) nedeni araştırılmadı (olasılıkla farklı çekirdekler/yuvarlama).

## Tekrar üretme

```bash
uv run --extra train python -m skymind.optimize.merge --adapter outputs/sft-v1/adapter --out outputs/merged-sft
# GGUF dönüşümü: llama.cpp convert_hf_to_gguf.py (sentencepiece gerekir) + llama-quantize
uv run python -m skymind.optimize.eval_gguf --gguf outputs/gguf/skymind-sft-Q5_K_M.gguf \
    --server third_party/llama-cuda/llama-server.exe --name gguf-Q5_K_M --limit 400
```

---

# ONNX Runtime (Faz 6b) — CPU

Aynı birleştirilmiş SFT modeli `onnxruntime-genai` model oluşturucusuyla ONNX'e çevrildi (fp32 ve int4),
aynı 400 örnekte, **CPU'da**, ardışık istek, greedy ölçüldü. Adil karşılaştırma için llama.cpp Q4_K_M de
CPU'da (`-ngl 0`) ölçüldü. GPU kullanılmadı.

| Çalışma zamanı / format | Boyut | json_valid | type_acc | field_acc | uydurma ↓ | gereksiz soru ↓ | p50 | p95 | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| ONNX fp32 | 5905 MB | 0.985 | 0.920 | 0.840 | 0.011 | 0.108 | 3.84 s | 8.31 s | 8.7 |
| ONNX int4 | 937 MB | 0.973 | 0.907 | **0.854** | 0.037 | 0.014 | 1.28 s | 2.50 s | 25.3 |
| llama.cpp Q4_K_M (CPU) | 940 MB | 0.973 | 0.828 | 0.703 | 0.011 | 0.255 | 1.08 s | 2.22 s | 30.6 |
| *llama.cpp Q4_K_M (GPU, önceki tablo)* | 940 MB | 0.988 | 0.870 | 0.758 | 0.011 | 0.203 | 0.21 s | 0.47 s | 152 |

## Okuma ve uyarılar

1. **ONNX fp32, doğruluk açısından HF/GGUF F16 ile aynı** (alan 0.840 vs 0.842): dönüşüm modeli bozmadı.
   Ama CPU'da 3.8 s/istek ve 5.9 GB. Dağıtım için uygun değil.
2. **ONNX int4 ve llama.cpp Q4_K_M aynı boyutta (~940 MB) ama farklı hata profili.** ONNX int4 gereksiz
   soruyu neredeyse hiç sormuyor (0.014) ve alan doğruluğu en yüksek (0.854), ama uydurma oranı 0.037
   (188 örnekten ~7), yani eksik komutta daha sık plan üretiyor. llama.cpp Q4_K_M ise tersine temkinli:
   uydurma 0.011 ama gereksiz soru 0.255. Operatör asistanında uydurma, gereksiz sorudan daha maliyetli
   olduğu için **bu iki Q4 varyantından hiçbiri koşulsuz üstün değil**; Q5_K_M (GPU/CPU) hâlâ en dengeli.
3. **Aynı GGUF dosyası farklı backend'de farklı sonuç verdi** (Q4_K_M alan doğruluğu GPU 0.758, CPU 0.703).
   Greedy üretim sayısal olarak backend'e bağlı; nicemlenmiş modelde küçük logit farkları karar
   sınırındaki örnekleri çeviriyor. Tek tek varyant farklarını (~±0.05) tek çalıştırmayla kesin yorumlamayın.
4. **Hız:** CPU'da llama.cpp Q4_K_M (30.6 tok/s) ONNX int4'ten (25.3 tok/s) ~%21 hızlı. Bu fark tek
   çalıştırma ve 24 iş parçacığı varsayılanıyla ölçüldü, iş parçacığı sayısı ayarlanmadı. GPU, CPU'dan
   ~5× hızlı (152 vs 30.6 tok/s). Bunun nedeni ölçülmedi; bellek bant genişliği olası bir açıklama.
5. **Ölçüm koşulları:** ölçüm sırasında PC'de başka uygulamalar açık olabilirdi (CPU ölçümü GPU'dan
   daha hassas). p95 değerleri özellikle gürültülü.

## Tekrar üretme

```bash
uv run --extra train --extra optimize python -m onnxruntime_genai.models.builder \
    -m outputs/merged-sft -o outputs/onnx/int4-cpu -p int4 -e cpu
uv run --extra optimize python -m skymind.optimize.eval_onnx --model-dir outputs/onnx/int4-cpu --name onnx-int4-cpu
```

# Faz 5: RAG ve METAR/NOTAM özetleme

## Retrieval (kurgusal "SkyMind X1/X2/X3" kılavuzu)

Korpus **kurgusaldır** ve şablonlardan üretilir (`skymind/rag/corpus.py`): 91 parça, 240 soru, 3 araç
modeli aynı cümle yapılarını farklı sayılarla tekrarlar, bu yüzden "benzer ama yanlış" pasajlar var.
İki soru türü: `lexical` (kılavuzdaki sözcüklerle) ve `paraphrase` (eş anlamlı/dolaylı anlatım).
Yoğun model: `intfloat/multilingual-e5-small` (CPU). Hibrit: RRF(BM25-F5, dense). 8 korpus tohumu,
ortalama ± standart sapma, altın parça düzeyinde:

| Yöntem | Tümü R@1 | Tümü R@5 | Tümü MRR | Paraphrase R@5 | Cümle isabeti |
|---|---|---|---|---|---|
| BM25 | 0.565±0.020 | 0.751±0.019 | 0.660±0.017 | 0.527±0.039 | 0.683 |
| BM25 + F5 kök | 0.578±0.015 | 0.819±0.011 | 0.689±0.012 | 0.647±0.021 | 0.684 |
| Dense (e5-small) | 0.526±0.012 | 0.822±0.012 | 0.659±0.005 | 0.663±0.021 | 0.693 |
| **Hibrit (RRF)** | **0.586±0.011** | **0.852±0.015** | **0.706±0.010** | **0.710±0.025** | **0.703** |

### Okuma ve uyarılar
1. **Hibrit en iyi, ama kazanç küçük.** R@5'te BM25-F5 / dense'e göre +3 puan (≈2 sd), paraphrase
   sorularında +5–6 puan. **R@1'de hibrit ile BM25-F5 farkı (0.586 vs 0.578) standart sapma içinde**,
   üstünlük iddia etmiyorum.
2. **Dense, kelime örtüşen sorularda BM25'ten kötü** (lexical R@1 0.719 vs 0.799): model adı ("X2")
   ve hata kodu ("E-217") gibi tam eşleşmeler yoğun gömmede zayıf kalıyor. Hibritin gerekçesi bu.
3. **Paraphrase R@1 hâlâ düşük (≤0.38).** Üç modelin neredeyse aynı cümleleri, doğru modeli seçmeyi
   zorlaştırıyor; reranker denenmedi.
4. **Cümle isabeti ~0.70**, ilk 3 parçadan seçilen cümlenin altın cümle olma oranı. Üst sınır
   retrieval R@3 ile sınırlı. Seçici basit BM25; eğitilmiş bir cümle seçici/LLM denenmedi.
5. **Genelleme uyarısı:** korpus ve sorular şablondan üretildi, gerçek kullanıcı sorularını ve gerçek
   doküman gürültüsünü temsil etmez. Mutlak sayılar iyimser olabilir, yalnızca yöntemler arası
   karşılaştırma anlamlıdır. Tek yoğun model (e5-small), reranker ve RAGAS tarzı LLM-tabanlı
   faithfulness bu aşamada **yok**.

## METAR/NOTAM (kural tabanlı, klasik NLP)

`skymind/aviation/`: METAR ve NOTAM ayrıştırıcıları, Türkçe özet, sentetik üreticiler. Testler (22):
el yazımı örnekler, 400 rastgele METAR ve 300 rastgele NOTAM ile gidiş-dönüş eşitliği, bilerek
bozulan ayrıştırıcıda testlerin başarısız olması (mutasyon).

- **Sayısal sadakat denetimi** (`faithfulness.py`): özetteki her sayı kaynaktan türetilebilir mi.
  Kural tabanlı özetleyicide yapı gereği 0 uydurma sayı çıkar; denetim LLM özetlerini aynı ölçütle
  ölçmek için yazıldı (henüz LLM özetleyici **yok**, ölçülmedi).
- **Sınır:** kaynakta tesadüfen bulunan bir sayıyla çakışan uydurma sayıyı yakalamaz (alt sınır
  ölçüsüdür). Yanlış yön/hadise gibi anlamsal hataları görmez.
- **Gidiş-dönüş doğruluğu kapsam gösterir, genelleme değil:** üretici ve ayrıştırıcıyı aynı kişi
  yazdı. Gerçek METAR/NOTAM akışı üzerinde ölçüm yapılmadı.
- Uçuş uygunluk eşikleri (`Limits`) prototip değerleridir, SHGM/ICAO kuralı değildir.

## Tekrar üretme
```bash
uv run --no-sync python -m skymind.rag.evaluate --seeds 1,2,3,4,5,6,7,8 --out outputs/eval/rag-8seeds.json
uv run --no-sync pytest tests/test_rag.py tests/test_aviation.py
```
