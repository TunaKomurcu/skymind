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

# Faz 7: C++ servis vs Python servis (yük testi) ve A/B denemesi

Aynı komutlar, aynı uçlar, aynı sunucu tarafı şema doğrulaması, aynı GPU (RTX 4060 Laptop, 8 GB).
Kapalı döngü yük testi (`skymind/serve/loadtest.py`), düzey başına 60 istek, 5 ısınma isteği,
istemci tarafı gecikme (kuyruk bekleme dahil). Servisler batch=1 ve tek bağlam: istekler sırayla işlenir.

| Servis | Eşzamanlılık | İstek/sn | Token/sn | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| **C++ / llama.cpp Q5_K_M** | 1 | 3.91 | 136.8 | **258 ms** | 489 ms | 521 ms |
| | 2 | 4.05 | 141.9 | 444 ms | 741 ms | 842 ms |
| | 4 | 4.08 | 142.8 | 940 ms | 1300 ms | 1337 ms |
| | 8 | 4.07 | 142.4 | 1953 ms | 2346 ms | 2397 ms |
| **Python / HF transformers bf16** | 1 | 0.76 | 27.3 | **1243 ms** | 2652 ms | 2890 ms |
| | 2 | 0.68 | 24.5 | 2728 ms | 4335 ms | 4525 ms |
| | 4 | 0.62 | 22.2 | 5912 ms | 9440 ms | 9590 ms |
| | 8 | 0.65 | 23.4 | 12171 ms | 14762 ms | 15221 ms |

Hata: iki serviste de 0. Geçerli çıktı oranı: ikisinde 0.983 (59/60).

## Okuma ve uyarılar
1. **Tek istekte C++ yaklaşık 4.8× daha düşük p50 gecikme, ~5× daha yüksek token/sn verdi.** Ama bu
   **iki yığının** karşılaştırması: model formatı da farklı (Q5_K_M nicemlenmiş GGUF vs bf16). Farkın
   ne kadarının dilden geldiği ilk ölçümde ayrıştırılmamıştı; aşağıdaki "aynı GGUF ve aynı llama.cpp
   ikilileri" bölümü bunu ölçtü: dil payı ~%6–7. "C++ Python'dan 5× hızlı" diye okunmamalı.
2. **Eşzamanlılıkla verim artmıyor, gecikme doğrusal artıyor** (iki serviste de): tek bağlam ve
   batch=1 olduğu için beklenen davranış. Sürekli batching (vLLM benzeri) yok, bu bir sınırlama.
3. **p99 bu örnek büyüklüğünde (n=60) neredeyse maksimuma eşit**, kararlı bir kuyruk istatistiği değil.
4. **Ölçüm koşulları:** tek çalıştırma. Ölçüm sırasında arka planda video oynuyordu (GPU kullanımı
   %3–35 arası, uzun sürmedi) ve sunucular BelowNormal öncelikteydi. Bellek karşılaştırması
   **yapılmadı** (Python servisi için nvidia-smi toplam 3.7 GB gösterdi, C++ için ölçülmedi).
5. Aynı GGUF'u Python'dan çalıştıran taban çizgisi (llama-cpp-python) sonradan eklendi, aşağıdaki bölüme bakın.

## A/B denemesi (gateway önünde SFT vs DPO v2, ikisi de GGUF Q5_K_M, aynı GPU)
400 farklı komut, eşzamanlılık 4, 50/50 ağırlık, deterministik atama (sft: 182, dpo: 223 istek).

| Varyant | İstek | Geçerli çıktı | p50 | p95 |
|---|---|---|---|---|
| SFT | 182 | 0.9725 | 1188 ms | 2320 ms |
| DPO v2 | 223 | 0.9776 | 1175 ms | 2236 ms |

Geçerlilik farkı için iki oranlı z-testi p = 0.74: **anlamlı fark yok**. Ağ geçidi yalnızca
geçerliliği karşılaştırabilir (altın etiket canlı trafikte yok); uydurma/alan doğruluğu için
`docs/RESULTS.md` başındaki çevrimdışı değerlendirme geçerlidir. İki model aynı GPU'yu paylaştığı
için gecikmeler tek sunucuyla ölçülenden yüksek (p50 ~1.2 s, c=4). Atama komut karmasına bağlı
olduğu için bölme tam 50/50 çıkmaz (burada %45/%55).

## Tekrar üretme
```bash
# C++ servis (Windows, hazır llama.dll yoluyla derlenmiş): cpp/README.md
build\server\skymind_server.exe -m outputs\gguf\skymind-sft-Q5_K_M.gguf --port 8091 --ngl 99 --system cpp\configs\system_short.txt
uv run --extra train python -m skymind.serve.app --model outputs/merged-sft --port 8092
uv run python -m skymind.serve.loadtest --url http://127.0.0.1:8091 --name cpp-q5km-gpu --levels 1,2,4,8 --requests 60
uv run python -m skymind.serve.gateway --variant sft=http://127.0.0.1:8091@50 --variant dpo=http://127.0.0.1:8093@50
```
Ham sonuçlar: `outputs/loadtest/*.json` (git'e girmez).

# Faz 2: klasik NLP — niyet sınıflandırma ve varlık çıkarımı (NER)

Veri: planlayıcı bölümleriyle aynı bölümler (`data/processed`). Niyet = planın baskın eylemi veya
`clarification`/`refusal` (10 sınıf, test n=1477). NER: 8 varlık türü (ALT, RADIUS, DIST, SPEED, DUR,
COORD, PLACE, DIR); etiketler komut metnine altın plan değerlerinin hizalanmasıyla **zayıf** üretildi
(kapsama %95.9, hizalanamayanlar atıldı). BERTurk (`dbmdz/bert-base-turkish-cased`) ince ayar, tek tohum.

## Niyet (test, n=1477)
| Model | Doğruluk | Makro-F1 |
|---|---|---|
| TF-IDF + lojistik regresyon | 0.9621 | 0.8505 |
| BERTurk ince ayar (3 epoch, ~200 sn) | **0.9858** | **0.9816** |

Makro-F1 farkı doğruluktan çok daha büyük: TF-IDF az örnekli sınıflarda zayıf. Tek tohum olduğu için
doğruluktaki 2.4 puanlık fark yönü gösterir, aralığı değil.

## NER: neden iki ayrı test
| Küme | BERTurk mikro-F1 | Kural tabanlı etiketleyici |
|---|---|---|
| Zayıf-etiket testi (n=688 cümle) | 0.9994 | 0.9936 |
| Elle yazılmış KOLAY (20 cümle) | 1.000 | — |
| Elle yazılmış ZOR (20 cümle) | 0.867 | 0.853 |

1. **Zayıf-etiket testi bilgi taşımıyor.** Etiketler komut şablonlarının/planların aynı kurallarından
   çıktığı için kural tabanlı etiketleyici de 0.994 alıyor; BERTurk'ün 0.999'u "kuralları öğrendi"
   demek, "gerçek yazıma genelliyor" demek değil. Döngüsel bir ölçü.
2. **Bağımsız sınama:** `data/gold/ner_manual.txt` (40 cümle, elle, `[metin|ETİKET]` imli). ZOR kısım
   bilerek zor örnekler içerir (sayı sözcükleri, `saatte 72 kilometre`, `20 dk`, `2,5 km` yükseklik, şablon dışı yer adları).
   Farkın kaynağı tek varlık düzeyinde (BERTurk 0.867 vs kurallar 0.853): **BERTurk'ün kurallardan
   daha iyi olduğu söylenemez.** Zayıf noktalar ALT (F1 0.58) ve DUR (0.67); diğer türler 1.0.
3. n=20/20 küçük: tek varlık hatası mikro-F1'i ~1–2 puan oynatır. Elle set eğitim verisinden bağımsız ama
   **Claude (asistan) tarafından yazıldı**, proje sahibi gözden geçirmedi; üslup önyargısı olabilir.

## Veri hatası ve düzeltmesi
Paraphrase üretiminde `{style, command}` nesneleri metne çevrilmişti (485 satır bozuk). Ayrıştırıcı
düzeltildi (`item_text`), veri onarıldı (`skymind/data/repair.py`), bölümler yeniden üretildi (eski
hali `outputs/data-backup/`). Planlayıcılar temiz test kümesinde yeniden ölçüldü:

| | Alan doğruluğu (önce → sonra) |
|---|---|
| SFT | 0.842 → 0.842 |
| DPO | 0.848 → 0.850 |

Sonuçlar ve sıralama değişmedi. Altın aday dosyası etkilenmemişti.

## Tekrar üretme
```bash
uv run python -m skymind.nlp.baselines
uv run --extra train python -m skymind.nlp.train --task intent --name intent-berturk
uv run --extra train python -m skymind.nlp.train --task ner --name ner-berturk
```

# Faz 5b: sıkıştırma — damıtma ve budama (niyet sınıflandırıcı, BERTurk 12 katman)

Hedef sorular: öğrenci ne kadar küçülür/hızlanır, ne kadar doğruluk kaybeder; KD, aynı bütçede düz
çapraz entropiden (CE) iyi mi; budama gerçekten hız kazandırıyor mu? Test: niyet, n=1477.

**Yöntem (ilk koşu kirliydi, yeniden tasarlandı):** ilk sürümde öğrenciler ince ayarlı öğretmenin
ağırlıklarından başlayıp fazladan eğitim almış, gecikmeler de ben başka iş yaparken ölçülmüştü
(`outputs/nlp/compress-v1-confounded.json`, kullanılmıyor). Şimdi:
* Öğrenciler **önceden eğitilmiş BERT**'in katman alt kümesinden başlar (6L: 0,2,4,6,8,10; 4L: 0,4,8,11;
  2L: 0,11), öğretmenle aynı 3 epoch. KD = CE + öğretmen yumuşak hedefleri.
* Budamalı satırlar öğretmen + **1 epoch** ince ayar alır; adil kıyas için kontrol: `teacher-ft1`
  (budamasız, 1 epoch ek). 2 tohum (0,1), ortalama ± standart sapma.
* **Gecikme doğruluktan ayrı ölçülür** ve yalnızca mimariye bağlıdır (eğitimsiz mimariler): ısınma,
  5 dönüşümlü tur, en küçük p50; batch 1, CPU 4 iş parçacığı ve GPU (RTX 4060 Laptop).

## Gecikme (batch 1, medyan)
| Varyant | Parametre | CPU p50 | Hızlanma | GPU p50 | Hızlanma |
|---|---|---|---|---|---|
| öğretmen 12L | 110.6M | 27.9 ms | 1.00× | 6.3 ms | 1.00× |
| öğrenci 6L | 68.1M | 13.8 ms | **2.0×** | 3.3 ms | 1.9× |
| öğrenci 4L | 53.9M | 9.1 ms | **3.1×** | 2.5 ms | 2.6× |
| öğrenci 2L | 39.7M | 5.4 ms | **5.2×** | 1.6 ms | 3.9× |
| kafa budama %30 | 102.2M | 28.9 ms | 0.96× | 6.5 ms | 0.97× |
| kafa budama %50 | 96.5M | 22.7 ms | 1.2× | 6.2 ms | 1.0× |
| kafa budama %70 | 90.9M | 22.0 ms | 1.3× | 6.0 ms | 1.05× |
| magnitude %80 (yoğun çekirdek) | 110.6M | 27.9 ms | 1.0× | 6.2 ms | 1.0× |

## Doğruluk (2 tohum, ort. ± sd)
| Varyant | Doğruluk | Makro-F1 | Sıfır olmayan param. |
|---|---|---|---|
| öğretmen 12L (3 epoch, tek model) | 0.9858 | 0.9816 | 110.6M |
| **teacher-ft1 (kontrol, +1 epoch)** | 0.9902 ± 0.0034 | 0.9887 ± 0.0057 | 110.6M |
| öğrenci 6L KD | 0.9770 ± 0.0038 | 0.8678 ± 0.0053 | 68.1M |
| öğrenci 6L CE | 0.9909 ± 0.0015 | 0.8920 ± 0.0008 | 68.1M |
| öğrenci 4L KD | 0.9875 ± 0.0014 | 0.9833 ± 0.0003 | 53.9M |
| öğrenci 4L CE | 0.9688 ± 0.0077 | 0.9421 ± 0.0199 | 53.9M |
| öğrenci 2L KD | 0.9865 ± 0.0019 | 0.8874 ± 0.0005 | 39.7M |
| öğrenci 2L CE | 0.9776 ± 0.0105 | 0.8751 ± 0.0181 | 39.7M |
| magnitude %50 + 1 epoch | 0.9905 ± 0.0000 | 0.9407 ± 0.0689 | 68.2M |
| magnitude %80 + 1 epoch | 0.8656 ± 0.0052 | 0.7636 ± 0.0070 | 42.7M |
| kafa budama %30 + 1 epoch | 0.9919 ± 0.0010 | 0.9421 ± 0.0705 | 102.2M |
| kafa budama %50 + 1 epoch | 0.9902 ± 0.0034 | 0.8916 ± 0.0030 | 96.5M |
| kafa budama %70 + 1 epoch | 0.9912 ± 0.0028 | 0.9422 ± 0.0679 | 90.9M |

## Okuma ve uyarılar
1. **Hız kazancı yalnızca katman çıkarmaktan geliyor.** 2/4/6 katmanlı öğrenciler CPU'da 5.2×/3.1×/2.0×
   hızlı; kafa budama en iyi ~1.3× (CPU), GPU'da ~1.0×; yapısal olmayan (magnitude) budama **hiç hızlandırmaz**:
   sıfırlanmış ağırlıklar yoğun matris çarpımında yine hesaplanır, "sıfır olmayan parametre" azalması
   gecikmeye yansımaz. Kafa budama %30 ölçüm gürültüsü sınırında (0.96×, yani hızlanma yok).
2. **Doğruluk bu görevde tavana yakın ve farklar küçük.** 1477 örnekte 0.3 puan ≈ 4–5 örnek; 2 tohumla
   çoğu farkın (örn. 6L CE 0.991 vs ft1 kontrolü 0.990) yönü bile güvenilir değil. Hiçbir sıkıştırma
   varyantı "kayıpsız" olarak sunulmamalı; yalnızca magnitude %80 açıkça çöküyor (0.866).
3. **Öğretmen (0.986) kendi +1 epoch kontrolünden (0.990) düşük:** öğretmen biraz eksik eğitilmişti.
   Sıkıştırılmış modellerin "öğretmenden iyi" görünmesi kısmen ek eğitimden gelir; bu yüzden
   karşılaştırma `teacher-ft1`'e göre yapılmalı.
4. **KD'nin CE'ye üstünlüğü tutarsız:** 4L ve 2L'de KD daha iyi (0.988 vs 0.969, 0.987 vs 0.978),
   6L'de CE daha iyi (0.977 vs 0.991). İki tohumla "KD yardım eder" denemez; bu deneyde tutarlı bir
   KD kazancı **gösterilmedi**.
5. **Makro-F1 neredeyse iki kipli (~0.89 ya da ~0.94–0.99) ve sd büyük.** Doğruluk 0.99 iken makro-F1'in
   0.89'a düşmesi, bir az örnekli sınıfın bazı koşularda hiç doğru bulunmadığını gösteriyor. Hangi
   sınıf olduğunu koşu başına sınıf raporu kaydetmediğim için **doğrulamadım** (tahmin: test kümesinde
   en az örnekli niyet). Makro-F1 farkları bu yüzden doğruluktan daha gürültülü okunmalı.
6. Tek veri kümesi, tek görev (niyet), 2 tohum, öğrenci eğitimi 3 epoch. Parametre sayısı ≠ gecikme;
   gecikme kendi donanımımda ölçüldü (4 iş parçacığı), başka CPU'da oranlar değişebilir.

## Tekrar üretme
```bash
uv run --extra train python -m skymind.nlp.compress_run --phase acc --seeds 0,1      # ~25 dk, GPU
uv run --extra train python -m skymind.nlp.compress_run --phase latency               # sessiz makinede
uv run --no-sync pytest tests/test_nlp.py                                               # budama eşdeğerlik testi
```

## Üretken RAG yanıtlayıcı (Faz 5, LLM ile)
`skymind/rag/generate.py`: hibrit (BM25-F5 + e5-small, RRF) ilk 3 parça + cümle kimlikli prompt;
`google/gemma-3-27b-it` (OpenRouter, sıcaklık 0). 60 cevaplanabilir soru (31 sözcüksel, 29 paraphrase) ve
**kılavuzda olmayan** 24 soru (tanımsız "X4" modeli, hiç yer almayan özellikler). Tek koşu, tek tohum
(3); toplam OpenRouter harcaması ~0.19 USD (limit 4).

| Ölçü | Değer |
|---|---|
| Doğru parça ilk 3'te (R@3) | 0.817 |
| Doğru cevap (60 sorunun) | 0.767 |
| Doğru cevap, doğru parça geldiyse | **0.939** (46/49) |
| Yanlışlıkla "Bilgi bulunamadı" | 0.183 (11/60; 8'i parça gelmediği için) |
| Atıf geçerliliği (var olan parça+cümle) | 1.000 |
| Cevap verip atıfsız bırakma | 0.000 |
| Atıf altın cümleyi gösteriyor (tüm sorular) | 0.733 (cevap verilenlerde 44/49) |
| Cevapta kaynaklarda olmayan sayı | 0.000 |
| **Cevapsız sorularda vazgeçme** | **1.000** (24/24) |

Sözcüksel sorularda doğruluk 31/31, paraphrase'te 0.517 (R@3 0.62).

### Okuma ve uyarılar
1. **Ana darboğaz üretici değil retrieval:** doğru parça geldiğinde doğruluk %94; gelmediğinde model
   8/11 kez dürüstçe vazgeçti, 3 kez **yanlış parçadan, makul görünen yanlış cevap** verdi
   (ör. sorulan duruma uymayan bir hata kodunu veya başka bir değeri, geçerli bir atıfla; başka modelin değeri olup olmadığını tek tek doğrulamadım).
2. **Atıf geçerliliği 1.0 desteklenme anlamına gelmez.** Ölçü yalnızca atfın var olan bir parça+cümleyi
   gösterdiğini doğrular; o cümlenin cevabı gerçekten desteklediğini değil. Yukarıdaki 3 yanlış cevap
   geçerli atıf taşıyor. Desteklenme (NLI/insan) ölçülmedi.
3. "Kaynaklarda olmayan sayı 0" alt sınır: sayı doğrulaması sayının kaynak parçalarda *herhangi bir yerde*
   geçmesine bakar (üç model benzer değerler paylaştığı için çarpışma olabilir).
4. Cevapsız soru kümesi küçük ve şablon tabanlı (24); "1.0 vazgeçme" bu şablon sorular için geçerli,
   kurnaz/çelişkili sorular için kanıt değil. Vekil doğruluk metriği (sayı eşleşmesi / kök örtüşmesi)
   kaba; elle denetlenmedi. Tek model, tek koşu: model karşılaştırması veya güven aralığı yok.

```bash
uv run --extra train python -m skymind.rag.generate --model google/gemma-3-27b-it --n 60 --out outputs/eval/rag-gen.json
```

## LLM ile METAR/NOTAM özeti vs kural tabanlı (Faz 5)
`skymind/aviation/llm_summary.py`. Sentetik üreticiden 80 METAR + 48 NOTAM (tohum 21; altın yapı bilinir).
Aynı prompt (sıfır-örnek, "yalnızca rapordaki bilgiyi kullan, sayıları aynen koru, birim çevirme"),
sıcaklık 0, `google/gemma-3-27b-it` ve `google/gemma-3-4b-it` (OpenRouter, toplam ~0.01 USD).

| Özetleyici | Rapor | Bilgi geri çağırma (ort.) | Tüm alanlar var | Uydurma sayı içeren | Anlamsal hata |
|---|---|---|---|---|---|
| kural tabanlı | METAR | 1.000 | 1.000 | 0.000 | 0.000 |
| gemma-3-27b | METAR | 0.852 | 0.463 | 0.350 | 0.000 |
| gemma-3-4b | METAR | 0.633 | 0.075 | 0.400 | 0.000 |
| kural tabanlı | NOTAM | 1.000 | 1.000 | 0.000 | 0.000 |
| gemma-3-27b | NOTAM | 0.739 | 0.271 | 0.312 | 0.021 |
| gemma-3-4b | NOTAM | 0.438 | 0.021 | 0.646 | 0.167 |

"Anlamsal hata": METAR'da rüzgâr yönü sözcüğünün altın dereceyle çelişmesi; NOTAM'da "kapalı / servis dışı /
yasak" durumunun özette atlanması. Kural tabanlı özetleyici, metrikler aynı ayrıştırıcıdan türediği için
yapı gereği 1.0 alır; bu satır yalnızca metriğin kendini doğrulamasıdır, kural tabanlının "daha iyi"
olduğunun kanıtı değildir (özet kalitesi/okunabilirliği ölçülmedi, kural tabanlının NOTAM metni ise büyük
ölçüde İngilizce kısaltmaları taşıyor: `ayrıntı: ... ERECTED 2NM FROM ARP`).

### Gerçek LLM hataları (örneklerden)
* **NOTAM tarih kodunu yanlış okuma** (en yaygın): `B) 2506062300` için "25 Haziran 2024 saat 23:00"
  (YYAAGGSSdd kodunu GG=25, yıl=2024 diye yorumlamış). Başlangıç tarihi 27B'de 48 NOTAM'ın 24'ünde
  eksik/yanlış; kaynakta olmayan "2024" yılı 19'unda geçiyor.
* **Konum karışıklığı:** `A) LTAF` yerine Q satırındaki FIR/konu kodu (`LTAA/QMXLC`) "havaalanı" diye yazılmış
  (15 NOTAM'da konum eksik/yanlış).
* **METAR rakam okuma:** `01031KT` → "3 knot", `0800` → "8000 m", `OVC030` → "300 ft", `M07/M12` → eksi
  işareti kaybolmuş ("7 ile 12 derece arasında"). 27B'de sıcaklık 15, çiğ noktası 21, bulut yüksekliği 16 raporda
  eksik/yanlış.
* 4B ayrıca NOTAM durumunu atlıyor (8/48 "kapalı/servis dışı" yok) ve çok daha sık sayı uyduruyor.

### Okuma ve uyarılar
1. **Metrik iki kez düzeltildi, sonuçlar görüldükten sonra.** İlk koşuda uydurma-sayı oranları 0.43/0.56
   (27B METAR/NOTAM) çıktı; incelemede bir kısmı metriğin kusuruydu (ör. `061650Z` → "1650", `0800-1600` →
   "08:00", yıl "25" → "2025", koordinatın derece/dakikaya bölünmesi, ham zaman kodunun parçalanması,
   ay adıyla yazılmış tarih). Bu meşru yeniden biçimlendirmeleri türetilmiş sayıdan saydım; yukarıdaki
   tablo düzeltilmiş metrikle. Düzeltmeler yalnızca *gevşetme* yönündeydi ama post-hoc yapıldı, bu bir
   ayarlama riskidir; ham özetler `outputs/eval/summ-llm.json`'da (git'e girmez) denetlenebilir.
2. **Sayı denetimi hâlâ alt sınır:** kaynakta başka yerde geçen bir sayıyla çakışan yanlış sayı yakalanmaz.
   Gerçek hata oranı tablodan düşük olamaz, yüksek olabilir.
3. **Ölçüler dar:** yanlış hadise/sıra, eksik uyarı ve akıcılık görülmez; "tüm alanlar var" oranı katı bir
   ölçüdür (tek eksik alan raporu başarısız sayar). Biçim farkları (ham kod vs sözlü tarih) yalnızca belirli
   kalıplar için hoş görülüyor.
4. **Tek prompt, sıfır örnek, tek koşu, küçük n; prompt ayarlanmadı.** Few-shot veya ham kodları açıklayan bir
   prompt, tarih ve rakam hatalarını büyük olasılıkla azaltır; bu denenmedi. Modeller havacılık için
   özelleştirilmemiş genel modeller.
5. **Sentetik üretici kusuru:** koordinat dakikaları 59'u aşabiliyor (ör. `4064N`); gerçek NOTAM'da olmaz.
   Sonuçları etkilemez ama üretici gerçekçi değil.
6. Pratik çıkarım: sayı ve tarih taşıyan operasyonel özetlerde LLM çıktısı **doğrulayıcısız** kullanılmamalı;
   kural tabanlı özet güvenilir çekirdek, LLM yalnızca anlatım katmanı olabilir ve yine sayı denetiminden
   geçirilmeli.

```bash
uv run --no-sync python -m skymind.aviation.llm_summary --models google/gemma-3-27b-it,google/gemma-3-4b-it
```

## C++ servis vs Python servis, aynı GGUF ve aynı llama.cpp ikilileri (Faz 7 ek)
Soru: C++'ın hız avantajının ne kadarı dilden? `skymind/serve/engines.py::LlamaCppPyEngine`: aynı
Q5_K_M GGUF, aynı llama.cpp **ikili dosyaları** (C++ servisin kullandığı `third_party/llama-cuda`
DLL'leri, `LLAMA_CPP_LIB_PATH` ile yüklendi), aynı ChatML istemi, açgözlü örnekleme, bağlam 2048, en çok 400
token, her istekte KV sıfırlama; yalnızca çevre kod (HTTP, JSON, şema doğrulama, istek döngüsü) Python
(FastAPI + llama-cpp-python bağlaması, ayrı sanal ortam). Yük testi birinci ölçümle aynı komutlar ve ayarlarla
(düzey başına 60 istek, 5 ısınma, **veri onarımından sonraki güncel komut havuzuyla; C++ yeniden koşuldu**).

| Servis | c | İstek/sn | Token/sn | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| C++ / llama.cpp | 1 | 3.75 | 131.3 | 277 ms | 517 ms | 526 ms |
| | 2 | 3.70 | 129.3 | 499 ms | 812 ms | 900 ms |
| | 4 | 3.77 | 131.8 | 1007 ms | 1389 ms | 1476 ms |
| | 8 | 3.75 | 131.3 | 2125 ms | 2530 ms | 2590 ms |
| Python / llama-cpp-python | 1 | 3.51 | 122.3 | 262 ms | 549 ms | 567 ms |
| | 2 | 3.56 | 124.3 | 505 ms | 852 ms | 962 ms |
| | 4 | 3.55 | 123.8 | 1070 ms | 1488 ms | 1564 ms |
| | 8 | 3.50 | 122.2 | 2290 ms | 2686 ms | 2771 ms |

* **Aynı çekirdek ve model olduğunda fark küçük: ~%6–7 daha düşük verim, p95'te ~%6–8 daha yüksek gecikme
  Python'da** (c=1 p50'de Python 262 ms, C++ 277 ms; bu fark ölçüm gürültüsü sınırında, yön tutarsız).
  Dolayısıyla ilk tablodaki ~4.8× farkın (HF bf16 vs C++ Q5_K_M) **büyük kısmı dilden değil**, HF
  `generate` yığınından ve model/nicemleme biçiminden geliyor. Dil payı bu denemede en çok birkaç yüzde.
* **Çıktı eşitliği (60 komut, açgözlü):** 58'i birebir aynı, 2'si farklı: biri iki servisin farklı
  netleştirme sorusu metni üretmesi, biri `radius_m` (C++) ve `altitude_m` (Python) alan adı (C++'ınki şemadan
  geçmiyor). Aynı ağırlık ve aynı ikili olsa da küçük sayısal farklar nadiren ayrışmaya yol açıyor; bu yüzden
  C++ koşusunda geçerlilik 0.983 (59/60), Python'da 1.0: **tek örnek, sistematik fark değil**.
* **Bellek (GPU, toplam kullanımın arka plana göre artışı):** C++ ~1533 MiB, Python ~1557 MiB (+24 MiB):
  aynı. C++ işlem çalışma kümesi ~1.5 GB (model dosyası eşlemesi dahil). **Python işleminin ana bellek
  ölçümü geçersiz** (venv başlatıcı süreci ölçüldü, 5 MiB gösterdi); bu sütun raporlanmıyor. HF bf16 servisi için
  önceki tek ölçüm (nvidia-smi toplam ~3.7 GB) yeni koşuyla karşılaştırılabilir değil.
* **Koşullar:** tek koşu, arka planda video akışı vardı, sunucular BelowNormal. C++ ve Python aynı anda değil
  sırayla ölçüldü. **İlk tablonun (HF vs C++) komut havuzu veri onarımından önceki havuzdu**; yeniden koşulan
  C++ (burada) ve eski C++ koşusu birbirine yakın (c=1: 277 ms vs 258 ms), ama HF ölçümü güncel havuzla
  tekrarlanmadı.
* Teknik not: llama-cpp-python'un hazır CUDA tekerleği bu CPU'da `0xc000001d` (geçersiz komut) ile çöküyor
  (CPU arka ucu uyumsuz komut setiyle derlenmiş); bu yüzden C++ servisin dinamik arka uçlu DLL'leri
  `ggml_backend_load_all_from_path` ile yüklendi. Tekerleğin kendi DLL'leri ölçümde kullanılmadı.

```bash
# Python (ayrı venv: llama-cpp-python), C++ DLL'leriyle
set LLAMA_CPP_LIB_PATH=third_party\llama-cuda
python -m skymind.serve.app --engine llamacpp --model outputs\gguf\skymind-sft-Q5_K_M.gguf --port 8094 --system cpp\configs\system_short.txt
python -m skymind.serve.loadtest --url http://127.0.0.1:8094 --name py-llamacpp-gpu-r2 --levels 1,2,4,8 --requests 60
python -m skymind.serve.parity --a http://127.0.0.1:8091 --b http://127.0.0.1:8094 --first 60
```

## LLM özeti: sıfır-örnek vs few-shot (yeni tohum 22, Faz 5 ek)
Önceki hatalara bakarak few-shot prompt yazdığım için aynı vakalarda değerlendirmek haksız olurdu; bu yüzden
**yeni tohumla (22) üretilmiş 80 METAR + 48 NOTAM** üzerinde iki prompt karşılaştırıldı (aynı metrik, aynı
modeller). Few-shot prompt: okuma kuralları (METAR rüzgâr/görüş/bulut/`M` eksi çözümü, NOTAM `A)`=havaalanı,
`B)/C)`=YYAAGGSSdd, Q satırı FIR) + 2 elle yazılmış örnek (değerlendirme vakalarıyla çakışmaz, testle doğrulandı).

| Model / prompt | METAR geri çağırma | METAR tüm alanlar | METAR uydurma sayı | NOTAM geri çağırma | NOTAM tüm alanlar | NOTAM uydurma sayı |
|---|---|---|---|---|---|---|
| 27B sıfır-örnek | 0.856 | 0.475 | 0.200 | 0.624 | 0.104 | 0.333 |
| **27B few-shot** | **0.968** | **0.850** | 0.188 | **0.979** | **0.917** | **0.042** |
| 4B sıfır-örnek | 0.657 | 0.087 | 0.537 | 0.493 | 0.021 | 0.604 |
| 4B few-shot | 0.899 | 0.525 | 0.412 | 0.880 | 0.521 | 0.062 |

* **NOTAM'da few-shot büyük fark yarattı** (27B: tarih/konum hataları neredeyse bitti, uydurma sayı %33 → %4).
  Bunlar format-bilgisi hatalarıydı; kısa bir açıklama ve iki örnek yetti.
* **METAR'da uydurma sayı neredeyse değişmedi (0.20 → 0.19):** kalan hatalar rakam okuma: `BKN002` → "2000 ft"
  (200 ft olmalı), `05033KT` → "3 kt", `M`/eksi işareti. Bu hatalar bir açıklamayla gitmiyor; sayı denetimi
  (faithfulness) bu yüzden hâlâ gerekli. 4B few-shot ile bile %41 oranında sayı uyduruyor.
* **Kural tabanlı özetleyici hâlâ tek sıfır-hatalı seçenek**, ama bu yapısal (metriklerle aynı ayrıştırıcı),
  özet kalitesi ve okunabilirliği ölçülmedi. Few-shot prompt kısaltma da getirdi (27B: ort. 268 → 186 karakter).
* Sınırlar: tek prompt varyantı, tek koşu, n=80/48, sıcaklık 0, iki Gemma modeli; prompt hata örneklerine bakarak
  yazıldı (yeni tohumla kısmen dengelendi ama aynı üretici dağılımı). Ham özetler `outputs/eval/summ-llm-seed22.json`.

```bash
uv run --no-sync python -m skymind.aviation.llm_summary --seed 22 --prompts zeroshot,fewshot --out outputs/eval/summ-llm-seed22.json
```

# Gold (insan doğrulamalı) alt kümede planlayıcı değerlendirmesi (Faz 3–4 ek)
`data/gold/gold.jsonl`: 300 örnek (plan 176, netleştirme 85, ret 39), test bölümünden yalnızca şablon üretimli
komutlar, proje sahibi tarafından 300/300 incelendi (veri kartı: `data/gold/README.md`). Eğitimle çakışma 0
(kimlik, komut metni, şablon grubu kontrol edildi). SFT (`sft-v1`) ve DPO v2 (`dpo-v2`, SFT üstüne), HF bf16, açgözlü
örnekleme, kısa sistem istemi. **Aynı 300 örnek, eşleşmiş karşılaştırma** (örnekleri yeniden örnekleyen bootstrap
%95 güven aralığı ve ayrışan çiftler üzerinde kesin McNemar).

| Ölçü | SFT | DPO v2 | Fark (DPO − SFT) [%95 GA] | Eşleşmiş test |
|---|---|---|---|---|
| Geçerli JSON | **0.983** | 0.953 | −0.030 [−0.053, −0.007] | McNemar p = 0.022 (SFT daha iyi: 11'e 2 örnek) |
| Tür doğruluğu | 0.903 | 0.907 | +0.003 [−0.027, +0.033] | p = 1.0 |
| Alan doğruluğu (plan) | 0.857 | 0.880 | +0.024 [−0.018, +0.069] | GA 0'ı içeriyor |
| Gereksiz soru ↓ (plan örnekleri) | 0.131 | **0.068** | −0.062 [−0.099, −0.029] | p = 0.001 (DPO daha iyi: 11'e 0 örnek) |
| Uydurma ↓ (netleştirme/ret örnekleri) | 0.000 | 0.000 | 0 | n = 124 |

Kategori bazında tür doğruluğu (SFT / DPO): plan 0.858 / 0.881, netleştirme **0.988 / 0.918**, ret 0.923 / **1.000**.
Hata dökümü: SFT 23 gereksiz soru (plan yerine soru) + 5 şema-geçersiz + 1 yanlış ret; DPO 12 gereksiz soru +
**14 şema-geçersiz** + 2 yanlış tür (netleştirme yerine başka).

## Okuma ve uyarılar
1. **Bulgu önceki ölçümle aynı yönde:** DPO v2 gereksiz soruyu yaklaşık yarıya indiriyor (23 → 12 / 176 plan örneği),
   karşılığında şemadan geçmeyen çıktı artıyor (5 → 14 / 300) ve netleştirme kategorisinde tür doğruluğu düşüyor
   (0.988 → 0.918). Alan doğruluğundaki +0.024 güven aralığı sıfırı içerdiği için **kanıtlanmış bir üstünlük değil**.
2. **Çoklu karşılaştırma:** 5 ölçü sınandı, düzeltme yapılmadı. Geçerlilik farkı (p = 0.022) Bonferroni
   eşiğinde (0.01) anlamlı kalmaz; gereksiz-soru farkı (p = 0.001) kalır.
3. **Bu küme test bölümünden daha kolay:** yalnızca şablon komutlar, paraphrase yok. Uydurma oranı burada 0.000, test
   bölümünde (400) SFT 0.016, DPO 0.011 idi; sayılar doğrudan karşılaştırılmamalı. Alan doğruluğu gold'da 0.857 / 0.880,
   test'te 0.842 / 0.850.
4. **Yeni bir genelleme kanıtı değil:** gold, test bölümünün alt kümesi ve aynı şablon üreticisinden. İnsan doğrulaması
   etiket hatası olasılığını azaltır (olsaydı iki modeli de aynı etkilerdi); dağılım kayması ölçmez.
5. **Tek eğitim tohumu** (SFT ve DPO): fark, bu iki eğitilmiş modelin farkıdır, yöntemlerin genel farkı değil.
6. **Gri alanlar** (veri kartındaki 23 kart: "havada kal"+yer yok, ret önceliği, "geri dön", yön-var-mesafe-yok):
   etiket hatası değil şablon varsayımı; modellerin bu kartlardaki davranışı ayrıca analiz edilmedi.
7. Değerlendirme HF bf16 üzerinde; Q5_K_M GGUF servisinde sayılar küçük farklarla değişebilir (önceki GGUF ölçümü ~aynı).

```bash
uv run --extra train python -m skymind.planner.evaluate --name sft-v1-gold --adapter outputs/sft-v1/adapter --data data/gold/gold.jsonl --limit 0
uv run --extra train python -m skymind.planner.evaluate --name dpo-v2-gold --adapter outputs/dpo-v2/adapter --base-adapter outputs/sft-v1/adapter --data data/gold/gold.jsonl --limit 0
```
