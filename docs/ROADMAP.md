# Yol Haritası

Kısıtlar: **ücretsiz** araçlar, RTX 4060 Laptop (8 GB VRAM), 32 GB RAM, yalnızca açık kaynak/sentetik veri.

| Faz | Hafta | İçerik | Bitti ölçütü |
|---|---|---|---|
| 0 | 1 | İskele, plan şeması, eval çerçevesi, MLflow, CI | pytest + CI yeşil |
| 1 | 1–2 | Sentetik veri (şablon + yerel Qwen + OpenRouter ücretsiz), NER seti, altın set (~300) | veri kartı, şema geçerliliği %100 |
| 2 | 3 | Intent + NER (BERTurk), baseline'lar | F1 tablosu, MLflow |
| 3 | 4 | Planlayıcı SFT (Qwen2.5-1.5B + QLoRA) | JSON geçerlilik, alan doğruluğu, halüsinasyon oranı |
| 4 | 5 | DPO | SFT vs DPO halüsinasyon oranı |
| 5 | 6 | RAG + özetleme (NOTAM/METAR) | retrieval + faithfulness tablosu |
| 6 | 7 | Optimizasyon (quant, distill, prune, ONNX) | doğruluk-gecikme-bellek tablosu |
| 7 | 8–9 | C++ servis, benchmark, A/B, izleme, Podman/K8s | C++ vs Python p50/p95/p99, tek komut demo |
| 8 | 10 | README, diyagram, demo | tekrar üretilebilir README |

## Sprint 1 (Faz 0 + Faz 1 başı)
- [x] git init, repo iskelesi
- [x] Plan şeması (Pydantic + JSON Schema) ve testleri
- [x] Eval metrikleri (JSON geçerlilik, alan doğruluğu, halüsinasyon)
- [x] MLflow yerel kurulum + `log_run` yardımcısı
- [x] Sentetik veri hattı (şablon + paraphrase + doğrulama + sızıntısız split)
- [~] Altın test seti: 300 aday üretildi (data/gold), elle gözden geçirme bekliyor
- [x] ADR-001 (model ve araç seçimleri)

## Sprint 2 (Faz 3–4)
- [x] Zero-shot ve trivial baseline'lar
- [x] SFT (QLoRA, Qwen2.5-1.5B): geçerli JSON 0.99, alan doğruluğu 0.84
- [x] DPO v1 (saf): şemayı bozdu, bulgu olarak belgelendi
- [x] DPO v2 (+SFT kaybı): yapıyı korudu, SFT'yi net geçmedi
- [x] Sonuç tablosu ve uyarılar: `docs/RESULTS.md`
- [ ] Çoklu tohum / güven aralığı (isteğe bağlı)

## Sprint 3 (Faz 6)
- [x] Adaptör birleştirme, GGUF dönüşümü, F16/Q8/Q5/Q4/Q3 nicemleme, doğruluk-boyut-gecikme tablosu
- [x] ONNX dışa aktarma (fp32/int4) + ONNX Runtime CPU ölçümü
- [x] Damıtma ve budama, niyet sınıflandırıcıda (BERTurk 12L → 6/4/2 katman; magnitude ve kafa budama), temiz yöntemle, 2 tohum: gecikme ve doğruluk `docs/RESULTS.md` Faz 5b
- [ ] LLM damıtma (1.5B → 0.5B): yapılmadı, kapsam sınıflandırıcıyla sınırlı

## Sprint 4 (Faz 7)
- [x] C++ servis: llama.cpp motoru, HTTP uçları, sunucu tarafı şema doğrulaması, Prometheus metrikleri
- [x] GoogleTest (15 test), Pydantic ile eşlik testi (263 vaka); doğrulayıcıya mutasyon testi yapıldı
- [x] Python baseline servisi (FastAPI + HF) ve yük testi aracı yazıldı
- [x] Yük testi ölçümü: C++ vs Python p50/p95/p99 (docs/RESULTS.md, Faz 7)
- [ ] ~~ONNX Runtime motoru (C++ `InferenceEngine`)~~ ertelendi: onnxruntime-genai C başlıkları kaynak depodan alınamadı (microsoft/onnxruntime-genai 404), imzaları tahminle bağlamak riskli. ONNX ölçümü Python tarafında var (docs/RESULTS.md)
- [x] A/B ağ geçidi (deterministik ağırlıklı atama, z-testi), Prometheus + Grafana yapılandırması, alarmlar
- [x] A/B için DPO v2 GGUF üretimi ve gerçek iki arka uçla deneme (anlamlı fark yok)
- [x] Containerfile'lar, compose.yaml, K8s manifest'leri (kustomize) yazıldı ve yapısal olarak test edildi
- [x] Podman (rootless, WSL2) ile imajlar derlendi, compose yığını çalıştırıldı: gateway → C++ konteyner → geçerli plan, Prometheus hedefleri/alarmlar, Grafana pano sorguları doğrulandı; Linux'ta C++ testleri 15/15
- [ ] Kubernetes kümesinde çalıştırma ve kubeconform/promtool doğrulaması (küme yok, yapılmadı); GitHub Actions'ta CI çalıştırma (uzak depo yok)

## Sprint 3b (Faz 2)
- [x] Niyet (10 sınıf) ve NER (8 tür): TF-IDF/kural taban çizgileri, BERTurk ince ayar
- [x] Zayıf NER etiketlerinin döngüselliği tespit edildi; bağımsız 42+ cümlelik elle set (Claude yazdı, sahibi gözden geçirmedi)
- [x] Paraphrase veri hatası bulundu, düzeltildi, planlayıcılar yeniden ölçüldü (bulgu değişmedi)
- [ ] NER elle setini proje sahibinin gözden geçirmesi

## Sprint 5 (Faz 5)
- [x] Kurgusal kılavuz + soru seti, BM25 (+F5 kök), e5-small dense, RRF hibrit, 8 tohumlu ölçüm
- [x] Cümle bazlı atıf (çıkarımsal yanıtlayıcı) ve atıf doğrulama
- [x] METAR/NOTAM ayrıştırıcı, Türkçe özet, sentetik üretici, sayısal sadakat denetimi
- [x] Üretken (LLM) yanıtlayıcı + atıf/vazgeçme ölçümü (gemma-3-27b, OpenRouter; R@3 darboğaz, atıf geçerliliği ≠ desteklenme)
- [x] LLM tabanlı NOTAM/METAR özeti (gemma-3 27B/4B) ve kural tabanlıyla sadakat karşılaştırması: LLM tarih/rakam hataları yapıyor (docs/RESULTS.md)
- [x] Few-shot prompt ile LLM özeti yeni tohumda yeniden ölçüldü (NOTAM düzeldi, METAR rakam hataları sürdü)
- [x] llama-cpp-python taban çizgisi (aynı GGUF ve aynı llama.cpp DLL'leri), çıktı eşitliği ve GPU belleği: dil payı ~%6–7
- [ ] Reranker denemesi
