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
- [ ] Sentetik veri hattı (şablon motoru → paraphrase → doğrulama)
- [ ] Altın test seti + pilot üretim (~500)
- [ ] ADR-001 (model ve araç seçimleri)
