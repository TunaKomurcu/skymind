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
