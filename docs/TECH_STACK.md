# Kullanılan teknolojiler: ne, nerede, ne işe yarıyor, neden

Bu belge projede kullanılan her aracı **ne işe yaradığıyla** açıklar; kısaltmalar ve kavramlar için kısa açıklamalar
içerir. Sürümler `uv.lock` dosyasından alınmıştır. Başlangıç planında yer alıp **kullanılmayan** araçlar en sonda ayrı listelenir.

## 1. Geliştirme ortamı ve kalite araçları

| Araç | Ne işe yarar | Projede nerede | Neden |
|---|---|---|---|
| **Python 3.11+** (3.12 ile çalışıldı) | Ana dil: veri, eğitim, değerlendirme, servisler | `skymind/` paketi | ML ekosistemi |
| **C++17** | Hızlı, bağımlılığı az çıkarım servisi | `cpp/` | Çıkarımda Python yükü yok, düşük gecikme |
| **uv** 0.x | Paket ve sanal ortam yöneticisi, `uv.lock` ile tekrar üretilebilir kurulum | `pyproject.toml`, `uv.lock` | pip'ten hızlı, tek kilit dosyası |
| **ruff** 0.17 | Lint (hata/üslup denetimi) ve biçimlendirme | CI + yerel | Tek araçta flake8+isort+black |
| **mypy** 2.4 (strict) | Statik tip denetimi | CI + yerel | Plan şeması gibi veri sözleşmelerinde hata erken yakalanır |
| **pytest** 9 + pytest-cov | Python birim/entegrasyon testleri | `tests/` (171 test geçti, 1 atlandı; kapsam %65) | Her modül testli |
| **GoogleTest** 1.15 | C++ birim testleri | `cpp/tests/` (15 test) | C++ için standart |
| **CMake** + **Ninja** | C++ derleme sistemi | `cpp/CMakeLists.txt` | Windows (MSVC) ve Linux (gcc) aynı betik |
| **GitHub Actions** | Her push'ta lint + tip + test + C++ derleme | `.github/workflows/ci.yml` | Otomatik kalite kapısı |

## 2. Veri

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **Pydantic v2** + JSON Schema | Plan şemasını tanımlar ve her çıktıyı doğrular (`extra="forbid"`, dar sınırlar). Şemaya uymayan çıktı "geçersiz" sayılır; sınır dışı değer "uydurma" sayılır | `skymind/schemas/plan.py` (tek doğruluk kaynağı; C++ doğrulayıcının eşi) |
| **Şablon/gramer motoru** (kendi yazdığımız) | Önce plan örneklenir, cümle sonra yazılır: etiket hatası tanım gereği olamaz | `skymind/data/generator.py` |
| **OpenRouter + Gemma-3-27B-it** | Şablon komutları serbest Türkçeye yeniden yazar (paraphrase). Kural tabanlı doğrulayıcı sayı/yön/yer kaymalarını eler | `skymind/data/paraphrase.py`, `skymind/llm/openrouter.py` (sert bütçe tavanı + önbellek) |
| **Yerel web arayüzleri** (stdlib `http.server` + tek dosya HTML) | Altın küme (300) ve NER seti (40) için hızlı insan incelemesi: klavyeyle doğru/hatalı, geri al, harita | `skymind/data/review_web.py`, `ner_review.py` |

## 3. Klasik NLP (niyet sınıflandırma, NER)

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **BERTurk** (`dbmdz/bert-base-turkish-cased`) | Türkçe için önceden eğitilmiş BERT. İnce ayarla niyet (10 sınıf) ve varlık (8 tür) çıkarımı | `skymind/nlp/train.py` |
| **Hugging Face Transformers** | Model yükleme/eğitim/çıkarım kütüphanesi | `skymind/nlp/`, `planner/` |
| **scikit-learn** | TF-IDF + lojistik regresyon taban çizgisi (BERTurk'ü neyle kıyasladığımızı gösterir) | `skymind/nlp/baselines.py` |
| **Kural tabanlı etiketleyici** | Regex ile birim/yön/yer yakalar; NER'in hem zayıf etiket kaynağı hem taban çizgisi | `skymind/nlp/ner_data.py` |
| **Varlık düzeyi F1** (conlleval kurallarıyla, kendi uygulama) | NER'i kelime değil, tam varlık eşleşmesiyle puanlar | `skymind/nlp/metrics.py` |

## 4. Planlayıcı LLM: ince ayar ve hizalama

| Araç / kavram | Ne işe yarar | Projede nerede |
|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 8 GB GPU'ya sığan küçük açık dil modeli; Türkçe komut → JSON plan | `skymind/planner/` |
| **QLoRA** (PEFT/LoRA + **bitsandbytes** 4-bit) | Tüm modeli değil, küçük "adaptör" matrislerini eğitir; temel model 4-bit sıkıştırılmış tutulur → 8 GB'a sığar | `planner/sft.py` (rank 16, lr 2e-4) |
| **TRL** (`SFTTrainer`) | Denetimli ince ayar (SFT): doğru yanıtı taklit etmeyi öğretir | `planner/sft.py` |
| **DPO** (TRL `DPOTrainer`) | Doğrudan tercih eniyilemesi: "geçerli plan/netleştirme" yanıtını "uydurma plan"a tercih etmeyi öğretir. v2'de SFT kaybı eklenerek şema bozulması önlendi | `planner/dpo.py`, `planner/preferences.py` |
| **Accelerate**, **Datasets** | Eğitim altyapısı, veri yükleme | eğitim betikleri |
| **MLflow** 3.18 (SQLite) | Deney takibi: her koşunun parametre ve metrikleri tek tabloda | `skymind/mlops/tracking.py` |

Ölçüler: JSON geçerlilik, alan doğruluğu, **uydurma oranı** (eksik/belirsiz/yasak komutta plan üretme), gereksiz soru oranı
(`skymind/eval/metrics.py`).

## 5. RAG, havacılık metinleri

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **BM25** (kendi uygulama, 5 harfli kök kırpma) | Anahtar kelime tabanlı arama | `skymind/rag/retrieval.py` |
| **multilingual-e5-small** (`intfloat`) | Çok dilli gömme modeli: soru ve parçaları vektöre çevirir (anlamsal arama) | `rag/retrieval.py` (`HFEmbedder`) |
| **RRF** (Reciprocal Rank Fusion) | BM25 ve gömme sıralamalarını birleştirir (hibrit arama) | `rag/retrieval.py` |
| **Cümle bazlı atıf** | Yanıtın her cümlesi `[parça, cK]` ile kaynağa bağlanır; sahte atıflar otomatik yakalanır | `rag/answer.py` |
| **Gemma-3-27B (OpenRouter)** | Üretken yanıtlayıcı + "bilgi yok" davranışı değerlendirmesi | `rag/generate.py` |
| **METAR/NOTAM ayrıştırıcı + özetleyici** (regex, kural tabanlı) | Havacılık mesajlarını yapıya çevirip Türkçe özetler; her sayı kaynaktan türetilebilir | `skymind/aviation/` |
| **Sayısal sadakat denetleyicisi** | Özetteki her sayının kaynakta olup olmadığını kontrol eder (uydurma sayı yakalama) | `aviation/faithfulness.py` |

## 6. Çıkarım optimizasyonu

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **llama.cpp** (b12655) + **GGUF** | CPU/GPU'da verimli LLM çıkarımı ve model biçimi; F16/Q8_0/Q5_K_M/Q4_K_M/Q3_K_M **nicemleme** (ağırlıkları az bitle saklayıp boyutu küçültme) | `scripts/make_gguf.ps1`, `optimize/eval_gguf.py` |
| **ONNX Runtime** + **onnxruntime-genai** | Modeli taşınabilir biçime çevirip CPU'da int4 çalıştırma | `optimize/eval_onnx.py` |
| **PyTorch ile elle damıtma/budama** | **Damıtma**: büyük modelin bilgisini küçüğe aktarma; **budama**: ağırlık ya da dikkat kafalarını silme. Katman alt kümesi öğrenciler, magnitude ve yapısal kafa budama | `skymind/nlp/compress.py`, `compress_run.py` (transformers 5 `prune_heads` kaldırdığı için kafa budama kendi yazıldı, eşdeğerlik testi var) |

## 7. Servisler ve LLMOps

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **C++ servis** (llama.cpp C API + **cpp-httplib** + **nlohmann/json**) | HTTP üzerinden `POST /v1/plan`; sunucu tarafı şema doğrulaması; Prometheus metrikleri | `cpp/` |
| **FastAPI** + **uvicorn** | Python taban çizgisi servisi ve A/B ağ geçidi | `skymind/serve/` |
| **llama-cpp-python** | Aynı GGUF'u Python'dan çalıştırıp C++ ile karşılaştırma ("dil payı" ölçümü) | `serve/engines.py` |
| **A/B ağ geçidi** (kendi, SHA-256 deterministik atama, iki oranlı z-testi) | İki model sürümüne trafiği ağırlıklı böler, geçerlilik farkının anlamlı olup olmadığını söyler | `serve/gateway.py` |
| **Prometheus** v2.55 + **Grafana** 11.3 | Metrik toplama, alarm kuralları ve pano | `deploy/prometheus/`, `deploy/grafana/` |
| **Yük testi aracı** (`httpx` asyncio, kendi) | p50/p95/p99, istek/sn, token/sn ölçer | `serve/loadtest.py`, `serve/parity.py` |

## 8. Dağıtım

| Araç | Ne işe yarar | Projede nerede |
|---|---|---|
| **Podman** 4.9 (rootless, WSL2) + **podman-compose** | Docker'sız konteyner derleme/çalıştırma; tek komutla tüm yığın | `deploy/containers/`, `deploy/compose.yaml` |
| **Kubernetes manifest'leri** + **kustomize** + HPA | Aynı yığının küme sürümü (yazıldı, yapısal test edildi, **kümeye uygulanmadı**) | `deploy/k8s/`, `deploy/kustomization.yaml` |

## 9. Donanım ve ortam
RTX 4060 Laptop (8 GB VRAM), 32 GB RAM, Windows 11; WSL2 Ubuntu 24.04 (Linux derleme/Podman). Dış hizmet: yalnızca
OpenRouter (veri üretimi ve LLM karşılaştırmaları, toplam ~0,22 USD, 4 USD sert tavan).

## 10. Planlanıp KULLANILMAYANLAR (dürüst liste)

| Önerilen | Durum | Neden |
|---|---|---|
| Constrained decoding (JSON grammar) | Yapılmadı | "Gerekirse" maddesiydi; şema doğrulaması ve DPO yetti |
| Qdrant / FAISS | Kullanılmadı | Küçük korpusta numpy kosinüs benzerliği yeterli |
| RAGAS | Kullanılmadı | Eşdeğer ölçüler kendi yazıldı (atıf geçerliliği, sayı sadakati, "bilgi yok" oranı) |
| AWQ / GPTQ / INT8 | Yapılmadı | Yalnızca GGUF ve ONNX int4 karşılaştırıldı |
| ONNX Runtime **C++** API | Yapılmadı | onnxruntime-genai C başlıkları kaynak depodan alınamadı (404); ONNX ölçümü Python'da |
| gRPC | Yapılmadı | HTTP yeterli |
| vLLM | Kullanılmadı | llama.cpp tercih edildi (8 GB GPU, tek bağlam) |
| MLflow **model registry** | Kullanılmadı | Yalnızca deney takibi |
| W&B | Kullanılmadı | MLflow seçildi |
| Docker | Kullanılmadı | Podman kullanıldı |
| k3s / kind ile gerçek Kubernetes | Yapılmadı | Manifest'ler yalnızca yapısal testten geçti |
| ARM / Jetson ölçümü | Yapılmadı | Donanım yok |
| TR↔EN çeviri | Yapılmadı | Opsiyoneldi |
