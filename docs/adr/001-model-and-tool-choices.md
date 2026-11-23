# ADR-001: Model ve araç seçimleri

Tarih: 2026-11-23 · Durum: kabul edildi

## Bağlam
Ücretli altyapı yok; RTX 4060 Laptop (8 GB VRAM), 32 GB RAM, ~84 GB disk. OpenRouter anahtarında 5 USD limit var; proje için sert tavan **4 USD**.

## Kararlar
| Konu | Karar | Neden / alternatif |
|---|---|---|
| Veri LLM'i | `google/gemma-3-27b-it` (ücretli, ~0,08/0,16 USD/1M token) | `:free` modeller denendi: Gemma-4 ücretsiz sürümü paylaşımlı havuzda 429 verdi, Nemotron reasoning token'larıyla çıktıyı yedi. Gemma-3-27B reasoning yapmaz, Türkçesi iyi; 6000 komutluk koşu ~0,18 USD |
| Maliyet koruması | İstemci tarafı sert tavan + defter + önbellek | Anahtar limiti istemciden değiştirilemez; en kötü durum maliyeti istekten **önce** hesaplanır |
| Plan şeması | Pydantic v2 + JSON Schema, `extra="forbid"`, dar sınırlar | Sınır dışı değer "uydurma" sayılır; constrained decoding için aynı şema kullanılır |
| Halüsinasyon tanımı | `fabrication_rate` (eksik/ret durumunda plan üretme) + `invalid_action_rate` | Ölçülebilir ve etiketten türetilebilir |
| Deney takibi | MLflow (SQLite, yerel) | Tam `mlflow` gerekti (`mlflow-skinny` SQLite desteklemiyor); model registry için de lazım |
| Ortam | uv + ruff + mypy + pytest | Hızlı, tek lock dosyası |
| Planlayıcı (sonraki faz) | Qwen2.5-1.5B-Instruct + QLoRA | 8 GB VRAM'e sığar; DPO için adaptör kapatılarak referans model |
| Konteyner | Podman (Faz 7) | Docker kurulu değil; kullanıcı tercihi |

## Sonuçlar
- Veri tamamen sentetik; genelleme iddiası ölçümle (altın set + ek sözdizimi) desteklenmeli.
- Gemma-3 paraphrase'lerinde etiket kayması gözlendi (çap/yarıçap, uydurma çağrı adı); doğrulayıcı bu yüzden zorunlu.
