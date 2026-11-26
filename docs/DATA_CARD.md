# Veri Kartı — SkyMind komut→plan seti (v0.1)

**Amaç:** Türkçe İHA operatör komutlarından doğrulanmış JSON görev planı (veya netleştirme sorusu / ret) üretmeyi eğitmek ve ölçmek.
**Durum:** Araştırma prototipi. Tamamı sentetiktir; gerçek operasyonel veri yoktur.

## Üretim
1. **Şablon/gramer motoru** (`skymind/data/generator.py`): önce plan örneklenir, cümle sonra yazılır; etiket hatası tanım gereği yoktur. 6000 benzersiz komut, tohum `1`.
2. **LLM paraphrase** (`skymind/data/paraphrase.py`): `google/gemma-3-27b-it` (OpenRouter), komut başına 3 yeniden yazım. Kural tabanlı doğrulayıcı sayı/birim, yön, yer adı, "yarıçap→çap" kaymasını ve irtifa ifadesi kaybını eler. Üretilen 9.832 geçerli paraphrase `data/synth/paraphrases.jsonl` içinde (komutların %73'ü en az bir geçerli paraphrase aldı).
3. **Bölme** (`skymind/data/build.py`): sızıntısız. Aynı altın plan aynı split'te; paraphrase'ler orijinaliyle aynı split'te; aynı metin iki split'te yok.

## Kategoriler
| Kategori | Oran | Altın çıktı | `rejected` (DPO) |
|---|---|---|---|
| `plan` | ~60% | doğrulanmış aksiyon listesi | yok |
| `clarification` | ~25% | eksik alanları soran soru + `missing` | eksik alanlar uydurulmuş plan |
| `refusal` | ~15% | silah / güvensiz / kapsam dışı / sınır dışı değer | uydurulmuş makul plan |

## Boyut
| Split | Örnek |
|---|---|
| train | 11.670 |
| val | 1.488 |
| test | 1.477 |
| altın aday (test'ten, şablon kaynaklı) | 300 |

## Bilinen sınırlar
- Şablonlar sınırlı bir sözdizimi kapsar (8 aksiyon, 11 adlandırılmış yer, 8 yön). Gerçek operatör diline genelleme **ölçülmeli**, varsayılmamalı.
- Paraphrase doğrulayıcı muhafazakârdır: sayıyı başka birimle yazan (ör. "1 km" → "1000 m") çıktıları eler.
- `rejected` planlar kural tabanlıdır; "iyi" DPO çiftleri için çeşitlilik sınırlıdır.
- Altın aday seti (300) proje sahibi tarafından **gözden geçirildi (300/300 doğru)**, ama bağımsız değil (aynı üretici) ve cümleler şablon izi taşıyor ("Operatör:" öneki, "bildir/teyit et/acil" ekleri, 115 komutta). 23 gri alan kartı için `data/gold/README.md`.

## Veri hatası ve onarımı (v0.1.1)
İlk sürümde paraphrase ayrıştırıcısı `{style, command}` nesnelerini metne çevirmiş, **485 satır bozulmuştu**. Ayrıştırıcı düzeltildi
(`item_text`), veri `skymind/data/repair.py` ile onarıldı, bölümler yeniden üretildi (güncel boyutlar yukarıda). Planlayıcılar temiz test
kümesinde yeniden ölçüldü; bulgular değişmedi (alan doğruluğu SFT 0,842, DPO 0,850). Altın aday dosyası etkilenmemişti.

## Yeniden üretme
```bash
uv run python -m skymind.data.run_paraphrase --n 6000 --seed 1   # yalnızca paraphrase'i yeniden üretmek için (ücretli API)
uv run python -m skymind.data.build --n 6000 --seed 1            # repodaki paraphrases.jsonl ile ücretsiz
```
