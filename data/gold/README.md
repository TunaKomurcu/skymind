# Altın küme (gold): veri kartı

## Ne bu
300 örneklik, **insan tarafından doğrulanmış** değerlendirme alt kümesi (`gold.jsonl`). Adaylar
`gold_candidates.jsonl` dosyasındadır (`skymind.data.build.gold_candidates`).

| | |
|---|---|
| Boyut | 300 (plan 176, netleştirme 85, ret 39) |
| Kaynak | **test bölümünden**, yalnızca *şablon üretimli* komutlar (paraphrase yok), kategori oranı korunarak örneklendi |
| Eğitimde kullanıldı mı | Hayır: 300 kimliğin 300'ü test bölümünde, 0'ı train/val'de; komut metni ve şablon-grubu çakışması da 0 (kontrol edildi) |
| İnceleyen | proje sahibi, web arayüzüyle (`python -m skymind.data.review --web`), 300/300 kart, hepsi "doğru" |
| Ön-inceleme | Claude, 300 adayın tamamını elle okudu (`claude_review.jsonl`), hata bulmadı, 23 kartı tartışmalı işaretledi |
| Dosyalar | `gold_candidates.jsonl` (aday), `gold_reviewed.jsonl` (insan kararları), `claude_review.jsonl` (ön-inceleme), `gold.jsonl` (doğrulananlar, değerlendirme biçimi) |

## Ne OLMADIĞI (sınırlar)
* **Bağımsız ya da gerçek dünya verisi değil.** Komutlar ve altın çıktılar aynı şablon üreticisinden gelir; insan
  incelemesi yalnızca "çıktı komuta uyuyor mu, komut anlaşılır mı" sorusunu doğrular. Dağılım, üslup ve kapsam
  eğitim verisiyle aynıdır.
* **Paraphrase içermez**, yani test bölümündeki serbest ifadeli komutlardan daha kolaydır; sonuçlar test bölümüyle
  doğrudan karşılaştırılmamalı.
* **Doğrulama tek kişilik** ve inceleyen Claude'un ön-incelemesini görerek çalıştı (otomasyon yanlılığı riski).
  İlk 25 kart ön-inceleme görülmeden işaretlendi: Claude ile 25/25 uyum. Ön-inceleyen, şablonları üreten
  sistemle aynı soydan: ortak kör nokta olasılığı var.
* "300/300 doğru" bulgusu, hata oranının sıfır olduğunu değil, **bu inceleme yöntemiyle hata bulunamadığını** söyler.

## Bilinen gri alanlar (insan kararı: hepsi "doğru")
Aşağıdaki kartlar hem Claude'un hem proje sahibinin aklına takıldı; ikisi de **doğru** olarak bıraktı. Bunlar etiket
hatası değil, **şablon kurallarının varsayımlarıdır**. Rapor bu varsayımlarla birlikte okunmalıdır.
Sıra numaraları `skymind.data.review` karıştırma sırasına (tohum 0) göredir.

| Grup | Kart no | Varsayım ve alternatifi |
|---|---|---|
| A. "Havada kal / bekle / çember çiz", yer verilmemiş (9) | 3, 26, 49, 86, 169, 190, 209, 226, 263 | Bekleme için merkez zorunlu, bu yüzden netleştirme sorusu. Alternatif: "bulunduğun yer" varsay, plan üret. |
| B. Hem eksik bilgi hem sınır aşımı (6) | 8, 37, 38, 172, 216, 283 | "Sınır aşımı varsa ret, netleştirmeden önce gelir". Alternatif: önce eksik bilgiyi sor. |
| C. "X m yükseklikten git" (3, düşük risk) | 12, 24, 221 | İrtifa olarak yorumlandı (mesafe değil); "yükseklik/irtifa" sözcüğü bunu destekliyor. |
| D. "Geri dön" (4) | 199, 243, 244, 250 | Eve dönüş (`return_to_launch`) olarak yorumlandı. Alternatif: önceki konuma dönüş. |
| E. Yön var, mesafe yok (4) | 56, 107, 161, 180 | Şema mesafeyi isteğe bağlı bıraktığı için plan. Alternatif: "ne kadar?" diye sor. |

**Sistematik varsayım:** 300 komutun 115'inde "bildir / teyit et / acil" ekleri var; altın çıktılar bunları
**dolgu sözcüğü** sayıp yok sayar (şemada "raporla/onayla" eylemi yok). "Bildir" gerçek bir istekse çıktılar eksik kalır.

Not: ilk turda D ve E gruplarını Claude yalnızca ikinci/üçüncü taramada fark etti; yani ön-incelemesi tam kapsamlı
değildi, başka gri alanlar olabilir.

## Tekrar üretme
```bash
uv run python -m skymind.data.review --web      # incele (kararlar gold_reviewed.jsonl'a)
uv run python -m skymind.data.review --stats    # ilerleme
uv run python -m skymind.data.review --export   # gold.jsonl üret
uv run --extra train python -m skymind.planner.evaluate --name sft-v1-gold --adapter outputs/sft-v1/adapter \
    --data data/gold/gold.jsonl --limit 0
```

## NER elle sınama seti (`ner_manual.txt`, 40 cümle)
Altın kümeden ayrı, **farklı bir görevin** (varlık çıkarımı) bağımsız sınama seti: 20 KOLAY (şablona yakın) + 20 ZOR (şablon dışı: yazıyla sayılar,
`dk`/`ft`/`m/sn`, görülmemiş yer adları). Cümleleri ve `[metin|ETİKET]` işaretlemelerini Claude yazdı; proje sahibi 40/40 inceledi, hepsi doğru,
düzeltme yok (`ner_manual_reviewed.jsonl`). Claude 3 cümleyi etiket **sınırı** açısından gri işaretledi (`ner_claude_notes.json`). Ek mekanik
kontroller (etiket dışı sayı, birimsiz süre, kenar noktalama) 0 sorun buldu. Aynı yazarın ürünü olduğu için ortak kör nokta olasılığı sürer.
