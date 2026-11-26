# Altın küme ve NER seti: insan inceleme rehberi

**Amaç:** 300 şablon üretimli aday komutu insan gözüyle doğrulayıp `data/gold/gold.jsonl` üretmek (ve 40 cümlelik NER elle
setini aynı yöntemle doğrulamak). Bu olmadan planlayıcı sonuçları yalnızca sentetik test kümesinde kalır.

**Durum:** tamamlandı. Altın küme 300/300 ve NER seti 40/40 incelendi, hepsi doğru bulundu (kayıtlar
`data/gold/gold_reviewed.jsonl` ve `data/gold/ner_manual_reviewed.jsonl`). Sonuçlar ve gri alanlar: `data/gold/README.md`.

## Ön-inceleme ve kökeni
`data/gold/claude_review.jsonl`: Claude 300 adayın tamamını okudu (komut + çıktı), hata bulmadı; 23 kartı "tartışmalı"
olarak işaretledi (haritada sarı çerçeve, `T` tuşuyla atlanır) ve kartın altında gerekçesini gösterir. Bu dosya **insan
kararından ayrıdır**; `gold_reviewed.jsonl` yalnızca inceleyenin kararlarını içerir.

Bilinen sınırlar:
* **Bağımsızlık sınırlı:** ön-inceleyen Claude, şablonları ve altın çıktıları üreten sistemle aynı soydandır; ortak bir kör nokta
  olabilir. "Hepsi doğru" bulgusu bu yüzden zayıf kanıttır, asıl değer insan incelemesinden gelir.
* **Otomasyon yanlılığı:** ön-incelemenin "✓ doğru" kutusunu görünce otomatik onaylamak kolaydır; önce komuta ve çıktıya, sonra
  kutuya bakılmalıdır. İlk 25 kart ön-inceleme görülmeden işaretlendi (Claude ile 25/25 uyum).

## Web arayüzü (önerilen, kart başına ~10 sn)
```bash
uv run python -m skymind.data.review --web       # altın küme, http://127.0.0.1:8765
uv run python -m skymind.data.ner_review --web   # NER seti,   http://127.0.0.1:8766
```
Sunucu yalnızca yerel makineden erişilebilir (127.0.0.1), kararlar her tıklamada diske yazılır; sekme kapatılıp sonra aynı komutla
devam edilebilir. Her kart komutu, sistemin vermesi gereken çıktıyı (okunur Türkçe, sayılar vurgulu) ve ön-inceleme kutusunu gösterir.
Üstteki haritada her kart bir kutudur (yeşil doğru, kırmızı hatalı, gri kararsız, sarı çerçeve tartışmalı); tıklayınca o karta gidilir.

| Tuş | Anlamı |
|---|---|
| `D` / `Enter` | doğru |
| `N` | hatalı (kümeden çıkar) |
| `←` / `→` | önceki / sonraki kart (karar vermez; geçmiş kararlar düzeltilebilir) |
| `Z` | oturumdaki son kararı geri al |
| `E` | komutu (altın küme) ya da etiketleri (NER) düzelt |
| `J` | ham JSON (altın küme) |
| `T` | sonraki tartışmalı kart |
| `1`–`5` | "hatalı" nedeni (isteğe bağlı) |

"Toplu düzeltme" bölümü, seçili karttan sonraki tüm kararları siler (silmeden önce yedek `gold_reviewed.bak.jsonl` alınır).

## Terminal aracı (alternatif)
```bash
uv run python -m skymind.data.review      # etkileşimli terminalde: Enter doğru, n hatalı, c komut, g JSON, s atla, b geri, q çık
```
Web arayüzüyle aynı dosyayı kullanır; biri bırakılıp diğerine geçilebilir.

## Karar ölçütleri
Her aday için iki soru:
1. **Komut** doğal, anlaşılır bir Türkçe operatör komutu mu?
2. **Altın çıktı** doğru mu?
   * `plan`: komuttaki her sayı, yön ve yer çıktıda doğru; komutta olmayan hiçbir şey eklenmemiş; adım sırası komutla uyumlu.
   * `clarification`: komutta gerçekten zorunlu bir bilgi (hedef, irtifa, hız gibi) eksik ve soru makul.
   * `refusal`: komut gerçekten güvensiz ya da kapsam dışı (silah, sınır ihlali, havalimanı pisti vb.).

**Netleştirme sorularında ölçüt:** *"Komutu uygulamak için komutta olmayan bir değer, hedef ya da yön uydurmak gerekir mi?"* Evet
ise soru **doğru**, hayır ise **gereksiz** (hatalı). Örnek: "Hızı artır" komutuna "Hız bilgisini eksik aldım, netleştirir misiniz?"
doğru çıktıdır (hız sayısı yok); "Hızı 20 m/s yap" komutuna aynı soru gelirse hatalıdır.

**Hatalı nedeni (isteğe bağlı):** `1` komut yapay/anlamsız · `2` çıktı komuta uymuyor (sayı/yön/yer) · `3` gereksiz soru (bilgi
komutta var) · `4` soru gerekliydi (bilgi eksik ama plan üretilmiş) · `5` ret yersiz ya da ret gerekliydi. Dağılım `--stats` ile
görülür.

**NER seti için ölçüt:** (1) her etiketli varlık doğru türde mi, (2) sınırlar doğru mu (birim varlığın içinde, ekler yer adıyla
birlikte, hızda "saatte" dahil), (3) etiket dışında kalmış sayı/yer/yön/süre var mı, (4) etiketlenmemesi gereken bir şey etiketlenmiş mi.
Düzeltmede cümlenin kendisi değiştirilemez, yalnızca etiketler (`[metin|ETİKET]`).

## İlerleme, dışa aktarma ve sonraki adım
```bash
uv run python -m skymind.data.review --stats      # toplam / ok / edited / bad / pending ve hatalı nedenleri
uv run python -m skymind.data.review --export     # ok + edited olanlardan data/gold/gold.jsonl üretir
uv run python -m skymind.data.ner_review --apply  # NER düzeltmelerini ner_manual.txt'ye yazar (yedekli)
```
`gold.jsonl` ile SFT ve DPO yeniden ölçüldü (`docs/RESULTS.md`, "Gold alt kümede planlayıcı değerlendirmesi"); atılan aday oranı
şablon veri kalitesi için ayrı bir göstergedir.
