# Altın küme inceleme rehberi

Amaç: 300 şablon-üretimli aday komutu insan gözüyle doğrulayıp `data/gold/gold.jsonl` üretmek. Bu olmadan
planlayıcı sonuçları yalnızca sentetik test kümesinde kalır (sentetik veri ve aynı üretici dağılımı).

## Claude ön-incelemesi ve kökeni (raporda belirtilecek)
`data/gold/claude_review.jsonl`: Claude 300 adayın **tamamını elle okudu** (komut + çıktı), hata bulmadı; 23 kartı
"tartışmalı" (sarı çerçeve, `T` tuşuyla atlanır), 3 kartı notla işaretledi. Arayüz bunu kartın altında gösterir.
Bu dosya **insan kararından ayrıdır**; `gold_reviewed.jsonl` yalnızca senin kararlarını içerir.

Dürüstlük notları (rapora girer):
* **Bağımsızlık sınırlı:** ön-inceleyen Claude, şablonları ve altın çıktıları üreten sistemle aynı soydan; ortak bir
  kör nokta olabilir. "Hepsi doğru" bulgusu bu yüzden **zayıf kanıttır**, gerçek değer insan gözünden gelir.
* **Otomasyon yanlılığı:** Claude'un "✓ doğru" kutusunu görünce körlemesine Enter'a basmak kolaydır. Önce komuta ve
  çıktıya bak, kutuya sonra. İlk 25 kartı Claude'un görüşünü görmeden kendin işaretledin (bağımsız): Claude ile
  25/25 uyum.
* Raporda şu şekilde yazılır: "300 adayın tamamı Claude tarafından ön-incelendi; insan (proje sahibi) bu ön-incelemeyi
  görerek N adayı doğruladı". N = senin incelediğin adet.

## Hızlı yol: web arayüzü (önerilen, ~10 sn/aday)

```bat
cd skymind
uv run python -m skymind.data.review --web
```

Tarayıcıda `http://127.0.0.1:8765` açılır (yalnızca bu bilgisayardan erişilir). Her aday tek kartta: komut,
sistemin vermesi gereken çıktı (okunur Türkçe, sayılar sarıyla vurgulu). **Klavyeyle:**
`→` / `Enter` / `D` doğru, `←` / `N` hatalı, `S` atla, `Z` geri al, `E` komutu düzelt, `J` ham JSON.
Her kararda otomatik diske yazılır; sekmeyi kapatıp sonra aynı komutla devam edebilirsin. Bitince terminalde
`Ctrl+C` ile sunucuyu kapat.

## Alternatif: terminal aracı (ayrı bir terminalde; Claude Code içindeki `!` istemi etkileşimli girdiyi desteklemez)

```bat
cd skymind
uv run python -m skymind.data.review
```

Web arayüzüyle terminal aracı **aynı dosyayı** (`gold_reviewed.jsonl`) kullanır, birini bırakıp diğerine geçebilirsin.
İlerlemen her karardan sonra `data/gold/gold_reviewed.jsonl` dosyasına yazılır. İstediğin an `q` ile çık, aynı
komutla kaldığın yerden devam edersin. Adaylar sabit bir sırayla karıştırılmıştır: **ilk N aday her zaman
kategorilere dengeli dağılmış bir örnektir**, yani hepsini bitirmesen de (örn. ilk 100) temsilî bir küme çıkar.

## Her aday için iki soru
1. **Komut** doğal, anlaşılır bir Türkçe operatör komutu mu? (Değilse `c` ile düzelt ya da `n` ile at.)
2. **Altın çıktı** doğru mu?
   * `plan`: komuttaki her sayı, yön ve yer çıktıda doğru; komutta olmayan hiçbir şey eklenmemiş. Adım sırası komutla uyumlu.
   * `clarification`: komutta gerçekten zorunlu bir bilgi eksik (hedef, irtifa, hız gibi) ve soru makul.
     Eksik bilgi komutta aslında varsa **hatalı** işaretle.
   * `refusal`: komut gerçekten güvensiz ya da kapsam dışı (silah, sınır ihlali, havalimanı pisti vb.).

## Tuşlar
| Tuş | Anlamı |
|---|---|
| Enter | doğru, kümeye al |
| `n` | hatalı, kümeden at (en hızlı yol; kararsızsan at) |
| `c` | komutu düzelt (yeni komutu yaz), sonra Enter |
| `g` | altın çıktıyı düzelt: tek satır JSON yapıştır, şemaya göre doğrulanır |
| `k` | karara not ekle |
| `s` | atla, sonra yeniden gelir |
| `b` | bu oturumdaki son kararı geri al |
| `q` | kaydet ve çık |

**Karar ölçütü (en sık takıldığın yer):** *"Komutu uygulamak için komutta olmayan bir değer, hedef ya da yön
uydurmam gerekir mi?"* Evet ise netleştirme sorusu **doğru**, hayır ise soru **gereksiz** (hatalı).
Örnek: komut "Hızı artır", çıktı "Hız bilgisini eksik aldım, netleştirir misiniz?" → **doğru** (hız sayısı yok,
uydurmadan plan çıkmaz). Komut "Hızı 20 m/s yap" iken aynı soru çıkarsa → **hatalı** (gereksiz soru).

**Hatalı nedeni (isteğe bağlı, hızı bozmaz):** `N`'ye basınca kart hemen ilerler; altta görünen şeritte
`1–5` ile neden seçebilirsin (aynı tuşa tekrar basmak siler). Seçmezsen yalnızca "hatalı" kalır.
`1` komut yapay/anlamsız · `2` çıktı komuta uymuyor (sayı/yön/yer) · `3` gereksiz soru (bilgi komutta var) ·
`4` soru gerekliydi (bilgi eksik ama plan üretilmiş) · `5` ret yersiz ya da ret gerekliydi.
Dağılım `--stats` ile görülür ve şablon veri kalitesi bulgusu olarak rapora girer.

Plan JSON'unu elle düzeltmek zahmetli; kararsız ya da bozuk plan adaylarını `n` ile atmak yeterlidir.

## İlerleme ve dışa aktarma
```bat
uv run python -m skymind.data.review --stats     :: toplam / ok / edited / bad / pending
uv run python -m skymind.data.review --export    :: ok + edited olanlardan data/gold/gold.jsonl üretir
```

Beklenen süre: ~30–60 sn/aday (bozuk olanlar daha hızlı). 100 aday ≈ 1 saat.

## İş bitince ben ne yapacağım
`data/gold/gold.jsonl` ile (planlayıcı değerlendirme formatıyla uyumlu) SFT ve DPO'yu yeniden ölçeriz:
`python -m skymind.planner.evaluate --name sft-gold --adapter ... --data data/gold/gold.jsonl --limit 0`
(~20–30 dk, GPU), sonuçlar `docs/RESULTS.md`'ye "gold" sütunu olarak eklenir. Kaç aday attığın da (bad oranı)
şablon verinin kalitesi hakkında kendi başına bir bulgudur ve rapora yazılır.
