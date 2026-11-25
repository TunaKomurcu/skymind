# Altın küme inceleme rehberi

Amaç: 300 şablon-üretimli aday komutu insan gözüyle doğrulayıp `data/gold/gold.jsonl` üretmek. Bu olmadan
planlayıcı sonuçları yalnızca sentetik test kümesinde kalır (sentetik veri ve aynı üretici dağılımı).

## Başlatma (ayrı bir terminalde; Claude Code içindeki `!` istemi etkileşimli girdiyi desteklemez)

```bat
cd C:\Users\Tuna\skymind
uv run python -m skymind.data.review
```

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
