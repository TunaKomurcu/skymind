"""Kurgusal "SkyMind X1" bakım/operasyon kılavuzu ve soru seti (RAG değerlendirmesi için).

Doküman KURGUSALDIR: gerçek bir hava aracına ait değildir. Gerçekler yapılandırılmış tablodan
üretilir; her sorunun altın cümlesi bellidir, bu yüzden etiket hatası olamaz. Sorular iki türdür:
``lexical`` (kılavuzdaki sözcüklerle) ve ``paraphrase`` (eş anlamlı/dolaylı anlatımla). İkincisi
BM25'i zorlar, yoğun aramayı ölçer.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

# bileşen: (kılavuzdaki ad, soruda kullanılan eş anlamlı, fiil kökü: (kılavuz, soru))
COMPONENTS = [
    ("pervane", "rotor kanadı", "değiştirilir"),
    ("akü paketi", "batarya", "değiştirilir"),
    ("motor yatağı", "motor rulmanı", "değiştirilir"),
    ("gimbal kamera", "kamera stabilizatörü", "kalibre edilir"),
    ("GNSS modülü", "konum alıcısı", "kalibre edilir"),
    ("pitot tüpü", "hava hızı sensörü", "kontrol edilir"),
    ("servo motor", "kontrol yüzeyi motoru", "kontrol edilir"),
    ("hız kontrolcü", "ESC kartı", "kontrol edilir"),
    ("gövde bağlantı cıvataları", "gövde cıvataları", "sıkılık kontrolünden geçirilir"),
    ("kurtarma paraşütü", "acil durum paraşütü", "paketlenir ve incelenir"),
]
INTERVALS = [25, 50, 100, 150, 200, 250, 300, 400, 500]

LIMITS: list[tuple[str, str, str, list[Any], str, tuple[str, str]]] = [
    (
        "lim_wind",
        "Azami sürekli rüzgâr hızı",
        "{v} m/s",
        [8, 10, 12, 15],
        "En fazla {v} m/s sürekli rüzgârda uçurulabilir.",
        ("En fazla kaç m/s rüzgârda uçulabilir?", "Hangi rüzgâr hızından sonra uçuş yapılmaz?"),
    ),
    (
        "lim_alt",
        "Azami operasyon irtifası",
        "{v} m",
        [1500, 2500, 3000, 4000],
        "Azami operasyon irtifası deniz seviyesinden {v} m'dir.",
        ("Azami operasyon irtifası nedir?", "Aracı en yüksek kaç metreye çıkarabilirim?"),
    ),
    (
        "lim_time",
        "Azami uçuş süresi",
        "{v} dakika",
        [35, 45, 55, 70],
        "Tam dolu bataryayla azami uçuş süresi {v} dakikadır.",
        ("Tam bataryayla uçuş süresi kaç dakikadır?", "Araç havada en çok ne kadar kalabilir?"),
    ),
    (
        "lim_tmin",
        "Asgari çalışma sıcaklığı",
        "{v} °C",
        [-20, -10, -5, 0],
        "Araç {v} °C'nin altındaki ortam sıcaklıklarında çalıştırılmaz.",
        (
            "Hangi sıcaklığın altında araç çalıştırılmaz?",
            "Soğuk havada alt sıcaklık sınırı kaç derecedir?",
        ),
    ),
    (
        "lim_tmax",
        "Azami çalışma sıcaklığı",
        "{v} °C",
        [40, 45, 50, 55],
        "Araç {v} °C'nin üzerindeki ortam sıcaklıklarında çalıştırılmaz.",
        (
            "Hangi sıcaklığın üstünde araç çalıştırılmaz?",
            "Sıcak havada üst sıcaklık sınırı kaç derecedir?",
        ),
    ),
    (
        "bat_charge",
        "Hücre başına tam şarj gerilimi",
        "{v} V",
        ["4,20", "4,35", "4,15"],
        "Şarj tamamlandığında hücre başına gerilim {v} V olmalıdır.",
        ("Şarj bitince hücre başına kaç volt olmalı?", "Dolu bir bataryanın hücre gerilimi nedir?"),
    ),
    (
        "bat_store",
        "Depolama gerilimi",
        "{v} V",
        ["3,80", "3,85", "3,75"],
        "Uzun süre saklanacak bataryalar hücre başına {v} V'ta depolanır.",
        (
            "Bataryayı depolarken hücre gerilimi kaç volt olmalı?",
            "Uzun süreli saklamada hangi gerilim hedeflenir?",
        ),
    ),
    (
        "bat_rtl",
        "Otomatik eve dönüş eşiği",
        "%{v}",
        [20, 25, 30],
        "Batarya yüzde {v} seviyesine düştüğünde araç otomatik olarak eve döner.",
        (
            "Batarya yüzde kaça düşünce araç kendiliğinden eve döner?",
            "Otomatik eve dönüş hangi şarj seviyesinde başlar?",
        ),
    ),
    (
        "rtl_wait",
        "Bağlantı kaybı bekleme süresi",
        "{v} saniye",
        [5, 10, 15, 30],
        "Telemetri bağlantısı kesilirse araç {v} saniye bekler, ardından eve dönüş başlatır.",
        (
            "Bağlantı koparsa araç kaç saniye bekler?",
            "Telemetri kesildiğinde eve dönüş ne zaman başlar?",
        ),
    ),
    (
        "geofence",
        "Varsayılan geofence yarıçapı",
        "{v} m",
        [1000, 2000, 5000],
        "Varsayılan geofence yarıçapı kalkış noktasından {v} m'dir.",
        (
            "Varsayılan geofence yarıçapı kaç metredir?",
            "Araç kalkış noktasından en çok ne kadar uzaklaşabilir?",
        ),
    ),
]

# hata kodu: (anlam cümlesi, eylem cümlesi, belirti cümlesi - kodsuz soru için)
ERRORS = [
    (
        "pusula tutarsızlığı",
        "Aracı metal yapılardan uzaklaştırıp pusulayı yeniden kalibre edin.",
        "Uçuş öncesi pusula uyarısı çıkıyor ve yön bilgisi tutarsız görünüyor",
    ),
    (
        "GNSS sinyali zayıf",
        "Açık alana çıkın ve en az 10 uydu kilitlenene kadar bekleyin.",
        "Konum doğruluğu düştü ve uydu sayısı az görünüyor",
    ),
    (
        "hücre dengesizliği",
        "Bataryayı dengeleme modunda şarj edin; fark sürerse değiştirin.",
        "Batarya hücreleri arasında gerilim farkı uyarısı alıyorum",
    ),
    (
        "motor aşırı akım",
        "Pervanenin serbest döndüğünü kontrol edin, hız kontrolcüyü inceleyin.",
        "Motor normalden fazla akım çekiyor ve ısınıyor",
    ),
    (
        "barometre hatası",
        "Aracı rüzgârsız bir ortamda 60 saniye bekletip yeniden başlatın.",
        "İrtifa değeri uçuş sırasında dalgalanıyor",
    ),
    (
        "hava hızı sensörü tıkalı",
        "Pitot tüpünü basınçlı hava ile temizleyin.",
        "Hava hızı okuması sıfır görünüyor",
    ),
    (
        "telemetri zayıf",
        "Anteni yeniden yönlendirin ve parazit kaynaklarından uzaklaşın.",
        "Yer istasyonuyla bağlantı kalitesi düşüyor",
    ),
    (
        "kamera gimbal kilidi",
        "Gimbal kilit sabitini çıkarın ve aracı yeniden başlatın.",
        "Gimbal kamera hareket etmiyor ve titreşim sesi geliyor",
    ),
    (
        "IMU ısınmadı",
        "Aracı 3 dakika çalışır halde bekletin, sonra yeniden başlatın.",
        "Araç açılışta eğim sensörünün hazır olmadığını bildiriyor",
    ),
    (
        "paraşüt mekanizması hatası",
        "Uçuşa kalkmayın; paraşüt tetikleyicisini servise gönderin.",
        "Kurtarma paraşütü sistemi kendi kendini test edemedi",
    ),
    (
        "kart tam",
        "Kayıt kartındaki dosyaları yedekleyip kartı biçimlendirin.",
        "Görüntü kaydı başlamıyor ve depolama dolu uyarısı var",
    ),
    (
        "geofence ihlali",
        "Aracı kalkış noktasına doğru manuel olarak yönlendirin.",
        "Araç sınır dışına çıktığı için uyarı veriyor",
    ),
    (
        "ESC sıcaklığı yüksek",
        "İniş yapın ve hız kontrolcü 15 dakika soğuyana kadar bekleyin.",
        "Hız kontrolcü kartı aşırı ısındı uyarısı var",
    ),
    (
        "sürücü gerilimi düşük",
        "Bataryayı değiştirin ve kablo bağlantılarını inceleyin.",
        "Uçuş sırasında besleme gerilimi aniden düştü",
    ),
    (
        "görev dosyası bozuk",
        "Görev planını yer istasyonundan yeniden yükleyin.",
        "Yüklenen görev planı doğrulamadan geçmiyor",
    ),
    (
        "yazılım sürümü uyumsuz",
        "Yer istasyonu ve araç yazılımını aynı sürüme getirin.",
        "Yer istasyonu araçla sürüm uyuşmazlığı bildiriyor",
    ),
    (
        "rüzgâr limiti aşıldı",
        "İnişe geçin; rüzgâr sınırın altına inene kadar uçmayın.",
        "Araç rüzgâr sınırı aşıldı diye uyarıyor",
    ),
    (
        "mıknatıs girişimi",
        "Aracı yüksek akım taşıyan kablolardan uzaklaştırın.",
        "Pusula değeri kabloların yanında bozuluyor",
    ),
    (
        "depolama gerilimi uyarısı",
        "Bataryayı depolama gerilimine getirin.",
        "Uzun süredir kullanılmayan batarya hakkında uyarı çıkıyor",
    ),
    (
        "servo sıkışması",
        "Kontrol yüzeyini elle hareket ettirip engeli temizleyin.",
        "Kanat yüzeyi komuta tam yanıt vermiyor",
    ),
]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    text: str


@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    gold_chunk: str  # altın cümleyi içeren parça
    gold_sentence: str
    kind: str  # lexical | paraphrase
    topic: str


@dataclass
class Corpus:
    chunks: list[Chunk]
    qas: list[QA] = field(default_factory=list)


FILLER = [
    "Bu bölümdeki işlemler yetkili bakım personeli tarafından uygulanır.",
    "Her işlemden sonra bakım defterine tarih, saat ve imza kaydı düşülür.",
    "Kılavuzdaki değerler fabrika ayarlarıdır; sahada değişiklik için mühendislik onayı alınır.",
    "Bakım sırasında araç enerjisiz olmalı ve batarya bağlantısı çıkarılmış olmalıdır.",
    "Hasarlı parçalar yerine aynı seri numarası aralığından yenisi takılır.",
]


def _sents_to_chunks(doc: str, sents: list[str], per: int) -> list[Chunk]:
    return [
        Chunk(f"{doc}#{i // per}", doc, " ".join(sents[i : i + per]))
        for i in range(0, len(sents), per)
    ]


MODELS = ["X1", "X2", "X3"]


def _lf(text: str) -> str:
    return text[0].lower() + text[1:]


def build_corpus(seed: int = 3) -> Corpus:
    """Üç araç modeli aynı cümle yapılarını farklı değerlerle tekrarlar (yanlış pasaj tuzağı)."""
    rng = random.Random(seed)
    # (doc, konu, altın cümle, [(soru, tür)], cevap)
    facts: list[tuple[str, str, str, list[tuple[str, str]], str]] = []

    for name, alias, verb in COMPONENTS:
        vals = rng.sample(INTERVALS, len(MODELS))
        for model, n in zip(MODELS, vals, strict=True):
            cap = name[0].upper() + name[1:]
            sent = f"SkyMind {model} modelinde {name} her {n} uçuş saatinde bir {verb}."
            qs = [
                (f"{model} modelinde {name} kaç uçuş saatinde bir {verb}?", "lexical"),
                (f"{model} için {alias} bakım periyodu kaç saattir?", "paraphrase"),
            ]
            del cap
            facts.append(("bakim", f"bakim:{name}:{model}", sent, qs, f"{n} uçuş saati"))

    for fid, _title, unit, values, tpl, (q_lex, q_par) in LIMITS:
        vals = rng.sample(values, min(len(MODELS), len(values)))
        vals += [rng.choice(values) for _ in range(len(MODELS) - len(vals))]
        doc = "batarya" if fid.startswith(("bat_", "rtl")) else "operasyon"
        for model, v in zip(MODELS, vals, strict=True):
            sent = f"SkyMind {model} modelinde {_lf(tpl.format(v=v))}"
            qs = [
                (f"{model} modelinde {_lf(q_lex)}", "lexical"),
                (f"{model} için {_lf(q_par)}", "paraphrase"),
            ]
            facts.append((doc, f"{fid}:{model}", sent, qs, unit.format(v=v)))

    for model in MODELS:
        codes = rng.sample(range(100, 400), len(ERRORS))
        for (meaning, action, symptom), code in zip(ERRORS, codes, strict=True):
            sent = f"{model} modelinde E-{code} hata kodu {meaning} anlamına gelir; {_lf(action)}"
            qs = [
                (f"{model} modelinde E-{code} hata kodu ne anlama gelir?", "lexical"),
                (f"{model} modelinde şu durum var: {_lf(symptom)}. Ne yapmalıyım?", "paraphrase"),
            ]
            facts.append(("hata", f"hata:{model}:{code}", sent, qs, action))

    docs: dict[str, list[str]] = {"bakim": [], "operasyon": [], "batarya": [], "hata": []}
    order = list(facts)
    rng.shuffle(order)
    for doc, _topic, sent, _qs, _ans in order:
        docs[doc].append(sent)
        docs[doc] += rng.sample(FILLER, 2)  # her gerçeğin yanında genel cümleler (gürültü)
    chunks: list[Chunk] = []
    for doc, sents in docs.items():
        chunks += _sents_to_chunks(doc, sents, per=4)

    corpus = Corpus(chunks)
    for _doc, topic, sent, qs, ans in facts:
        host = next(c for c in chunks if sent in c.text)
        for q, kind in qs:
            corpus.qas.append(QA(q, ans, host.id, sent, kind, topic))
    return corpus


_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]
