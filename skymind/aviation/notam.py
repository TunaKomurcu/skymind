"""NOTAM ayrıştırıcı, varlık çıkarımı ve Türkçe özet (kural tabanlı).

Biçim: ICAO Annex 15 (Q/A/B/C/D/E/F/G maddeleri). Q kodu sözlüğü bilerek küçüktür; tanınmayan kod
ham haliyle özete girer, tahmin edilmez. E) metni İngilizce kısaltmalardan oluşur; özet bunları
kelime kelime Türkçeye açar ve çevrilemeyenleri olduğu gibi bırakır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

SUBJECT_TR = {
    "FA": "aerodrom",
    "LR": "pist ışıkları",
    "MR": "pist",
    "MX": "taksi yolu",
    "MN": "apron",
    "NV": "VOR",
    "NB": "NDB",
    "PI": "aletli yaklaşma",
    "OB": "engel",
    "RT": "kısıtlı bölge",
    "RD": "tehlikeli bölge",
    "RP": "yasak bölge",
    "RR": "uçuş yasağı bölgesi",
    "WP": "paraşüt atlayışı",
    "WU": "insansız hava aracı faaliyeti",
    "WE": "gösteri/tören",
    "WM": "füze/top atışı",
    "CO": "iletişim",
    "SA": "ATIS",
}
CONDITION_TR = {
    "LC": "kapalı",
    "AS": "servis dışı",
    "AH": "hizmet saatleri değişti",
    "CA": "aktif",
    "CH": "değişti",
    "AK": "normal çalışıyor",
    "AL": "şartlı olarak kullanılabilir",
    "LT": "sınırlı",
    "AU": "kullanılamaz",
    "CN": "iptal edildi",
    "XX": "diğer",
    "LW": "gerçekleşecek",
}
ABBREV_TR = {
    "RWY": "pist",
    "TWY": "taksi yolu",
    "APRON": "apron",
    "CLSD": "kapalı",
    "CLOSED": "kapalı",
    "OPS": "faaliyet",
    "MAINT": "bakım",
    "MAINTENANCE": "bakım",
    "WIP": "çalışma devam ediyor",
    "WI": "içinde",
    "RADIUS": "yarıçap",
    "U/S": "servis dışı",
    "UNSERVICEABLE": "servis dışı",
    "AVBL": "kullanılabilir",
    "ACT": "aktif",
    "ACTIVE": "aktif",
    "AREA": "bölge",
    "RESTRICTED": "kısıtlı",
    "PROHIBITED": "yasak",
    "DANGER": "tehlikeli",
    "UAS": "insansız hava aracı",
    "UAV": "insansız hava aracı",
    "DRONE": "insansız hava aracı",
    "OPR": "faaliyet",
    "OBST": "engel",
    "CRANE": "vinç",
    "LGT": "ışık",
    "LIGHTING": "aydınlatma",
    "FREQ": "frekans",
    "AIRSPACE": "hava sahası",
    "ALL": "tüm",
    "AND": "ve",
    "BTN": "arasında",
    "SFC": "yüzey",
    "UNL": "sınırsız",
    "AMSL": "deniz seviyesinden",
    "AGL": "yerden",
    "DAILY": "günlük",
    "EXC": "hariç",
    "ONLY": "yalnızca",
    "PERM": "süresiz",
    "AD": "aerodrom",
}
ACTION_TR = {"NOTAMN": "yeni", "NOTAMR": "değiştirir", "NOTAMC": "iptal eder"}

RE_HEAD = re.compile(r"^\s*([A-Z])(\d{4})/(\d{2})\s+(NOTAM[NRC])(?:\s+([A-Z]\d{4}/\d{2}))?", re.M)
RE_ITEM = re.compile(r"(?:^|\s)([QABCDEFG])\)\s*", re.M)
RE_Q = re.compile(
    r"^(\w{4})/Q(\w{2})(\w{2})/(\w+)/(\w+)/(\w+)/(\d{3})/(\d{3})(?:/(\d{4}[NS]\d{5}[EW])(\d{3})?)?$"
)
RE_DT = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(EST)?$")

# --- E) metninden varlık çıkarımı ---------------------------------------------------------------
RE_RUNWAY = re.compile(r"\bRWY\s+(\d{2}[LCR]?(?:/\d{2}[LCR]?)?)")
RE_TAXIWAY = re.compile(r"\bTWY\s+([A-Z]{1,2}\d?)\b")
RE_FREQ = re.compile(r"\b(1[0-3]\d\.\d{1,3})\s*(?:MHZ)?")
RE_FL = re.compile(r"\bFL\s?(\d{2,3})\b")
RE_ALT = re.compile(r"\b(\d{2,5})\s?(?:FT|M)\s?(AMSL|AGL)\b")
RE_COORD = re.compile(r"\b(\d{4}(?:\d{2})?[NS])\s?(\d{5}(?:\d{2})?[EW])\b")
RE_RADIUS = re.compile(r"\b(\d+(?:\.\d+)?)\s?(NM|KM)\b")


@dataclass(frozen=True)
class QLine:
    fir: str
    code: str  # örn. QRTCA
    subject: str
    condition: str
    traffic: str
    purpose: str
    scope: str
    lower_fl: int
    upper_fl: int
    center: str | None = None  # 4059N02845E
    radius_nm: int | None = None


@dataclass
class Notam:
    raw: str
    series: str
    number: int
    year: int
    action: str  # NOTAMN/R/C
    replaces: str | None = None
    q: QLine | None = None
    locations: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    permanent: bool = False
    estimated_end: bool = False
    schedule: str | None = None
    text: str = ""
    lower: str | None = None
    upper: str | None = None

    @property
    def ident(self) -> str:
        return f"{self.series}{self.number:04d}/{self.year:02d}"


def _dt(token: str) -> tuple[datetime, bool]:
    m = RE_DT.match(token)
    if not m:
        raise ValueError(f"geçersiz tarih grubu: {token!r}")
    yy, mo, dd, hh, mi = (int(m[i]) for i in range(1, 6))
    return datetime(2000 + yy, mo, dd, hh, mi), bool(m[6])


def parse_notam(text: str) -> Notam:
    head = RE_HEAD.search(text)
    if not head:
        raise ValueError("NOTAM başlığı (örn. A1234/25 NOTAMN) bulunamadı")
    n = Notam(
        raw=text.strip(),
        series=head[1],
        number=int(head[2]),
        year=int(head[3]),
        action=head[4],
        replaces=head[5],
    )
    body = text[head.end() :]
    marks = list(RE_ITEM.finditer(body))
    items: dict[str, str] = {}
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        items[mk[1]] = body[mk.end() : end].strip()
    if "A" not in items:
        raise ValueError("A) maddesi (konum) yok")
    n.locations = items["A"].split()
    if "Q" in items:
        m = RE_Q.match(items["Q"].replace(" ", ""))
        if m:
            n.q = QLine(
                fir=m[1],
                code="Q" + m[2] + m[3],
                subject=m[2],
                condition=m[3],
                traffic=m[4],
                purpose=m[5],
                scope=m[6],
                lower_fl=int(m[7]),
                upper_fl=int(m[8]),
                center=m[9],
                radius_nm=int(m[10]) if m[10] else None,
            )
    if "B" in items:
        n.valid_from, _ = _dt(items["B"].split()[0])
    if "C" in items:
        c = items["C"].split()[0]
        if c.startswith("PERM"):
            n.permanent = True
        else:
            n.valid_to, n.estimated_end = _dt(c)
    n.schedule = items.get("D")
    n.text = " ".join(items.get("E", "").split())
    n.lower, n.upper = items.get("F"), items.get("G")
    return n


def extract_entities(text: str) -> dict[str, list[str]]:
    """E) serbest metninden belirli varlık türleri (regex tabanlı NER)."""
    return {
        "runway": [m[1] for m in RE_RUNWAY.finditer(text)],
        "taxiway": [m[1] for m in RE_TAXIWAY.finditer(text)],
        "frequency": [m[1] for m in RE_FREQ.finditer(text)],
        "flight_level": [f"FL{m[1]}" for m in RE_FL.finditer(text)],
        "altitude": [m[0] for m in RE_ALT.finditer(text)],
        "coordinate": [f"{m[1]} {m[2]}" for m in RE_COORD.finditer(text)],
        "distance": [m[0] for m in RE_RADIUS.finditer(text)],
    }


def _expand(text: str) -> str:
    # çok anlamlı "TO" yalnızca "DUE TO" kalıbında çevrilir; diğer yerlerde olduğu gibi kalır
    text = re.sub(r"\bDUE TO\b", "nedeniyle", text, flags=re.I)
    return " ".join(ABBREV_TR.get(w.upper().strip(",.;"), w) for w in text.split())


def _fmt(d: datetime) -> str:
    return d.strftime("%d.%m.%Y %H:%M")


def summarize_tr(n: Notam) -> str:
    parts = [f"NOTAM {n.ident} ({ACTION_TR[n.action]})"]
    if n.locations:
        parts.append("konum " + ", ".join(n.locations))
    if n.q:
        subj = SUBJECT_TR.get(n.q.subject, n.q.subject)
        cond = CONDITION_TR.get(n.q.condition, n.q.condition)
        parts.append(f"konu: {subj}, durum: {cond} ({n.q.code})")
    if n.valid_from:
        end = "süresiz" if n.permanent else (_fmt(n.valid_to) if n.valid_to else "belirsiz")
        suffix = " (tahmini)" if n.estimated_end else ""
        parts.append(f"geçerlilik {_fmt(n.valid_from)} – {end}{suffix} UTC")
    if n.schedule:
        parts.append("saatler: " + _expand(n.schedule))
    if n.lower or n.upper:
        parts.append(f"irtifa: {_expand(n.lower or '?')} – {_expand(n.upper or '?')}")
    if n.text:
        parts.append("ayrıntı: " + _expand(n.text))
    return "; ".join(parts) + "."
