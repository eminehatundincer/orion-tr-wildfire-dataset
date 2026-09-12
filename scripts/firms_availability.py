"""
ORION-TR / Adim 1e
FIRMS 'data availability' ucundan sensor bazli tarih araligi envanterini ceker.

Amac: indirmeye baslamadan once saglayicinin bize hangi sensor icin
hangi tarihleri verdigini kayit altina almak (provenans + gap analizi icin).

Kullanim (proje kokunden):
    .venv\\Scripts\\python.exe scripts\\firms_availability.py
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "config" / "secrets.env"
OUT_DIR = ROOT / "data_raw" / "firms" / "metadata"
OUT_CSV = OUT_DIR / "data_availability.csv"

BASE = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"


def read_key() -> str:
    if not SECRETS.exists():
        sys.exit(f"HATA: {SECRETS} bulunamadi. Once bu dosyayi olusturun.")
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("FIRMS_MAP_KEY="):
            key = line.split("=", 1)[1].strip()
            if not key or key.startswith("buraya"):
                sys.exit("HATA: FIRMS_MAP_KEY bos veya ornek deger.")
            return key
    sys.exit("HATA: secrets.env icinde FIRMS_MAP_KEY satiri yok.")


def main() -> None:
    key = read_key()
    url = f"{BASE}/{key}/ALL"

    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as e:
        sys.exit(f"HATA: istek basarisiz -> {e}")

    if r.status_code != 200:
        # Anahtari loglamamak icin URL'yi basmiyoruz.
        sys.exit(f"HATA: HTTP {r.status_code}. Anahtari ve internet baglantisini kontrol edin.")

    text = r.text.strip()
    if not text or "," not in text:
        sys.exit(f"HATA: beklenmeyen yanit (ilk 200 karakter): {text[:200]}")
    if "Invalid" in text or "invalid" in text.lower()[:200]:
        sys.exit("HATA: sunucu anahtari reddetti gibi gorunuyor.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text(text + "\n", encoding="utf-8")

    rows = list(csv.DictReader(io.StringIO(text)))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Kaydedildi : {OUT_CSV}")
    print(f"Cekim (UTC): {stamp}")
    print(f"Satir      : {len(rows)}")
    print()

    if rows:
        cols = list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        print("  ".join(c.ljust(widths[c]) for c in cols))
        print("  ".join("-" * widths[c] for c in cols))
        for r_ in rows:
            print("  ".join(str(r_.get(c, "")).ljust(widths[c]) for c in cols))

    print()
    print("Bu ciktiyi paylasirken anahtar yer almiyor; guvenle gonderebilirsiniz.")


if __name__ == "__main__":
    main()
