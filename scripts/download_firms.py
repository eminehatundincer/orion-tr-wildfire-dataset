"""
ORION-TR / Adim 2
FIRMS aktif yangin arsivini Turkiye icin 2016-2025 araliginda ceker.

Tasarim ilkeleri (devir rehberi bolum 4.2 ve 4.3):
  - once <ad>.part olarak yazilir, dogrulanip atomik olarak nihai ada tasinir
  - gecerli nihai dosya yeniden calistirmada atlanir (urun duzeyinde idempotent)
  - 429 ve gecici 5xx icin jitterli exponential backoff
  - bozuk dosya silinmez, <ad>.invalid.<UTC> olarak karantinaya alinir
  - her dosya icin SHA-256 ve request manifesti tutulur
  - "o gun gozlem yok" ile "dosya indirilemedi" ayri kaydedilir

Kullanim (proje kokunden):
    .venv\\Scripts\\python.exe scripts\\download_firms.py
    .venv\\Scripts\\python.exe scripts\\download_firms.py --sources VIIRS_SNPP_SP
    .venv\\Scripts\\python.exe scripts\\download_firms.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------- sabitler

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "config" / "secrets.env"
RAW_DIR = ROOT / "data_raw" / "firms" / "original"
META_DIR = ROOT / "data_raw" / "firms" / "metadata"
REPORT_DIR = ROOT / "_reports"

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Turkiye sinir kutusu: bati, guney, dogu, kuzey  (EPSG:4326)
BBOX = "25.5,35.5,45.0,42.5"

# Sensor -> saglayicinin verdigi gecerli tarih araligi (data_availability ciktisi)
SOURCES = {
    "VIIRS_SNPP_SP":   ("2012-01-20", "2026-04-27"),
    "MODIS_SP":        ("2000-11-01", "2026-04-30"),
    "VIIRS_NOAA20_SP": ("2018-04-01", "2026-05-31"),
}

DEFAULT_SOURCES = ["VIIRS_SNPP_SP", "MODIS_SP", "VIIRS_NOAA20_SP"]

WINDOW_DAYS = 5         # API'nin tek istekte izin verdigi azami gun sayisi
SLEEP_BETWEEN = 1.5       # istekler arasi bekleme (saniye) - hiz limiti icin
MAX_RETRY = 5


# ---------------------------------------------------------------- yardimci

def read_key() -> str:
    if not SECRETS.exists():
        sys.exit(f"HATA: {SECRETS} yok.")
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("FIRMS_MAP_KEY="):
            key = line.split("=", 1)[1].strip()
            if not key or key.lower().startswith(("mailden", "buraya")):
                sys.exit("HATA: FIRMS_MAP_KEY bos veya ornek deger.")
            return key
    sys.exit("HATA: secrets.env icinde FIRMS_MAP_KEY yok.")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def windows(start: date, end: date, size: int):
    """[start, end] araligini size gunluk pencerelere boler."""
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=size - 1), end)
        yield cur, (last - cur).days + 1
        cur = last + timedelta(days=1)


def clamp(d: date, lo: str, hi: str) -> bool:
    return date.fromisoformat(lo) <= d <= date.fromisoformat(hi)


def looks_like_csv(text: str) -> bool:
    head = text.lstrip().split("\n", 1)[0].lower()
    return "latitude" in head and "longitude" in head and "acq_date" in head


# ---------------------------------------------------------------- indirme

def fetch(url: str) -> tuple[bool, str, str]:
    """(basarili, metin, not) dondurur. Anahtari asla loglamaz."""
    delay = 2.0
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as e:
            note = f"ag hatasi: {type(e).__name__}"
            if attempt == MAX_RETRY:
                return False, "", note
            time.sleep(delay + random.uniform(0, 1.5))
            delay *= 2
            continue

        if r.status_code == 200:
            return True, r.text, "ok"

        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait + random.uniform(0, 2))
            delay = max(delay * 2, wait)
            continue

        if 500 <= r.status_code < 600:
            if attempt == MAX_RETRY:
                return False, "", f"HTTP {r.status_code}"
            time.sleep(delay + random.uniform(0, 1.5))
            delay *= 2
            continue

        return False, "", f"HTTP {r.status_code}"

    return False, "", "tukenmis deneme"


def run(sources: list[str], start: date, end: date, dry: bool) -> None:
    key = read_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = META_DIR / "request-manifest.jsonl"
    inventory: list[dict] = []
    gaps: list[dict] = []
    counters = {"indirildi": 0, "atlandi": 0, "bos": 0, "hata": 0, "kapsam_disi": 0}

    for src in sources:
        if src not in SOURCES:
            print(f"UYARI: bilinmeyen kaynak {src}, atlaniyor.")
            continue
        lo, hi = SOURCES[src]
        out_dir = RAW_DIR / src
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {src}  (saglayici araligi {lo} .. {hi}) ===")

        for wstart, ndays in windows(start, end, WINDOW_DAYS):
            wend = wstart + timedelta(days=ndays - 1)
            tag = f"{src}_{wstart.isoformat()}_{ndays}d"
            final = out_dir / f"{tag}.csv"
            part = out_dir / f"{tag}.csv.part"

            if not clamp(wstart, lo, hi) or not clamp(wend, lo, hi):
                counters["kapsam_disi"] += 1
                gaps.append({
                    "source": src, "window_start": wstart.isoformat(),
                    "days": ndays, "status": "provider_out_of_range",
                    "note": f"saglayici araligi {lo}..{hi}",
                })
                continue

            if final.exists() and final.stat().st_size > 0:
                counters["atlandi"] += 1
                continue

            url = f"{BASE}/{key}/{src}/{BBOX}/{ndays}/{wstart.isoformat()}"
            print(f"  {tag} ...", end=" ", flush=True)

            if dry:
                print("(dry-run)")
                continue

            ok, text, note = fetch(url)
            time.sleep(SLEEP_BETWEEN)

            if not ok:
                counters["hata"] += 1
                print(f"HATA ({note})")
                gaps.append({
                    "source": src, "window_start": wstart.isoformat(),
                    "days": ndays, "status": "download_failed", "note": note,
                })
                continue

            if not looks_like_csv(text):
                counters["hata"] += 1
                bad = out_dir / f"{tag}.csv.invalid.{utc_stamp().replace(':', '')}"
                bad.write_text(text[:20000], encoding="utf-8")
                print("GECERSIZ (karantinaya alindi)")
                gaps.append({
                    "source": src, "window_start": wstart.isoformat(),
                    "days": ndays, "status": "invalid_payload",
                    "note": text.strip()[:120].replace("\n", " "),
                })
                continue

            part.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            part.replace(final)   # atomik tasima

            rows = list(csv.DictReader(io.StringIO(text)))
            digest = sha256_of(final)

            if rows:
                counters["indirildi"] += 1
                print(f"{len(rows)} kayit")
            else:
                counters["bos"] += 1
                print("0 kayit (kapsanmis, olay yok)")

            rec = {
                "source": "FIRMS", "provider": "NASA LANCE/FIRMS",
                "dataset_or_product_id": src,
                "retrieved_at_utc": utc_stamp(),
                "temporal_start": wstart.isoformat(),
                "temporal_end": wend.isoformat(),
                "bbox_or_aoi": BBOX, "crs": "EPSG:4326",
                "original_filename": final.name,
                "bytes": final.stat().st_size,
                "sha256": digest,
                "record_count": len(rows),
                "license": "NASA Open Data",
                "source_url_without_credentials":
                    f"{BASE}/<MAP_KEY>/{src}/{BBOX}/{ndays}/{wstart.isoformat()}",
            }
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            inventory.append({
                "source": src, "file": final.name,
                "window_start": wstart.isoformat(), "window_end": wend.isoformat(),
                "days": ndays, "records": len(rows),
                "bytes": final.stat().st_size, "sha256": digest,
                "status": "covered_events" if rows else "covered_zero_events",
            })

    # ------------------------------------------------------------ raporlar
    if inventory:
        inv = REPORT_DIR / "firms_inventory.csv"
        write_header = not inv.exists()
        with inv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(inventory)

    if gaps:
        gp = REPORT_DIR / "firms_gaps.csv"
        write_header = not gp.exists()
        with gp.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(gaps[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(gaps)

    status = {
        "source": "FIRMS",
        "generated_at_utc": utc_stamp(),
        "sources": sources,
        "bbox": BBOX, "crs": "EPSG:4326",
        "period": [start.isoformat(), end.isoformat()],
        "window_days": WINDOW_DAYS,
        "counters": counters,
        "note": "Etiket uretiminde yalniz VIIRS_SNPP_SP + MODIS_SP kullanilmalidir; "
                "VIIRS_NOAA20_SP 2018-04-01'de seriye girdigi icin kalibrasyon kirilimi olusturur.",
    }
    (REPORT_DIR / "firms_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- OZET ---")
    for k, v in counters.items():
        print(f"{k:14s}: {v}")
    print(f"\nRapor: {REPORT_DIR / 'firms_status.json'}")


# ---------------------------------------------------------------- giris

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    run(
        sources=list(a.sources),
        start=date.fromisoformat(a.start),
        end=date.fromisoformat(a.end),
        dry=a.dry_run,
    )


if __name__ == "__main__":
    main()
