"""
ORION-TR / Adim 8
CEMS yangin tehlike indekslerini (Kanada FWI sistemi) Turkiye icin 2016-2025
araliginda indirir.

Urun : cems-fire-historical-v1 (EWDS)
DOI  : 10.24381/cds.0e89c522
Grid : 0.25 derece, ERA5 ile birebir ayni (29 x 79, Turkiye bbox)

Indirilen indeksler:
  fire_weather_index          FWI   genel yangin tehlikesi
  fine_fuel_moisture_code     FFMC  ince yakit nemi (tutusma kolayligi)
  duff_moisture_code          DMC   orta katman nemi
  drought_code                DC    derin katman kurakligi
  initial_fire_spread_index   ISI   baslangic yayilim hizi
  build_up_index              BUI   yanabilir yakit miktari

Bunlar Algerian Forest Fires veri setindeki sutunlarin Turkiye karsiligidir.

Dayaniklilik (devir rehberi 4.2):
  .part -> dogrulama -> atomik tasima; gecerli dosya atlanir; 429/5xx backoff;
  bozuk dosya .invalid.<UTC> olarak karantinaya alinir; SHA-256 manifesti.

Kullanim:
    .venv\\Scripts\\python.exe scripts\\download_fwi.py --probe
    .venv\\Scripts\\python.exe scripts\\download_fwi.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "config" / "secrets.env"
RAW = ROOT / "data_raw" / "fwi"
META = RAW / "metadata"
REPORT = ROOT / "_reports"

DATASET = "cems-fire-historical-v1"
AREA = [42.5, 25.5, 35.5, 45.0]          # kuzey, bati, guney, dogu
GRID = "0.25/0.25"

VARIABLES = [
    "fire_weather_index",
    "fine_fuel_moisture_code",
    "duff_moisture_code",
    "drought_code",
    "initial_fire_spread_index",
    "build_up_index",
]

YEARS = list(range(2016, 2026))
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

MAX_RETRY = 4
HDF5_MAGIC = b"\x89HDF"
ZIP_MAGIC = b"PK\x03\x04"


def read_env() -> tuple[str, str]:
    if not SECRETS.exists():
        sys.exit(f"HATA: {SECRETS} yok.")
    env = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    url, key = env.get("EWDS_URL", ""), env.get("EWDS_KEY", "")
    if not url or not key:
        sys.exit("HATA: secrets.env icinde EWDS_URL / EWDS_KEY eksik.")
    return url, key


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def valid_container(p: Path) -> bool:
    """Uzantiya degil magic byte'a bakar (devir rehberi 8.4)."""
    if not p.exists() or p.stat().st_size < 1024:
        return False
    head = p.open("rb").read(8)
    return head.startswith(HDF5_MAGIC) or head.startswith(ZIP_MAGIC)


def build_request(variable: str, year: int | None, day_probe: bool) -> dict:
    req = {
        "product_type": "reanalysis",
        "variable": [variable],
        "dataset_type": "consolidated_dataset",
        "system_version": "4_1",
        "data_format": "netcdf",
        "grid": GRID,
        "area": AREA,
    }
    if day_probe:
        req |= {"year": "2021", "month": "07", "day": ["28"]}
    else:
        req |= {"year": str(year), "month": MONTHS, "day": DAYS}
    return req


def retrieve(client, request: dict, target: Path, secret: str) -> tuple[bool, str]:
    delay = 5.0
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, MAX_RETRY + 1):
        try:
            client.retrieve(DATASET, request, str(part))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if secret and len(secret) > 4:
                msg = msg.replace(secret, "<GIZLI>")
            transient = any(s in msg for s in ("429", "500", "502", "503", "504",
                                               "Timeout", "Connection"))
            if transient and attempt < MAX_RETRY:
                time.sleep(delay + random.uniform(0, 3))
                delay *= 2
                continue
            part.unlink(missing_ok=True)
            return False, msg[:300]

        if valid_container(part):
            part.replace(target)
            return True, "ok"

        bad = target.with_suffix(f".invalid.{utc().replace(':', '')}")
        part.replace(bad)
        return False, f"gecersiz container -> {bad.name}"

    return False, "deneme tukendi"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Degisken adlarini tek gunluk isteklerle sina")
    ap.add_argument("--variables", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    a = ap.parse_args()

    import cdsapi

    url, key = read_env()
    variables = a.variables or VARIABLES
    years = a.years or YEARS

    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=url, key=key, quiet=True,
                           wait_until_complete=True, delete=False)

    # ------------------------------------------------------------- probe
    if a.probe:
        probe_dir = RAW / "_probe"
        probe_dir.mkdir(exist_ok=True)
        print("Degisken adlari sinaniyor (tek gun, 2021-07-28)\n")
        ok_vars, bad_vars = [], []
        for v in variables:
            t = probe_dir / f"probe_{v}.nc"
            print(f"  {v:28s} ...", end=" ", flush=True)
            if valid_container(t):
                print("zaten var")
                ok_vars.append(v)
                continue
            good, note = retrieve(client, build_request(v, None, True), t, key)
            print("GECERLI" if good else f"HATA -> {note}")
            (ok_vars if good else bad_vars).append(v)

        print(f"\nGecerli: {len(ok_vars)}/{len(variables)}")
        if bad_vars:
            print("Gecersiz degisken adlari:", ", ".join(bad_vars))
            print("Urun sayfasindaki 'Show API request' ciktisiyla karsilastirin.")
        else:
            print("Hepsi gecerli. Tam indirme icin --probe olmadan calistirin.")
        (META / "probe_result.json").write_text(json.dumps(
            {"checked_at_utc": utc(), "ok": ok_vars, "failed": bad_vars},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # -------------------------------------------------------- tam indirme
    total = len(variables) * len(years)
    print(f"Toplam istek: {total}  ({len(variables)} degisken x {len(years)} yil)")
    print("Her istek kuyruga giriyor; ekranda hareket olmasa da bekleyin.\n")

    manifest = META / "request-manifest.jsonl"
    counters = {"indirildi": 0, "atlandi": 0, "hata": 0}
    gaps: list[dict] = []
    i = 0

    for v in variables:
        vdir = RAW / v
        vdir.mkdir(exist_ok=True)
        for y in years:
            i += 1
            target = vdir / f"fwi_{v}_{y}_turkey.nc"
            tag = f"[{i}/{total}] {v} {y}"

            if valid_container(target):
                counters["atlandi"] += 1
                print(f"{tag} zaten var")
                continue

            print(f"{tag} ...", end=" ", flush=True)
            t0 = time.time()
            req = build_request(v, y, False)
            good, note = retrieve(client, req, target, key)
            dt = time.time() - t0

            if not good:
                counters["hata"] += 1
                print(f"HATA ({note})")
                gaps.append({"variable": v, "year": y, "status": "failed",
                             "note": note})
                continue

            counters["indirildi"] += 1
            size = target.stat().st_size
            print(f"{size / 1e6:.1f} MB  ({dt / 60:.1f} dk)")

            rec = {
                "source": "CEMS Fire danger indices historical",
                "provider": "Copernicus EMS / ECMWF (EWDS)",
                "dataset_or_product_id": DATASET,
                "doi": "10.24381/cds.0e89c522",
                "version": "system_version 4_1, consolidated_dataset",
                "variable": v,
                "retrieved_at_utc": utc(),
                "temporal_start": f"{y}-01-01",
                "temporal_end": f"{y}-12-31",
                "bbox_or_aoi": AREA,
                "crs": "EPSG:4326",
                "spatial_resolution": GRID,
                "original_filename": target.name,
                "bytes": size,
                "sha256": sha256_of(target),
                "license": "Copernicus Licence to Use Copernicus Products",
                "request": req,
            }
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if gaps:
        (REPORT / "fwi_gaps.json").write_text(
            json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")

    status = {
        "source": "CEMS FWI", "generated_at_utc": utc(),
        "dataset": DATASET, "variables": variables, "years": years,
        "area": AREA, "grid": GRID, "counters": counters,
        "expected_files": total,
        "found_files": sum(1 for v in variables for y in years
                           if valid_container(RAW / v / f"fwi_{v}_{y}_turkey.nc")),
    }
    (REPORT / "fwi_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- OZET ---")
    for k, val in counters.items():
        print(f"{k:12s}: {val}")
    print(f"{'beklenen':12s}: {total}")
    print(f"{'bulunan':12s}: {status['found_files']}")
    print(f"\nRapor: {REPORT / 'fwi_status.json'}")


if __name__ == "__main__":
    main()
