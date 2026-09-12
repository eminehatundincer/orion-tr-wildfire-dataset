"""
ORION-TR / Adim 10
ERA5 ve ERA5-Land gunluk degiskenlerini Turkiye icin 2016-2025 indirir.

Devir rehberindeki 23 serilik matris 920 urun ve 7-12 hafta demekti.
FWI hazir geldigi icin sicaklik/nem/ruzgar/yagisin turetilmis yangin indekslerine
gerek kalmadi. Matris 13 seriye indirildi:

  ERA5 (~0.25 derece)          6 seri
    daily_maximum : 2m_temperature
    daily_mean    : 2m_dewpoint_temperature, 10m_u_component_of_wind,
                    10m_v_component_of_wind
    daily_sum     : total_precipitation, surface_solar_radiation_downwards

  ERA5-Land (~0.1 derece)      7 seri
    daily_mean    : volumetric_soil_water_layer_1..4,
                    leaf_area_index_high_vegetation,
                    leaf_area_index_low_vegetation
    daily_maximum : skin_temperature

Toprak nemi katmanlari yakit nemi, LAI ise bitki ortusu yogunlugu icin
NDVI'nin yerini tutar (GEE hesabi gerekmez).

Rehber 8.4'teki sema notlari uygulandi:
  - istekte 'daily_maximum' (dosya adinda kisa etiket 'daily_max')
  - gunluk urunlerde 'data_format' alani yok
  - 'product_type: reanalysis' yalniz ERA5'te, ERA5-Land'de YOK
  - time_zone utc+03:00 ve frequency 1_hourly butun seride sabit

Maliyet siniri: bir istek limiti ~400. Yillik istek reddedilirse betik otomatik
olarak ceyreklere boler ve devam eder.

Kullanim:
    .venv\\Scripts\\python.exe scripts\\download_era5.py --probe
    .venv\\Scripts\\python.exe scripts\\download_era5.py
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
RAW = ROOT / "data_raw" / "era5"
META = RAW / "metadata"
REPORT = ROOT / "_reports"

AREA = [42.5, 25.5, 35.5, 45.0]     # kuzey, bati, guney, dogu
TIME_ZONE = "utc+03:00"
FREQUENCY = "1_hourly"
YEARS = list(range(2016, 2026))

DS_ERA5 = "derived-era5-single-levels-daily-statistics"
DS_LAND = "derived-era5-land-daily-statistics"

# (dataset, variable, daily_statistic)
MATRIX: list[tuple[str, str, str]] = [
    (DS_ERA5, "2m_temperature", "daily_maximum"),
    (DS_ERA5, "2m_dewpoint_temperature", "daily_mean"),
    (DS_ERA5, "10m_u_component_of_wind", "daily_mean"),
    (DS_ERA5, "10m_v_component_of_wind", "daily_mean"),
    (DS_ERA5, "total_precipitation", "daily_sum"),
    (DS_ERA5, "surface_solar_radiation_downwards", "daily_sum"),
    (DS_LAND, "volumetric_soil_water_layer_1", "daily_mean"),
    (DS_LAND, "volumetric_soil_water_layer_2", "daily_mean"),
    (DS_LAND, "volumetric_soil_water_layer_3", "daily_mean"),
    (DS_LAND, "volumetric_soil_water_layer_4", "daily_mean"),
    (DS_LAND, "leaf_area_index_high_vegetation", "daily_mean"),
    (DS_LAND, "leaf_area_index_low_vegetation", "daily_mean"),
    (DS_LAND, "skin_temperature", "daily_maximum"),
]

SHORT = {"daily_maximum": "daily_max", "daily_mean": "daily_mean",
         "daily_sum": "daily_sum", "daily_minimum": "daily_min"}

MONTHS_ALL = [f"{m:02d}" for m in range(1, 13)]
QUARTERS = {1: ["01", "02", "03"], 2: ["04", "05", "06"],
            3: ["07", "08", "09"], 4: ["10", "11", "12"]}
DAYS = [f"{d:02d}" for d in range(1, 32)]

MAX_RETRY = 3
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
    url, key = env.get("CDS_URL", ""), env.get("CDS_KEY", "")
    if not url or not key:
        sys.exit("HATA: secrets.env icinde CDS_URL / CDS_KEY eksik.")
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
    if not p.exists() or p.stat().st_size < 1024:
        return False
    head = p.open("rb").read(8)
    return head.startswith(HDF5_MAGIC) or head.startswith(ZIP_MAGIC)


def make_request(dataset: str, variable: str, stat: str,
                 year: int, months: list[str]) -> dict:
    req = {
        "variable": [variable],
        "year": [str(year)],
        "month": months,
        "day": DAYS,
        "daily_statistic": stat,
        "time_zone": TIME_ZONE,
        "frequency": FREQUENCY,
        "area": AREA,
    }
    if dataset == DS_ERA5:
        req["product_type"] = "reanalysis"     # ERA5-Land'e GONDERILMEZ
    return req


def retrieve(client, dataset: str, req: dict, target: Path,
             secret: str) -> tuple[bool, str]:
    delay = 10.0
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, MAX_RETRY + 1):
        try:
            client.retrieve(dataset, req, str(part))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if secret and len(secret) > 4:
                msg = msg.replace(secret, "<GIZLI>")
            low = msg.lower()
            if "cost" in low or "too large" in low or "limit" in low:
                part.unlink(missing_ok=True)
                return False, "COST_LIMIT"
            if attempt < MAX_RETRY:
                time.sleep(delay + random.uniform(0, 5))
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
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    a = ap.parse_args()

    import cdsapi
    url, key = read_env()
    years = a.years or YEARS

    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=url, key=key, quiet=True,
                           wait_until_complete=True, delete=False)

    # ------------------------------------------------------------ probe
    if a.probe:
        pdir = RAW / "_probe"
        pdir.mkdir(exist_ok=True)
        print("13 seri tek gunluk isteklerle sinaniyor (2021-07-28)\n")
        ok, bad = [], []
        for ds, var, stat in MATRIX:
            tag = f"{'LAND' if ds == DS_LAND else 'ERA5'} {var} {stat}"
            t = pdir / f"probe_{var}_{SHORT[stat]}.nc"
            print(f"  {tag:62s} ...", end=" ", flush=True)
            if valid_container(t):
                print("zaten var"); ok.append(tag); continue
            req = make_request(ds, var, stat, 2021, ["07"])
            req["day"] = ["28"]
            good, note = retrieve(client, ds, req, t, key)
            print("GECERLI" if good else f"HATA -> {note}")
            (ok if good else bad).append(tag)
        print(f"\nGecerli: {len(ok)}/{len(MATRIX)}")
        if bad:
            print("Sorunlu seriler:")
            for b in bad:
                print("  -", b)
        (META / "probe_result.json").write_text(json.dumps(
            {"checked_at_utc": utc(), "ok": ok, "failed": bad},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # ------------------------------------------------------ tam indirme
    total = len(MATRIX) * len(years)
    print(f"Toplam mantiksal urun: {total}  ({len(MATRIX)} seri x {len(years)} yil)")
    print("Maliyet siniri asilirsa istek otomatik ceyreklere bolunur.\n")

    manifest = META / "request-manifest.jsonl"
    counters = {"indirildi": 0, "atlandi": 0, "ceyrek": 0, "hata": 0}
    gaps: list[dict] = []
    i = 0

    for ds, var, stat in MATRIX:
        prefix = "era5_land" if ds == DS_LAND else "era5"
        vdir = RAW / f"{prefix}_{var}_{SHORT[stat]}"
        vdir.mkdir(exist_ok=True)

        for y in years:
            i += 1
            tag = f"[{i}/{total}] {prefix} {var} {SHORT[stat]} {y}"
            target = vdir / f"{prefix}_{y}_{SHORT[stat]}_{var}_turkey.nc"

            if valid_container(target):
                counters["atlandi"] += 1
                print(f"{tag} zaten var")
                continue

            print(f"{tag} ...", end=" ", flush=True)
            t0 = time.time()
            req = make_request(ds, var, stat, y, MONTHS_ALL)
            good, note = retrieve(client, ds, req, target, key)

            if not good and note == "COST_LIMIT":
                print("maliyet siniri -> ceyreklere bolunuyor")
                qok = 0
                for q, months in QUARTERS.items():
                    qt = vdir / f"{prefix}_{y}_q{q}_{SHORT[stat]}_{var}_turkey.nc"
                    if valid_container(qt):
                        qok += 1
                        continue
                    qreq = make_request(ds, var, stat, y, months)
                    g2, n2 = retrieve(client, ds, qreq, qt, key)
                    print(f"    q{q}: {'ok' if g2 else n2}")
                    if g2:
                        qok += 1
                counters["ceyrek" if qok == 4 else "hata"] += 1
                if qok < 4:
                    gaps.append({"dataset": prefix, "variable": var,
                                 "statistic": stat, "year": y,
                                 "status": "partial_quarters", "ok": qok})
                continue

            if not good:
                counters["hata"] += 1
                print(f"HATA ({note})")
                gaps.append({"dataset": prefix, "variable": var,
                             "statistic": stat, "year": y,
                             "status": "failed", "note": note})
                continue

            counters["indirildi"] += 1
            size = target.stat().st_size
            print(f"{size / 1e6:.1f} MB  ({(time.time() - t0) / 60:.1f} dk)")

            rec = {
                "source": "ERA5 / ERA5-Land daily statistics",
                "provider": "Copernicus C3S / ECMWF (CDS)",
                "dataset_or_product_id": ds,
                "doi": ("10.24381/cds.e9c9c792" if ds == DS_LAND
                        else "10.24381/cds.4991cf48"),
                "variable": var, "daily_statistic": stat,
                "retrieved_at_utc": utc(),
                "temporal_start": f"{y}-01-01", "temporal_end": f"{y}-12-31",
                "bbox_or_aoi": AREA, "crs": "EPSG:4326",
                "time_zone": TIME_ZONE, "frequency": FREQUENCY,
                "original_filename": target.name, "bytes": size,
                "sha256": sha256_of(target),
                "license": "CC BY 4.0",
                "request": req,
            }
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if gaps:
        (REPORT / "era5_gaps.json").write_text(
            json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")

    found = sum(1 for p in RAW.rglob("*.nc") if valid_container(p))
    (REPORT / "era5_status.json").write_text(json.dumps({
        "source": "ERA5 / ERA5-Land", "generated_at_utc": utc(),
        "series": len(MATRIX), "years": years,
        "expected_logical_products": total,
        "found_physical_files": found,
        "area": AREA, "time_zone": TIME_ZONE, "frequency": FREQUENCY,
        "counters": counters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- OZET ---")
    for k, v in counters.items():
        print(f"{k:12s}: {v}")
    print(f"{'beklenen':12s}: {total}")
    print(f"{'dosya':12s}: {found}")
    print(f"\nRapor: {REPORT / 'era5_status.json'}")


if __name__ == "__main__":
    main()
