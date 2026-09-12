"""
ORION-TR / Adim 13
ERA5-Land AYLIK toprak nemi ve bitki ortusu degiskenlerini indirir.

Neden aylik:
  FWI ailesi (FFMC, DMC, DC, ISI, BUI) gunluk sicaklik, nem, ruzgar ve
  yagistan hesaplanir; yani gunluk meteorolojik dinamik zaten veri setinde.
  ERA5'in yeni kattigi bilgi toprak nemi ve bitki ortusu yogunlugudur ve
  bunlar yavas degisen buyukluklerdir. Derin toprak nemi (28-100 cm, 100-289 cm)
  ile LAI gunluk degil mevsimsel olcekte anlamlidir.

  Pratik kazanc: gunluk urunde 130 ayri istek ve ongorulemez kuyruk suresi
  gerekirken, aylik urunde 10 yilin tamami TEK istekte gelir.

Degiskenler:
  volumetric_soil_water_layer_1   0-7 cm     yuzey yakit nemi
  volumetric_soil_water_layer_2   7-28 cm    kok bolgesi
  volumetric_soil_water_layer_3   28-100 cm  derin kuraklik
  volumetric_soil_water_layer_4   100-289 cm uzun donem su dengesi
  leaf_area_index_high_vegetation agac ortusu yaprak alani
  leaf_area_index_low_vegetation  cali/ot ortusu yaprak alani
  skin_temperature                yuzey sicakligi

Cikti:
  data_raw/era5_monthly/era5_land_monthly_turkey.nc
  data_raw/era5_monthly/metadata/request.json

Kullanim:
    .venv\\Scripts\\python.exe scripts\\download_era5_monthly.py
    .venv\\Scripts\\python.exe scripts\\download_era5_monthly.py --split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "config" / "secrets.env"
RAW = ROOT / "data_raw" / "era5_monthly"
META = RAW / "metadata"

DATASET = "reanalysis-era5-land-monthly-means"
AREA = [42.5, 25.5, 35.5, 45.0]
YEARS = [str(y) for y in range(2016, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

VARIABLES = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "skin_temperature",
]

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
        sys.exit("HATA: CDS_URL / CDS_KEY eksik.")
    return url, key


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def valid(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 1024:
        return False
    head = p.open("rb").read(8)
    return head.startswith(HDF5_MAGIC) or head.startswith(ZIP_MAGIC)


def make_request(variables: list[str]) -> dict:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variables,
        "year": YEARS,
        "month": MONTHS,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def fetch(client, variables: list[str], target: Path, key: str) -> bool:
    part = target.with_suffix(target.suffix + ".part")
    req = make_request(variables)
    try:
        client.retrieve(DATASET, req, str(part))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if key and len(key) > 4:
            msg = msg.replace(key, "<GIZLI>")
        print(f"  HATA: {msg[:400]}")
        part.unlink(missing_ok=True)
        return False

    if not valid(part):
        bad = target.with_suffix(f".invalid.{utc().replace(':', '')}")
        part.replace(bad)
        print(f"  gecersiz container -> {bad.name}")
        return False

    part.replace(target)
    print(f"  OK -> {target.name}  ({target.stat().st_size / 1e6:.1f} MB)")

    META.mkdir(parents=True, exist_ok=True)
    rec = {
        "source": "ERA5-Land monthly averaged reanalysis",
        "provider": "Copernicus C3S / ECMWF (CDS)",
        "dataset_or_product_id": DATASET,
        "doi": "10.24381/cds.68d2bb30",
        "retrieved_at_utc": utc(),
        "temporal_start": "2016-01", "temporal_end": "2025-12",
        "bbox_or_aoi": AREA, "crs": "EPSG:4326",
        "spatial_resolution": "~0.1 derece",
        "original_filename": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_of(target),
        "license": "CC BY 4.0",
        "request": req,
    }
    out = META / f"{target.stem}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                    help="Her degiskeni ayri istekte cek (tek istek reddedilirse)")
    a = ap.parse_args()

    import cdsapi
    url, key = read_env()
    RAW.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=url, key=key, quiet=True,
                           wait_until_complete=True, delete=False)

    if a.split:
        print(f"{len(VARIABLES)} degisken ayri ayri cekiliyor...\n")
        ok = 0
        for v in VARIABLES:
            t = RAW / f"era5_land_monthly_{v}_turkey.nc"
            print(f"{v} ...")
            if valid(t):
                print("  zaten var")
                ok += 1
                continue
            if fetch(client, [v], t, key):
                ok += 1
        print(f"\nTamamlanan: {ok}/{len(VARIABLES)}")
        return

    target = RAW / "era5_land_monthly_turkey.nc"
    if valid(target):
        print(f"Zaten var: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return

    print("ERA5-Land aylik veri, tek istek")
    print(f"  degisken : {len(VARIABLES)}")
    print(f"  donem    : 2016-01 .. 2025-12 (120 ay)")
    print(f"  alan     : {AREA}")
    print("\nKuyruga giriliyor, bekleyin...\n")

    if not fetch(client, VARIABLES, target, key):
        print("\nTek istek basarisiz. Degisken basina bolerek deneyin:")
        print("  .venv\\Scripts\\python.exe scripts\\download_era5_monthly.py --split")
        return

    print("\nTamam. Sirada: join_era5_monthly.py")


if __name__ == "__main__":
    main()
