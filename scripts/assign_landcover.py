"""
ORION-TR / Adim 4b
ESA WorldCover 10 m (2021 v200) karolarini indirir ve her FIRMS tespitine
arazi ortusu sinifi atar.

Amac: type=0 "vejetasyon yangini" icindeki tarimsal aniz yakmayi orman/maki
yangininden ayirmak. Ekim ayi zirvesi bu ayrim yapilmadan cozulmez.

Lisans: ESA WorldCover, CC BY 4.0, ucretsiz.
Atif: (c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel
data (2021) processed by ESA WorldCover consortium.

Cikti:
  data_raw/landcover/worldcover/*.tif
  data_derived/firms_events_turkey_lc.parquet
  _reports/firms_landcover_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\assign_landcover.py
    .venv\\Scripts\\python.exe scripts\\assign_landcover.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
LC_DIR = ROOT / "data_raw" / "landcover" / "worldcover"
REPORT_DIR = ROOT / "_reports"

EVENTS = DERIVED / "firms_events_turkey.parquet"
OUT = DERIVED / "firms_events_turkey_lc.parquet"

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
FNAME = "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"

WEST, SOUTH, EAST, NORTH = 25.5, 35.5, 45.0, 42.5

CLASSES = {
    10: "agac_ortusu",
    20: "calilik_maki",
    30: "otlak",
    40: "tarim_alani",
    50: "yapili_alan",
    60: "ciplak_seyrek",
    70: "kar_buz",
    80: "daimi_su",
    90: "otsu_sulak",
    95: "mangrov",
    100: "yosun_liken",
}

# Orman yangini etiketi icin dogal yanici ortu
NATURAL = {10, 20}
AMBIGUOUS = {30}


def tiles_for_bbox() -> list[str]:
    lats = range(int(math.floor(SOUTH / 3) * 3), int(math.floor(NORTH / 3) * 3) + 1, 3)
    lons = range(int(math.floor(WEST / 3) * 3), int(math.floor(EAST / 3) * 3) + 1, 3)
    out = []
    for la in lats:
        for lo in lons:
            ns = f"N{la:02d}" if la >= 0 else f"S{abs(la):02d}"
            ew = f"E{lo:03d}" if lo >= 0 else f"W{abs(lo):03d}"
            out.append(f"{ns}{ew}")
    return out


def download_tiles(tiles: list[str]) -> list[Path]:
    LC_DIR.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for i, t in enumerate(tiles, 1):
        name = FNAME.format(tile=t)
        final = LC_DIR / name
        if final.exists() and final.stat().st_size > 0:
            print(f"  [{i}/{len(tiles)}] {t} zaten var")
            got.append(final)
            continue

        url = f"{BASE}/{name}"
        print(f"  [{i}/{len(tiles)}] {t} ...", end=" ", flush=True)
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code == 404:
                    print("yok (kara alani degil)")
                    continue
                r.raise_for_status()
                part = final.with_suffix(".tif.part")
                total = 0
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        total += len(chunk)
                with rasterio.open(part):   # dogrulama
                    pass
                part.replace(final)
                print(f"{total / 1e6:.0f} MB")
                got.append(final)
        except Exception as e:
            print(f"HATA: {type(e).__name__}")
    return got


def sample_classes(df: pd.DataFrame, tif_paths: list[Path]) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int16)
    lon = df["longitude"].to_numpy()
    lat = df["latitude"].to_numpy()

    for p in tif_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            m = (lon >= b.left) & (lon < b.right) & (lat >= b.bottom) & (lat < b.top)
            idx = np.nonzero(m & (out == 0))[0]
            if idx.size == 0:
                continue
            coords = list(zip(lon[idx], lat[idx]))
            vals = np.fromiter(
                (v[0] for v in src.sample(coords)), dtype=np.int16, count=len(coords)
            )
            out[idx] = vals
            print(f"  {p.name}: {idx.size:,} nokta ornedlendi")
    return out


def dist_table(f: pd.DataFrame, log) -> None:
    log("| Ay | " + " | ".join(str(i) for i in range(1, 13)) + " | Toplam |")
    log("|---" * 14 + "|")
    ay = pd.to_datetime(f["acq_date"]).dt.month.value_counts().sort_index()
    log("| Kayit | " + " | ".join(f"{ay.get(i, 0):,}" for i in range(1, 13))
        + f" | {len(f):,} |")
    log()
    yil = pd.to_datetime(f["acq_date"]).dt.year.value_counts().sort_index()
    log("| Yil | " + " | ".join(str(y) for y in yil.index) + " |")
    log("|---" * (len(yil) + 1) + "|")
    log("| Kayit | " + " | ".join(f"{v:,}" for v in yil.values) + " |")
    log()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not EVENTS.exists():
        raise SystemExit(f"HATA: {EVENTS} yok. Once clip_firms_turkey.py calistirin.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    tiles = tiles_for_bbox()
    print(f"Gerekli karo: {len(tiles)} -> {', '.join(tiles)}\n")

    if args.skip_download:
        paths = sorted(LC_DIR.glob("*.tif"))
        print(f"Indirme atlandi, mevcut {len(paths)} karo kullanilacak.\n")
    else:
        print("WorldCover karolari indiriliyor (~1.5-2 GB, ilk sefer uzun surer)...")
        paths = download_tiles(tiles)
        print()

    if not paths:
        raise SystemExit("HATA: hic karo yok.")

    (LC_DIR.parent / "source.json").write_text(json.dumps({
        "source": "ESA WorldCover 10 m 2021 v200",
        "provider": "ESA / VITO",
        "doi": "10.5281/zenodo.7254221",
        "license": "CC BY 4.0",
        "attribution": "(c) ESA WorldCover project 2021 / Contains modified "
                       "Copernicus Sentinel data (2021) processed by ESA "
                       "WorldCover consortium",
        "crs": "EPSG:4326",
        "spatial_resolution": "10 m",
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tiles": [p.name for p in paths],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_parquet(EVENTS)
    print(f"Olay sayisi: {len(df):,}\nOrnekleme...")
    df["worldcover_class"] = sample_classes(df, paths)
    df["worldcover_label"] = df["worldcover_class"].map(CLASSES).fillna("ornekleneMedi")
    df.to_parquet(OUT, index=False)
    print()

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# FIRMS arazi ortusu maskesi - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log("- Kaynak: ESA WorldCover 10 m 2021 v200 (CC BY 4.0)")
    log(f"- Girdi: `{EVENTS.name}`, {len(df):,} kayit")
    log()
    log("> Not: nokta ornekleme 10 m cozunurlukte yapildi; VIIRS pikseli 375 m'dir.")
    log("> Kenar etkileri icin 1 km grid asamasinda ortu orani hesaplanacaktir.")
    log()

    log("## Sinif dagilimi")
    log()
    log("| Kod | Sinif | Kayit | Oran |")
    log("|---|---|---|---|")
    vc = df["worldcover_class"].value_counts().sort_index()
    for k, v in vc.items():
        log(f"| {k} | {CLASSES.get(int(k), 'ornekleneMedi')} | {v:,} | {v / len(df):.1%} |")
    log()

    nat = df[df["worldcover_class"].isin(NATURAL)]
    amb = df[df["worldcover_class"].isin(AMBIGUOUS)]
    agr = df[df["worldcover_class"] == 40]

    log(f"- Dogal yanici ortu (agac + maki): **{len(nat):,}** ({len(nat) / len(df):.1%})")
    log(f"- Otlak (belirsiz): **{len(amb):,}** ({len(amb) / len(df):.1%})")
    log(f"- Tarim alani: **{len(agr):,}** ({len(agr) / len(df):.1%})")
    log()

    log("## Dogal ortu (10+20) - ay ve yil dagilimi")
    log()
    dist_table(nat, log)

    log("## Tarim alani (40) - ay ve yil dagilimi")
    log()
    dist_table(agr, log)

    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB) — tum kayitlar, sinif sutunu eklendi")
    log("- Filtreleme yapilmadi; ayiklama karari bir sonraki adimda verilecek.")

    (REPORT_DIR / "firms_landcover_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT_DIR / 'firms_landcover_qa.md'}")


if __name__ == "__main__":
    main()
