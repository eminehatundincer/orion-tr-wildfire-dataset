"""
ORION-TR / Adim 9
Copernicus DEM GLO-30 karolarini indirir ve 1 km grid hucrelerinde topografya
oznitelikleri hesaplar.

Neden onemli:
  - Yangin yamac yukari hizli yayilir  -> egim
  - Guney bakili yamaclar daha sicak ve kuru -> baki (kuzey yarimkure)
  - Engebeli arazi mudahaleyi zorlastirir -> puruzluluk

Hizalama:
  DEM pikseli 1 yay saniyesi = 1/3600 derece.
  0.01 derece hucre = tam 36 x 36 piksel. 1 derecelik karo = tam 100 x 100 hucre.
  Grid sinirlari (25.50 / 42.50) 0.01'in tam kati oldugu icin kayma yok.

Lisans: Copernicus DEM, kamuya acik.
Atif: Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and
      (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS
      by the European Union and ESA.

Cikti:
  data_raw/dem/*.tif
  data_derived/grid_1km_terrain.parquet
  _reports/terrain_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_terrain.py
    .venv\\Scripts\\python.exe scripts\\build_terrain.py --skip-download
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
DEM_DIR = ROOT / "data_raw" / "dem"
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
GRID_FILE = DERIVED / "grid_1km_landcover.parquet"
OUT = DERIVED / "grid_1km_terrain.parquet"

BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
NAME = "Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"

GRID_WEST, GRID_NORTH = 25.50, 42.50
CELL_DEG = 0.01
NCOLS, NROWS = 1950, 700
PX_PER_DEG = 3600
PX_PER_CELL = 36                     # 0.01 * 3600

LAT_RANGE = range(35, 43)            # 35N .. 42N (karo alt kenari)
LON_RANGE = range(25, 46)            # 25E .. 45E


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return NAME.format(ns=ns, ew=ew)


def download() -> list[Path]:
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    tiles = [(la, lo) for la in LAT_RANGE for lo in LON_RANGE]
    print(f"DEM karolari indiriliyor ({len(tiles)} aday, deniz karolari yok)...")

    for i, (la, lo) in enumerate(tiles, 1):
        stem = tile_name(la, lo)
        final = DEM_DIR / f"{stem}.tif"
        if final.exists() and final.stat().st_size > 0:
            got.append(final)
            continue

        url = f"{BASE}/{stem}/{stem}.tif"
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                part = final.with_suffix(".tif.part")
                total = 0
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        total += len(chunk)
                with rasterio.open(part):
                    pass
                part.replace(final)
                got.append(final)
                print(f"  [{i}/{len(tiles)}] {stem}  {total / 1e6:.0f} MB")
        except Exception as e:
            print(f"  [{i}/{len(tiles)}] {stem}  HATA: {type(e).__name__}")

    print(f"\nIndirilen/mevcut karo: {len(got)}")
    return sorted(got)


def terrain_from_tile(path: Path):
    """Bir karodan hucre bazli topografya istatistikleri uretir."""
    with rasterio.open(path) as src:
        tr = src.transform
        lon0, lat_top = tr.c, tr.f
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -400] = np.nan          # bariz hatali degerler

    H, W = arr.shape
    H -= H % PX_PER_CELL
    W -= W % PX_PER_CELL
    arr = arr[:H, :W]
    if H == 0 or W == 0:
        return None

    # metrik piksel araligi
    lat_mid = lat_top - (H / 2) / PX_PER_DEG
    dy = 111_320.0 / PX_PER_DEG
    dx = dy * math.cos(math.radians(lat_mid))

    dz_dy, dz_dx = np.gradient(arr, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.arctan2(dz_dy, -dz_dx)        # radyan

    nr, nc = H // PX_PER_CELL, W // PX_PER_CELL
    def blocks(a):
        return a.reshape(nr, PX_PER_CELL, nc, PX_PER_CELL)

    with np.errstate(invalid="ignore"):
        elev_mean = np.nanmean(blocks(arr), axis=(1, 3))
        elev_std = np.nanstd(blocks(arr), axis=(1, 3))
        elev_max = np.nanmax(blocks(arr), axis=(1, 3))
        elev_min = np.nanmin(blocks(arr), axis=(1, 3))
        slope_mean = np.nanmean(blocks(slope), axis=(1, 3))
        slope_max = np.nanmax(blocks(slope), axis=(1, 3))
        asp_sin = np.nanmean(blocks(np.sin(aspect)), axis=(1, 3))
        asp_cos = np.nanmean(blocks(np.cos(aspect)), axis=(1, 3))

    col_off = int(round((lon0 - GRID_WEST) * PX_PER_DEG)) // PX_PER_CELL
    row_off = int(round((GRID_NORTH - lat_top) * PX_PER_DEG)) // PX_PER_CELL

    rows = np.arange(nr) + row_off
    cols = np.arange(nc) + col_off
    in_r = (rows >= 0) & (rows < NROWS)
    in_c = (cols >= 0) & (cols < NCOLS)
    if not in_r.any() or not in_c.any():
        return None

    rr, cc = rows[in_r], cols[in_c]
    sel = np.ix_(in_r, in_c)
    cell = (rr[:, None] * NCOLS + cc[None, :]).ravel()

    out = pd.DataFrame({
        "cell_id": cell.astype(np.int32),
        "elev_mean": elev_mean[sel].ravel(),
        "elev_std": elev_std[sel].ravel(),
        "elev_range": (elev_max - elev_min)[sel].ravel(),
        "slope_mean": slope_mean[sel].ravel(),
        "slope_max": slope_max[sel].ravel(),
        "aspect_sin": asp_sin[sel].ravel(),
        "aspect_cos": asp_cos[sel].ravel(),
    })
    return out[np.isfinite(out["elev_mean"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    tifs = sorted(DEM_DIR.glob("*.tif")) if a.skip_download else download()
    if not tifs:
        raise SystemExit("HATA: hic DEM karosu yok.")

    (DEM_DIR / "source.json").write_text(json.dumps({
        "source": "Copernicus DEM GLO-30 Public",
        "provider": "ESA / Airbus / DLR, AWS Open Data (Sinergise)",
        "attribution": "Produced using Copernicus WorldDEM-30 (c) DLR e.V. "
                       "2010-2014 and (c) Airbus Defence and Space GmbH "
                       "2014-2018, provided under COPERNICUS by the European "
                       "Union and ESA",
        "crs": "EPSG:4326 (yatay), EGM2008 (dusey)",
        "spatial_resolution": "1 arcsec (~30 m)",
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tiles": len(tifs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(tifs)} karodan topografya hesaplaniyor...")
    parts = []
    for i, t in enumerate(tifs, 1):
        try:
            r = terrain_from_tile(t)
        except Exception as e:
            print(f"  [{i}/{len(tifs)}] {t.name} HATA: {type(e).__name__}: {e}")
            continue
        if r is not None and len(r):
            parts.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(tifs)}")

    if not parts:
        raise SystemExit("HATA: hic hucre uretilemedi.")

    ter = pd.concat(parts, ignore_index=True)
    ter = ter.groupby("cell_id", as_index=False).mean()   # karo kenari ortusmeleri

    for c in ter.columns:
        if c != "cell_id":
            ter[c] = ter[c].astype(np.float32)
    ter.to_parquet(OUT, index=False)

    # ---- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# Topografya oznitelikleri - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Kaynak: Copernicus DEM GLO-30 (~30 m)")
    log(f"- Karo sayisi: {len(tifs)}")
    log(f"- Uretilen hucre: **{len(ter):,}**")
    log()

    if GRID_FILE.exists():
        grid = pd.read_parquet(GRID_FILE)
        tr_cells = set(grid.loc[grid["in_turkey"], "cell_id"])
        forest = set(grid.loc[grid["orman_maskesi"], "cell_id"])
        have = set(ter["cell_id"])
        log(f"- Turkiye ici hucre: {len(tr_cells):,}, topografyasi olan: "
            f"**{len(tr_cells & have):,}** ({len(tr_cells & have) / len(tr_cells):.1%})")
        log(f"- Orman maskesi: {len(forest):,}, topografyasi olan: "
            f"**{len(forest & have):,}** ({len(forest & have) / len(forest):.1%})")
        missing = forest - have
        if missing:
            log(f"- **Eksik orman hucresi: {len(missing):,}** (DEM karosu yok veya deniz)")
        log()

        m = ter[ter["cell_id"].isin(forest)]
    else:
        m = ter

    log("## Deger dagilimlari (orman maskesi)")
    log()
    log("| Degisken | Min | %25 | Ortanca | %75 | Maks |")
    log("|---|---|---|---|---|---|")
    for c in ("elev_mean", "elev_std", "elev_range", "slope_mean",
              "slope_max", "aspect_sin", "aspect_cos"):
        q = m[c].quantile([0, .25, .5, .75, 1])
        log(f"| {c} | {q.iloc[0]:.1f} | {q.iloc[1]:.1f} | {q.iloc[2]:.1f} | "
            f"{q.iloc[3]:.1f} | {q.iloc[4]:.1f} |")
    log()
    log("> `aspect_cos` = +1 tam kuzey, -1 tam guney bakili.")
    log("> Kuzey yarimkurede guney bakili yamaclar daha sicak ve kuru oldugu icin")
    log("> bu degiskenin yangin olasiligiyla negatif korelasyonu beklenir.")
    log("> Ortanca degerin 0'a yakin cikmasi normaldir (bakilar dengeli dagilir).")
    log()
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")
    log(f"- Sutunlar: {', '.join(ter.columns)}")

    (REPORT / "terrain_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'terrain_qa.md'}")


if __name__ == "__main__":
    main()
