"""
ORION-TR / Adim 5
Turkiye icin 0.01 derece (~1 km) grid kurar ve her hucrede ESA WorldCover
sinif oranlarini hesaplar.

Neden hucre orani, nokta ornekleme degil:
  VIIRS pikseli 375 m, MODIS 1 km. 10 m'de tek nokta okumak orman kenarindaki
  bir yangini "tarim" gosterebilir. Hucre bazinda oran hem daha saglam hem de
  modelin gercekten kullanacagi ozniteliktir (ornegin "hucrenin %65'i agac").

Hizalama:
  WorldCover pikseli 1/12000 derece. 0.01 derece = tam 120 piksel.
  Grid bati sinirini 25.50, kuzey sinirini 42.50 aldigimizda (ikisi de 0.01'in
  tam kati) hucre sinirlari piksel sinirlariyla birebir ortusur.

Cikti:
  data_derived/grid_1km_landcover.parquet
  _reports/grid_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_grid.py
    .venv\\Scripts\\python.exe scripts\\build_grid.py --block-rows 600
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
LC_DIR = ROOT / "data_raw" / "landcover" / "worldcover"
BOUNDARY = ROOT / "data_raw" / "boundary" / "tur_adm0.geojson"
DERIVED = ROOT / "data_derived"
REPORT_DIR = ROOT / "_reports"
OUT = DERIVED / "grid_1km_landcover.parquet"

# Grid tanimi
GRID_WEST, GRID_NORTH = 25.50, 42.50
GRID_EAST, GRID_SOUTH = 45.00, 35.50
CELL_DEG = 0.01
PX_PER_DEG = 12000                      # WorldCover: 1/12000 derece
PX_PER_CELL = int(round(CELL_DEG * PX_PER_DEG))   # = 120

NCOLS = int(round((GRID_EAST - GRID_WEST) / CELL_DEG))    # 1950
NROWS = int(round((GRID_NORTH - GRID_SOUTH) / CELL_DEG))  # 700
NCELLS = NROWS * NCOLS

CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
NAMES = {
    10: "agac", 20: "maki", 30: "otlak", 40: "tarim", 50: "yapili",
    60: "ciplak", 70: "karbuz", 80: "su", 90: "sulak", 95: "mangrov",
    100: "yosun",
}
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
NCLS = len(CLASSES)

# 0-100 arasi kod -> sinif indeksi tablosu (hizli vektorel esleme)
LUT = np.full(256, -1, dtype=np.int16)
for c, i in CLASS_INDEX.items():
    LUT[c] = i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-rows", type=int, default=1200,
                    help="Tek seferde okunacak piksel satiri (120'nin kati olmali)")
    args = ap.parse_args()
    block = args.block_rows - (args.block_rows % PX_PER_CELL)
    if block <= 0:
        block = PX_PER_CELL * 10

    tifs = sorted(LC_DIR.glob("*.tif"))
    if not tifs:
        raise SystemExit(f"HATA: {LC_DIR} altinda karo yok.")
    if not BOUNDARY.exists():
        raise SystemExit(f"HATA: {BOUNDARY} yok.")

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {NROWS} x {NCOLS} = {NCELLS:,} hucre ({CELL_DEG} derece)")
    print(f"Karo sayisi: {len(tifs)}, blok yuksekligi: {block} piksel\n")

    counts = np.zeros(NCELLS * NCLS, dtype=np.int32)

    for ti, tif in enumerate(tifs, 1):
        with rasterio.open(tif) as src:
            tr = src.transform
            lon0, lat0 = tr.c, tr.f
            H, W = src.height, src.width

            # Piksel indeksinden grid hucresine sabit tam sayi kaymasi
            col_off = int(round((lon0 - GRID_WEST) * PX_PER_DEG))
            row_off = int(round((GRID_NORTH - lat0) * PX_PER_DEG))

            print(f"  [{ti}/{len(tifs)}] {tif.name}", end="", flush=True)
            used = 0

            for r0 in range(0, H, block):
                h = min(block, H - r0)
                arr = src.read(1, window=Window(0, r0, W, h))

                cls_idx = LUT[arr]
                valid = cls_idx >= 0
                if not valid.any():
                    continue

                rows = (np.arange(r0, r0 + h, dtype=np.int64) + row_off) // PX_PER_CELL
                cols = (np.arange(W, dtype=np.int64) + col_off) // PX_PER_CELL

                in_r = (rows >= 0) & (rows < NROWS)
                in_c = (cols >= 0) & (cols < NCOLS)
                if not in_r.any() or not in_c.any():
                    continue

                sub = cls_idx[np.ix_(in_r, in_c)]
                rr = rows[in_r]
                cc = cols[in_c]

                cell = (rr[:, None] * NCOLS + cc[None, :])
                m = sub >= 0
                flat = (cell[m].astype(np.int64) * NCLS + sub[m].astype(np.int64))
                counts += np.bincount(flat, minlength=NCELLS * NCLS).astype(np.int32)
                used += int(m.sum())

            print(f"  ({used / 1e6:.0f}M piksel)")

    counts = counts.reshape(NCELLS, NCLS)
    total = counts.sum(axis=1)
    keep = total > 0
    print(f"\nVeri iceren hucre: {keep.sum():,} / {NCELLS:,}")

    idx = np.nonzero(keep)[0]
    cell_row = (idx // NCOLS).astype(np.int32)
    cell_col = (idx % NCOLS).astype(np.int32)

    df = pd.DataFrame({
        "cell_id": idx.astype(np.int32),
        "cell_row": cell_row,
        "cell_col": cell_col,
        "lat_center": (GRID_NORTH - (cell_row + 0.5) * CELL_DEG).astype(np.float32),
        "lon_center": (GRID_WEST + (cell_col + 0.5) * CELL_DEG).astype(np.float32),
        "n_pixels": total[keep].astype(np.int32),
    })
    for c in CLASSES:
        df[f"frac_{NAMES[c]}"] = (
            counts[keep, CLASS_INDEX[c]] / total[keep]
        ).astype(np.float32)

    # ---- Turkiye ici filtresi
    print("Turkiye sinirina gore isaretleniyor...")
    tur = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    geom = tur.union_all() if hasattr(tur, "union_all") else tur.unary_union
    pts = gpd.GeoDataFrame(
        df[["cell_id"]],
        geometry=gpd.points_from_xy(df["lon_center"], df["lat_center"]),
        crs="EPSG:4326",
    )
    df["in_turkey"] = pts.geometry.within(geom).values

    # ---- yanici ortu maskesi
    df["frac_dogal"] = (df["frac_agac"] + df["frac_maki"]).astype(np.float32)
    df["frac_yanici"] = (df["frac_dogal"] + df["frac_otlak"]).astype(np.float32)
    df["orman_maskesi"] = (df["in_turkey"] & (df["frac_dogal"] >= 0.30))

    df.to_parquet(OUT, index=False)

    # ---- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    tr_df = df[df["in_turkey"]]
    log("# 1 km grid ve arazi ortusu oranlari - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Grid: {CELL_DEG} derece, {NROWS} x {NCOLS}, EPSG:4326")
    log(f"- Hizalama: 1 hucre = {PX_PER_CELL} x {PX_PER_CELL} WorldCover pikseli")
    log()
    log(f"- Veri iceren hucre: **{len(df):,}**")
    log(f"- Turkiye ici hucre: **{len(tr_df):,}**")
    log(f"- Orman maskesi (dogal ortu >= %30): **{int(df['orman_maskesi'].sum()):,}**")
    log()

    log("## Turkiye ici ortalama ortu bilesimi")
    log()
    log("| Sinif | Ortalama oran | Baskin oldugu hucre |")
    log("|---|---|---|")
    fr = [f"frac_{NAMES[c]}" for c in CLASSES]
    dom = tr_df[fr].idxmax(axis=1).value_counts()
    for c in CLASSES:
        col = f"frac_{NAMES[c]}"
        log(f"| {NAMES[c]} | {tr_df[col].mean():.1%} | {dom.get(col, 0):,} |")
    log()

    log("## Dogal ortu orani dagilimi (Turkiye ici)")
    log()
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30),
                   (0.30, 0.50), (0.50, 0.75), (0.75, 1.01)]:
        n = int(((tr_df["frac_dogal"] >= lo) & (tr_df["frac_dogal"] < hi)).sum())
        log(f"- %{lo * 100:.0f}-%{hi * 100:.0f}: {n:,} hucre ({n / len(tr_df):.1%})")
    log()
    log("> Esik %30 secildi. Yangin riski modelinde yalniz bu hucreler kullanilacak;")
    log("> boylece 1.4 milyon hucrelik uzaydan yaklasik dortte birine inilir ve")
    log("> negatif ornekleme tarim/sehir hucreleriyle kirlenmez.")
    log()
    log(f"## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")

    (REPORT_DIR / "grid_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT_DIR / 'grid_qa.md'}")


if __name__ == "__main__":
    main()
