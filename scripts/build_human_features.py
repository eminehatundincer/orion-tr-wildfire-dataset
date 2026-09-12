"""
ORION-TR / Adim 11
Insan faktoru ve yakit surekliligi oznitelikleri.

Yeni indirme yok: hepsi mevcut 1 km arazi ortusu gridinden turetilir.
Bu hem daha hizli hem de tutarli, cunku ayni kaynagin ayni cozunurlugunden gelir.

Uretilen degiskenler:

  dist_built_km    en yakin yapili alana mesafe
                   Turkiye'de tutusmalarin buyuk cogunlugu insan kaynakli;
                   yerlesime yakinlik en guclu prediktorlerden biridir.

  dist_crop_km     en yakin tarim alanina mesafe
                   Aniz yakma ormana sicriyor; orman-tarim siniri kritik bolge.

  dist_water_km    en yakin su kutlesine mesafe
                   Mudahale lojistigi ve yerel nem.

  natural_5x5      5 km cevrede dogal ortu orani
                   Yakit surekliligi: yuksekse yangin engelsiz yayilir.

  built_5x5        5 km cevrede yapili alan orani
                   Insan baskisi yogunlugu.

  crop_edge        hucrenin orman-tarim siniri uzerinde olup olmadigi
                   Kendi ortusu dogal ama komsulugunda tarim varsa 1.

Yontem: scipy.ndimage.distance_transform_edt ve uniform_filter, grid
matrisinde vektorel olarak. Mesafeler hucre biriminden kilometreye cevrilir
(enleme bagli boylam kisalmasi hesaba katilir).

Cikti:
  data_derived/grid_1km_human.parquet
  _reports/human_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_human_features.py
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
GRID = DERIVED / "grid_1km_landcover.parquet"
OUT = DERIVED / "grid_1km_human.parquet"

NROWS, NCOLS = 700, 1950
GRID_NORTH, CELL_DEG = 42.50, 0.01

KM_PER_DEG_LAT = 111.32
WIN = 5                      # 5 x 5 hucre ~ 5 km pencere


def to_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    m = np.zeros(NROWS * NCOLS, dtype=np.float32)
    m[df["cell_id"].to_numpy()] = df[col].to_numpy(dtype=np.float32)
    return m.reshape(NROWS, NCOLS)


def main() -> None:
    if not GRID.exists():
        raise SystemExit(f"HATA: {GRID} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    grid = pd.read_parquet(GRID)
    print(f"Grid okundu: {len(grid):,} hucre")

    built = to_matrix(grid, "frac_yapili")
    crop = to_matrix(grid, "frac_tarim")
    water = to_matrix(grid, "frac_su")
    natural = to_matrix(grid, "frac_dogal")

    # ---- mesafe donusumleri (hucre birimi)
    print("Mesafe donusumleri...")
    # EDT sifir olmayan hucrelerden uzakligi olcer, bu yuzden hedefi 0 yapiyoruz
    dist_built = ndimage.distance_transform_edt(built < 0.05)
    dist_crop = ndimage.distance_transform_edt(crop < 0.10)
    dist_water = ndimage.distance_transform_edt(water < 0.10)

    # ---- pencere ortalamalari
    print("Pencere ortalamalari...")
    nat_win = ndimage.uniform_filter(natural, size=WIN, mode="nearest")
    built_win = ndimage.uniform_filter(built, size=WIN, mode="nearest")
    crop_win = ndimage.uniform_filter(crop, size=WIN, mode="nearest")

    # ---- hucre -> km donusumu (enleme bagli)
    rows = np.arange(NROWS)
    lat = GRID_NORTH - (rows + 0.5) * CELL_DEG
    km_lat = CELL_DEG * KM_PER_DEG_LAT                      # ~1.113 km
    km_lon = km_lat * np.cos(np.radians(lat))               # satira gore degisir
    km_cell = np.sqrt(km_lat * km_lon)[:, None]             # geometrik ortalama

    cid = grid["cell_id"].to_numpy()
    r, c = cid // NCOLS, cid % NCOLS

    out = pd.DataFrame({
        "cell_id": cid.astype(np.int32),
        "dist_built_km": (dist_built * km_cell)[r, c].astype(np.float32),
        "dist_crop_km": (dist_crop * km_cell)[r, c].astype(np.float32),
        "dist_water_km": (dist_water * km_cell)[r, c].astype(np.float32),
        "natural_5x5": nat_win[r, c].astype(np.float32),
        "built_5x5": built_win[r, c].astype(np.float32),
        "crop_5x5": crop_win[r, c].astype(np.float32),
    })
    out["crop_edge"] = (
        (grid["frac_dogal"].to_numpy() >= 0.30) & (out["crop_5x5"] >= 0.10)
    ).astype(np.int8)

    out.to_parquet(OUT, index=False)

    # ---------------------------------------------------------------- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    forest_ids = set(grid.loc[grid["orman_maskesi"], "cell_id"])
    m = out[out["cell_id"].isin(forest_ids)]

    log("# Insan faktoru ve yakit surekliligi - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log("- Kaynak: mevcut 1 km arazi ortusu gridi (ESA WorldCover turevi)")
    log(f"- Pencere: {WIN} x {WIN} hucre (~5 km)")
    log(f"- Uretilen hucre: **{len(out):,}**, orman maskesinde: **{len(m):,}**")
    log()

    log("## Deger dagilimlari (orman maskesi)")
    log()
    log("| Degisken | Min | %25 | Ortanca | %75 | %95 | Maks |")
    log("|---|---|---|---|---|---|---|")
    for c_ in ("dist_built_km", "dist_crop_km", "dist_water_km",
               "natural_5x5", "built_5x5", "crop_5x5"):
        q = m[c_].quantile([0, .25, .5, .75, .95, 1])
        log(f"| {c_} | {q.iloc[0]:.2f} | {q.iloc[1]:.2f} | {q.iloc[2]:.2f} | "
            f"{q.iloc[3]:.2f} | {q.iloc[4]:.2f} | {q.iloc[5]:.2f} |")
    log()

    n_edge = int(m["crop_edge"].sum())
    log(f"- Orman-tarim siniri hucresi: **{n_edge:,}** ({n_edge / len(m):.1%})")
    log()
    log("> Bu hucreler ozellikle onemli: aniz yakma kaynakli sicramalarin")
    log("> gerceklestigi bolge burasi. Model egitiminden sonra `crop_edge`")
    log("> degiskeninin onem siralamasina bakiniz; yuksek cikarsa PDR'de")
    log("> 'tarim-orman gecis kusaginda onleyici konumlanma' onerisi yapabilirsiniz.")
    log()

    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")
    log(f"- Sutunlar: {', '.join(out.columns)}")
    log()
    log("> Not: bu katman GHSL nufus veya OSM yol agi yerine gecmez, onlarin")
    log("> yaklasik karsiligidir. v2'de gercek nufus yogunlugu ve yol mesafesi")
    log("> eklenirse dogruluk artar; PDR'de bu bir iyilestirme kalemi olarak")
    log("> yazilabilir.")

    (REPORT / "human_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'human_qa.md'}")


if __name__ == "__main__":
    main()
