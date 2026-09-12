"""
ORION-TR / Adim 6
FIRMS tespitlerini 1 km grid hucrelerine baglar, orman maskesiyle filtreler ve
"tutusma" ile "devam eden yangin" ayrimini yapar.

Iki kritik karar:

1) Filtre artik nokta sinifina degil HUCRE maskesine gore.
   10 m'de tek nokta okumak kirilgandi; modelin birimi hucre oldugu icin
   filtre de hucre duzeyinde olmali.

2) Tutusma / devam ayrimi.
   Buyuk bir yangin gunlerce tespit uretir. Model "dun yangin vardi -> bugun de
   var" iliskisini ogrenirse ise yaramaz. Bir hucre-gunu, onceki N gunde o
   hucrede veya 3x3 komsulugunda tespit YOKSA tutusma sayilir.

Sensor homojenligi:
   Etiket yalniz VIIRS_SNPP_SP + MODIS_SP ile uretilir (ikisi de 2016-2025
   boyunca kesintisiz). VIIRS_NOAA20_SP 2018-04'te devreye girdigi icin
   kalibrasyon kirilimi olusturur; ayri sutunda QA amaciyla tutulur.

Cikti:
  data_derived/fire_celldays.parquet
  _reports/fire_celldays_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_fire_celldays.py
    .venv\\Scripts\\python.exe scripts\\build_fire_celldays.py --quiet-days 5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT_DIR = ROOT / "_reports"

EVENTS = DERIVED / "firms_events_turkey_lc.parquet"
GRID = DERIVED / "grid_1km_landcover.parquet"
OUT = DERIVED / "fire_celldays.parquet"

GRID_WEST, GRID_NORTH = 25.50, 42.50
CELL_DEG = 0.01
NCOLS = 1950
NROWS = 700

LABEL_SENSORS = ("VIIRS_SNPP_SP", "MODIS_SP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet-days", type=int, default=3,
                    help="Tutusma sayilmasi icin gereken sessiz gun sayisi")
    ap.add_argument("--forest-threshold", type=float, default=0.30)
    args = ap.parse_args()

    for p in (EVENTS, GRID):
        if not p.exists():
            raise SystemExit(f"HATA: {p} yok.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    ev = pd.read_parquet(EVENTS)
    grid = pd.read_parquet(GRID)
    log("# Yangin hucre-gunu tablosu - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Girdi tespit: **{len(ev):,}**")
    log(f"- Sessiz gun esigi: **{args.quiet_days}**")
    log(f"- Orman esigi: dogal ortu >= %{args.forest_threshold * 100:.0f}")
    log()

    # ---- 1. hucre atamasi
    ev["acq_date"] = pd.to_datetime(ev["acq_date"])
    cr = np.floor((GRID_NORTH - ev["latitude"].to_numpy()) / CELL_DEG).astype(np.int32)
    cc = np.floor((ev["longitude"].to_numpy() - GRID_WEST) / CELL_DEG).astype(np.int32)
    ok = (cr >= 0) & (cr < NROWS) & (cc >= 0) & (cc < NCOLS)
    ev = ev[ok].copy()
    ev["cell_id"] = (cr[ok].astype(np.int64) * NCOLS + cc[ok]).astype(np.int32)

    g = grid.set_index("cell_id")
    ev = ev.join(
        g[["frac_dogal", "frac_agac", "frac_otlak", "frac_tarim", "in_turkey"]],
        on="cell_id",
    )

    # ---- 2. iki filtre tanimini karsilastir
    by_point = ev["worldcover_class"].isin([10, 20])
    by_cell = ev["in_turkey"].fillna(False) & (ev["frac_dogal"] >= args.forest_threshold)

    log("## 1. Filtre tanimi karsilastirmasi")
    log()
    log("| Tanim | Tespit | Oran |")
    log("|---|---|---|")
    log(f"| Nokta sinifi 10/20 (eski) | {int(by_point.sum()):,} | {by_point.mean():.1%} |")
    log(f"| Hucre maskesi >= %{args.forest_threshold * 100:.0f} (yeni) | "
        f"{int(by_cell.sum()):,} | {by_cell.mean():.1%} |")
    log(f"| Her ikisi | {int((by_point & by_cell).sum()):,} | |")
    log(f"| Sadece hucre (nokta kacirmis) | {int((~by_point & by_cell).sum()):,} | |")
    log()

    fire = ev[by_cell].copy()
    log(f"Secilen tanim: **hucre maskesi**. Kalan tespit: **{len(fire):,}**")
    log()

    # ---- 3. sensor ayrimi
    is_label = fire["source_sensor"].isin(LABEL_SENSORS)
    log("## 2. Sensor dagilimi")
    log()
    for s, n in fire["source_sensor"].value_counts().items():
        mark = " *(etikette kullanilir)*" if s in LABEL_SENSORS else " *(yalniz QA)*"
        log(f"- `{s}`: {n:,}{mark}")
    log()

    # ---- 4. hucre-gunu toplama
    fire["date"] = fire["acq_date"].dt.normalize()
    lab = fire[is_label]

    agg = lab.groupby(["cell_id", "date"]).agg(
        n_det=("frp", "size"),
        max_frp=("frp", "max"),
        mean_frp=("frp", "mean"),
        n_day=("daynight", lambda s: int((s == "D").sum())),
    ).reset_index()

    aux = fire[~is_label].groupby(["cell_id", "date"]).size().rename("n_det_noaa20")
    agg = agg.join(aux, on=["cell_id", "date"])
    agg["n_det_noaa20"] = agg["n_det_noaa20"].fillna(0).astype(np.int32)

    log(f"## 3. Hucre-gunu")
    log()
    log(f"- Yangin hucre-gunu (SNPP+MODIS): **{len(agg):,}**")
    log(f"- Benzersiz hucre: **{agg['cell_id'].nunique():,}**")
    log(f"- Benzersiz gun: **{agg['date'].nunique():,}**")
    log()

    # ---- 5. tutusma / devam ayrimi
    print("Tutusma/devam ayrimi hesaplaniyor...")
    day0 = agg["date"].min()
    agg["day_idx"] = (agg["date"] - day0).dt.days.astype(np.int32)
    agg["c_row"] = (agg["cell_id"] // NCOLS).astype(np.int32)
    agg["c_col"] = (agg["cell_id"] % NCOLS).astype(np.int32)

    # 3x3 komsuluk x onceki quiet_days gun icinde tespit var mi?
    active = set(zip(agg["c_row"].tolist(), agg["c_col"].tolist(),
                     agg["day_idx"].tolist()))

    ignition = np.ones(len(agg), dtype=bool)
    rows = agg["c_row"].to_numpy()
    cols = agg["c_col"].to_numpy()
    days = agg["day_idx"].to_numpy()

    for i in range(len(agg)):
        r, c, d = rows[i], cols[i], days[i]
        found = False
        for dd in range(1, args.quiet_days + 1):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc, d - dd) in active:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        ignition[i] = not found

    agg["is_ignition"] = ignition
    agg = agg.drop(columns=["day_idx", "c_row", "c_col"])

    n_ig = int(ignition.sum())
    log("## 4. Tutusma / devam")
    log()
    log(f"- Tutusma: **{n_ig:,}** ({n_ig / len(agg):.1%})")
    log(f"- Devam eden: **{len(agg) - n_ig:,}** ({1 - n_ig / len(agg):.1%})")
    log()
    log("> Model yalniz tutusmalari pozitif olarak gorecek. Devam eden yangin")
    log("> gunleri egitimden cikarilir; aksi halde model 'dun yandi -> bugun de")
    log("> yanar' kisayolunu ogrenir ve onleyici degeri kalmaz.")
    log()

    # ---- 6. dagilimlar
    ig = agg[agg["is_ignition"]]
    log("## 5. Tutusma dagilimi")
    log()
    yil = ig["date"].dt.year.value_counts().sort_index()
    log("| Yil | " + " | ".join(str(y) for y in yil.index) + " |")
    log("|---" * (len(yil) + 1) + "|")
    log("| Tutusma | " + " | ".join(f"{v:,}" for v in yil.values) + " |")
    log()
    ay = ig["date"].dt.month.value_counts().sort_index()
    log("| Ay | " + " | ".join(str(i) for i in range(1, 13)) + " |")
    log("|---" * 13 + "|")
    log("| Tutusma | " + " | ".join(f"{ay.get(i, 0):,}" for i in range(1, 13)) + " |")
    log()

    log("### En cok tutusma goren 10 hucre")
    log()
    top = ig["cell_id"].value_counts().head(10)
    gi = grid.set_index("cell_id")
    log("| Hucre | Enlem | Boylam | Tutusma | Agac orani |")
    log("|---|---|---|---|---|")
    for cid, n in top.items():
        r = gi.loc[cid]
        log(f"| {cid} | {r['lat_center']:.3f} | {r['lon_center']:.3f} | {n} | "
            f"{r['frac_agac']:.0%} |")
    log()

    agg.to_parquet(OUT, index=False)
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")
    log(f"- Sutunlar: cell_id, date, n_det, max_frp, mean_frp, n_day, "
        f"n_det_noaa20, is_ignition")

    (REPORT_DIR / "fire_celldays_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT_DIR / 'fire_celldays_qa.md'}")


if __name__ == "__main__":
    main()
