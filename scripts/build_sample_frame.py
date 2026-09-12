"""
ORION-TR / Adim 7
Ornekem cercevesini (sample frame) kurar: pozitif ve negatif hucre-gunleri.

Satir tanimi : (cell_id, date t)
Etiket       : t+1..t+H araliginda o hucrede tutusma oldu mu  -> fire_next_72h

Tasarim kararlari:

1) Negatif ornekleme
   Orman maskesinde 259k hucre x 3653 gun = ~950M aday var. Hepsi kullanilamaz;
   rastgele ornekleme yapilir. Oran varsayilan 1:15.

2) Belirsiz bolge dislama
   Herhangi bir tespitin (NOAA-20 dahil) 3x3 komsulugunda ve +/- 7 gun icindeki
   hucre-gunler negatif olarak SECILMEZ. Orasi "yangin yok" degil "emin degiliz"
   bolgesidir; negatif sayilirsa model bulanik sinyal ogrenir.

3) Zamansal bolme
   train 2016-2022 / val 2023 / test 2024-2025.
   Rastgele bolme yapilmaz: ayni yanginin komsu gunleri hem egitimde hem testte
   olursa model kopya ceker ve basari sahte cikar.

Cikti:
  data_derived/sample_frame.parquet
  _reports/sample_frame_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_sample_frame.py
    .venv\\Scripts\\python.exe scripts\\build_sample_frame.py --neg-ratio 10
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT_DIR = ROOT / "_reports"

CELLDAYS = DERIVED / "fire_celldays.parquet"
GRID = DERIVED / "grid_1km_landcover.parquet"
OUT = DERIVED / "sample_frame.parquet"

NCOLS = 1950
START, END = date(2016, 1, 1), date(2025, 12, 31)
N_DAYS = (END - START).days + 1

SPLITS = {
    "train": range(2016, 2023),
    "val": [2023],
    "test": [2024, 2025],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3, help="Tahmin ufku (gun)")
    ap.add_argument("--neg-ratio", type=int, default=15)
    ap.add_argument("--buffer-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in (CELLDAYS, GRID):
        if not p.exists():
            raise SystemExit(f"HATA: {p} yok.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    cd = pd.read_parquet(CELLDAYS)
    grid = pd.read_parquet(GRID)
    forest = grid[grid["orman_maskesi"]].copy()
    forest_ids = forest["cell_id"].to_numpy(dtype=np.int64)

    log("# Ornekem cercevesi - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Tahmin ufku: t+1 .. t+{args.horizon} gun")
    log(f"- Negatif orani: 1:{args.neg_ratio}")
    log(f"- Dislama tamponu: 3x3 hucre, +/- {args.buffer_days} gun")
    log(f"- Orman maskesi hucre sayisi: **{len(forest_ids):,}**")
    log(f"- Gun sayisi: **{N_DAYS}**")
    log(f"- Aday uzay: **{len(forest_ids) * N_DAYS / 1e6:.0f} milyon** hucre-gun")
    log()

    cd["date"] = pd.to_datetime(cd["date"])
    cd["day_idx"] = (cd["date"] - pd.Timestamp(START)).dt.days.astype(np.int32)

    # ---------------------------------------------------------- pozitifler
    ig = cd[cd["is_ignition"]]
    pos_pairs: set[tuple[int, int]] = set()
    for cid, d in zip(ig["cell_id"].to_numpy(), ig["day_idx"].to_numpy()):
        for h in range(1, args.horizon + 1):
            t = int(d) - h
            if 0 <= t < N_DAYS:
                pos_pairs.add((int(cid), t))

    log("## 1. Pozitifler")
    log()
    log(f"- Tutusma sayisi: **{len(ig):,}**")
    log(f"- Uretilen pozitif satir: **{len(pos_pairs):,}**")
    log(f"- Ortalama satir/tutusma: {len(pos_pairs) / len(ig):.2f} "
        f"(ust sinir {args.horizon}; ortusme nedeniyle daha dusuk)")
    log()

    # ------------------------------------------------- dislama (yasak) kumesi
    print("Dislama kumesi kuruluyor...")
    forbidden: set[tuple[int, int]] = set(pos_pairs)
    for cid, d in zip(cd["cell_id"].to_numpy(), cd["day_idx"].to_numpy()):
        r, c = int(cid) // NCOLS, int(cid) % NCOLS
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nid = (r + dr) * NCOLS + (c + dc)
                for dd in range(-args.buffer_days, args.buffer_days + 1):
                    t = int(d) + dd
                    if 0 <= t < N_DAYS:
                        forbidden.add((nid, t))

    log("## 2. Dislama")
    log()
    log(f"- Yasak hucre-gun: **{len(forbidden):,}**")
    log(f"- Aday uzayin: {len(forbidden) / (len(forest_ids) * N_DAYS):.3%}")
    log()

    # ------------------------------------------------------------ negatifler
    target = len(pos_pairs) * args.neg_ratio
    print(f"Negatif orneklemesi ({target:,} hedef)...")
    neg: set[tuple[int, int]] = set()
    tries = 0
    batch = max(target // 4, 100_000)

    while len(neg) < target and tries < 60:
        cids = rng.choice(forest_ids, size=batch)
        days = rng.integers(0, N_DAYS, size=batch)
        for cid, t in zip(cids.tolist(), days.tolist()):
            pair = (int(cid), int(t))
            if pair not in forbidden and pair not in neg:
                neg.add(pair)
                if len(neg) >= target:
                    break
        tries += 1

    log("## 3. Negatifler")
    log()
    log(f"- Hedef: {target:,}")
    log(f"- Uretilen: **{len(neg):,}**")
    log(f"- Ornekleme turu: {tries}")
    log()

    # ------------------------------------------------------------ birlestir
    rows_c = np.fromiter((p[0] for p in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    rows_t = np.fromiter((p[1] for p in pos_pairs), dtype=np.int32, count=len(pos_pairs))
    neg_c = np.fromiter((p[0] for p in neg), dtype=np.int32, count=len(neg))
    neg_t = np.fromiter((p[1] for p in neg), dtype=np.int32, count=len(neg))

    df = pd.DataFrame({
        "cell_id": np.concatenate([rows_c, neg_c]),
        "day_idx": np.concatenate([rows_t, neg_t]),
        "label": np.concatenate([
            np.ones(len(rows_c), dtype=np.int8),
            np.zeros(len(neg_c), dtype=np.int8),
        ]),
    })
    df["date"] = pd.Timestamp(START) + pd.to_timedelta(df["day_idx"], unit="D")
    df["yil"] = df["date"].dt.year.astype(np.int16)
    df["ay"] = df["date"].dt.month.astype(np.int8)

    split = pd.Series("train", index=df.index, dtype="object")
    for name, years in SPLITS.items():
        split[df["yil"].isin(list(years))] = name
    df["split"] = split.astype("category")

    # hucre ozniteliklerini ekle
    keep_cols = ["cell_id", "lat_center", "lon_center", "frac_agac", "frac_maki",
                 "frac_otlak", "frac_tarim", "frac_yapili", "frac_ciplak",
                 "frac_dogal"]
    df = df.merge(grid[keep_cols], on="cell_id", how="left")
    df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    # ------------------------------------------------------------- rapor
    log("## 4. Ornekem cercevesi")
    log()
    log(f"- Toplam satir: **{len(df):,}**")
    log(f"- Pozitif: **{int((df['label'] == 1).sum()):,}** "
        f"({(df['label'] == 1).mean():.2%})")
    log()

    log("### Bolme dagilimi")
    log()
    log("| Bolme | Yillar | Satir | Pozitif | Pozitif orani |")
    log("|---|---|---|---|---|")
    for name in ("train", "val", "test"):
        s = df[df["split"] == name]
        if s.empty:
            continue
        yrs = f"{s['yil'].min()}-{s['yil'].max()}"
        log(f"| {name} | {yrs} | {len(s):,} | {int(s['label'].sum()):,} | "
            f"{s['label'].mean():.2%} |")
    log()

    log("### Ay bazinda pozitif orani")
    log()
    log("| Ay | " + " | ".join(str(i) for i in range(1, 13)) + " |")
    log("|---" * 13 + "|")
    rate = df.groupby("ay")["label"].mean()
    log("| Pozitif % | " + " | ".join(f"{rate.get(i, 0):.1%}" for i in range(1, 13)) + " |")
    log()
    log("> Negatifler yil boyunca duzgun dagitildigi icin yaz aylarinda pozitif")
    log("> orani yukselir. Model mevsimselligi ogrenecektir; bu istenen bir sey,")
    log("> ancak tek basina mevsimden daha fazlasini ogrendigini dogrulamak icin")
    log("> egitimde ay degiskenini cikarip performans karsilastirmasi yapiniz.")
    log()

    log("## 5. Sutunlar")
    log()
    log("| Sutun | Aciklama |")
    log("|---|---|")
    log("| cell_id, day_idx, date | anahtar (hucre + gun) |")
    log("| label | t+1..t+" + str(args.horizon) + " tutusma (0/1) |")
    log("| split | train / val / test (zamansal) |")
    log("| lat_center, lon_center | hucre merkezi |")
    log("| frac_* | arazi ortusu oranlari |")
    log()
    log("> Sirada meteoroloji katmanlari var: FWI (CEMS), ERA5, NDVI, egim,")
    log("> nufus ve yol mesafesi. Hepsi (cell_id, date) anahtarina baglanacak.")
    log()
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")

    (REPORT_DIR / "sample_frame_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT_DIR / 'sample_frame_qa.md'}")


if __name__ == "__main__":
    main()
