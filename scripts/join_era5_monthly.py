"""
ORION-TR / Adim 14
ERA5-Land aylik toprak nemi ve bitki ortusu verisini ornekem cercevesine baglar.

Ham deger + anomali:
  Ham toprak nemi tek basina yanilticidir; Mugla'nin normali ile Erzurum'un
  normali farklidir. Asil soru "bu hucre KENDI normaline gore ne kadar kurak".
  Bu yuzden her hucre x ay kombinasyonu icin 10 yillik ortalamadan sapma
  hesaplanir. Kuraklik literaturunde standart yaklasim budur.

  Ayni mantik LAI icin de gecerli: bitki ortusu o yil normalinden zayifsa
  yakit daha kurudur.

Grid: ERA5-Land ~0.1 derece. Deniz uzerinde tanimsiz oldugu icin kiyi
hucreleri icin en yakin gecerli kara noktasi aranir.

Cikti:
  data_derived/features_era5_monthly.parquet
  _reports/features_era5_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\join_era5_monthly.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
SRC = ROOT / "data_raw" / "era5_monthly" / "era5_land_monthly_turkey.nc"
SPLIT_DIR = ROOT / "data_raw" / "era5_monthly"
FRAME = DERIVED / "sample_frame.parquet"
OUT = DERIVED / "features_era5_monthly.parquet"

START = pd.Timestamp("2016-01-01")
MAX_SEARCH = 8

SHORT = {
    "swvl1": "soil_w1", "swvl2": "soil_w2", "swvl3": "soil_w3", "swvl4": "soil_w4",
    "lai_hv": "lai_high", "lai_lv": "lai_low", "skt": "skin_temp",
}
ANOM_FOR = ["soil_w1", "soil_w2", "soil_w3", "soil_w4", "lai_high", "lai_low"]


def load() -> xr.Dataset:
    if SRC.exists():
        return xr.open_dataset(SRC)
    parts = sorted(SPLIT_DIR.glob("era5_land_monthly_*_turkey.nc"))
    if not parts:
        raise SystemExit(f"HATA: {SRC} yok ve parcali dosya da bulunamadi.")
    print(f"Parcali {len(parts)} dosya birlestiriliyor...")
    return xr.merge([xr.open_dataset(p) for p in parts])


def map_cells(lat_c, lon_c, lat, lon, valid):
    li = np.abs(lat[None, :] - lat_c[:, None]).argmin(axis=1)
    oi = np.abs(lon[None, :] - lon_c[:, None]).argmin(axis=1)
    shifted = np.zeros(len(lat_c), dtype=np.int8)
    nlat, nlon = valid.shape
    for k in np.nonzero(~valid[li, oi])[0]:
        placed = False
        for rad in range(1, MAX_SEARCH + 1):
            best, bd = None, None
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    r, c = li[k] + dr, oi[k] + dc
                    if 0 <= r < nlat and 0 <= c < nlon and valid[r, c]:
                        d = (lat[r] - lat_c[k]) ** 2 + (lon[c] - lon_c[k]) ** 2
                        if bd is None or d < bd:
                            best, bd = (r, c), d
            if best is not None:
                li[k], oi[k] = best
                shifted[k] = rad
                placed = True
                break
        if not placed:
            shifted[k] = -1
    return li, oi, shifted


def main() -> None:
    if not FRAME.exists():
        raise SystemExit(f"HATA: {FRAME} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    ds = load()
    print(f"Degiskenler: {list(ds.data_vars)}")
    print(f"Boyutlar   : {dict(ds.sizes)}")

    tname = "valid_time" if "valid_time" in ds.coords else "time"
    times = pd.to_datetime(ds[tname].values)
    lat = ds["latitude"].values.astype("float64")
    lon = ds["longitude"].values.astype("float64")
    print(f"Donem      : {times.min():%Y-%m} .. {times.max():%Y-%m} ({times.size} ay)")
    print(f"Grid       : {lat.size} x {lon.size}")

    frame = pd.read_parquet(FRAME, columns=["cell_id", "day_idx", "date",
                                            "lat_center", "lon_center"])
    cells = frame.drop_duplicates("cell_id")[
        ["cell_id", "lat_center", "lon_center"]].reset_index(drop=True)
    print(f"\nOrnekem: {len(frame):,} satir, {len(cells):,} benzersiz hucre")

    # ---- ay indeksi: her satirin tarihi hangi aya denk geliyor
    ym = pd.to_datetime(frame["date"]).dt.to_period("M")
    ym_src = pd.PeriodIndex(times, freq="M")
    lut = pd.Series(np.arange(len(ym_src)), index=ym_src)
    t_idx = ym.map(lut).to_numpy()
    if np.isnan(t_idx).any():
        raise SystemExit("HATA: bazi aylar kaynak veride yok.")
    t_idx = t_idx.astype(np.int32)

    # ---- gecerlilik maskesi ilk degiskenden
    first = list(ds.data_vars)[0]
    a0 = np.squeeze(ds[first].values)
    valid = np.isfinite(a0).mean(axis=0) > 0.5
    print(f"Gecerli kara noktasi: {valid.sum()}/{valid.size}")

    li, oi, shifted = map_cells(
        cells["lat_center"].to_numpy("float64"),
        cells["lon_center"].to_numpy("float64"), lat, lon, valid)
    n_shift, n_fail = int((shifted > 0).sum()), int((shifted < 0).sum())
    print(f"Kiyi duzeltmesi: {n_shift:,} kaydirildi, {n_fail:,} eslenemedi")

    pos = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(),
                        "_li": li, "_oi": oi, "_sh": shifted})
    f = frame.merge(pos, on="cell_id", how="left")
    r_idx, c_idx = f["_li"].to_numpy(), f["_oi"].to_numpy()

    out = pd.DataFrame({
        "cell_id": f["cell_id"].to_numpy(),
        "day_idx": f["day_idx"].to_numpy(),
        "era5_grid_shift": f["_sh"].to_numpy(),
    })

    month_of = ym_src.month.to_numpy()

    for vname in ds.data_vars:
        short = SHORT.get(str(vname), str(vname))
        arr = np.squeeze(ds[vname].values).astype(np.float32)
        if arr.ndim != 3:
            print(f"  {vname}: beklenmeyen boyut {arr.shape}, atlandi")
            continue
        print(f"  {short} ...", end=" ", flush=True)

        out[short] = arr[t_idx, r_idx, c_idx]

        if short in ANOM_FOR:
            # her ay icin 10 yillik iklim normali, sonra sapma
            clim = np.empty_like(arr)
            for m in range(1, 13):
                sel = month_of == m
                with np.errstate(invalid="ignore"):
                    mu = np.nanmean(arr[sel], axis=0)
                clim[sel] = mu
            out[f"{short}_anom"] = (arr - clim)[t_idx, r_idx, c_idx]
        print("ok")

    for c in out.columns:
        if c not in ("cell_id", "day_idx", "era5_grid_shift"):
            out[c] = out[c].astype(np.float32)

    out.to_parquet(OUT, index=False)
    ds.close()

    # ------------------------------------------------------------- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# ERA5-Land aylik oznitelikleri - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Kaynak: reanalysis-era5-land-monthly-means, ~0.1 derece")
    log(f"- Donem: {times.min():%Y-%m} .. {times.max():%Y-%m} ({times.size} ay)")
    log(f"- Satir: **{len(out):,}**")
    log(f"- Kiyi duzeltmesi: {n_shift:,} hucre kaydirildi, {n_fail:,} eslenemedi")
    log()

    feats = [c for c in out.columns
             if c not in ("cell_id", "day_idx", "era5_grid_shift")]
    log("## Deger dagilimlari")
    log()
    log("| Degisken | Eksik | Min | Ortanca | Maks |")
    log("|---|---|---|---|---|")
    for c in feats:
        s = out[c]
        q = s.quantile([0, .5, 1])
        log(f"| {c} | {s.isna().mean():.2%} | {q.iloc[0]:.3f} | "
            f"{q.iloc[1]:.3f} | {q.iloc[2]:.3f} |")
    log()

    lab = pd.read_parquet(FRAME, columns=["cell_id", "day_idx", "label"])
    j = out.merge(lab, on=["cell_id", "day_idx"], how="left")
    log("## Ayirt edicilik testi")
    log()
    log("| Degisken | Negatif | Pozitif | Fark |")
    log("|---|---|---|---|")
    for c in feats:
        n = j.loc[j["label"] == 0, c].mean()
        p = j.loc[j["label"] == 1, c].mean()
        log(f"| {c} | {n:.3f} | {p:.3f} | **{p - n:+.3f}** |")
    log()
    log("> Toprak nemi ve LAI icin pozitif satirlarda DUSUK deger beklenir")
    log("> (kurak zemin, zayif ortu). Anomali sutunlarinda negatif fark, yani")
    log("> hucrenin kendi normalinin altinda olmasi en anlamli gostergedir.")
    log()
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")

    (REPORT / "features_era5_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'features_era5_qa.md'}")


if __name__ == "__main__":
    main()
