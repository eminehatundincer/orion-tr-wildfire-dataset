"""
ORION-TR / Adim 12
CEMS FWI indekslerini ornekem cercevesine baglar.

Iki kritik ayrinti:

1) Gecikmeli degiskenler
   Yakit kurulugu birikimli bir surectir. "Bugun FWI kac" kadar "son haftanin
   ortalamasi kac" da belirleyicidir. FWI icin 7 gunluk ortalama, 7 gunluk
   maksimum ve 30 gunluk ortalama hesaplanir.

2) Kiyi duzeltmesi
   FWI 0.25 derece gridde ve deniz uzerinde tanimsizdir (~%12 NaN). Mugla ve
   Antalya kiyisindaki orman hucrelerinin bir kismi en yakin grid noktasi
   olarak denize duser. Bunlar icin genisleyen komsulukta en yakin gecerli
   kara noktasi aranir; aksi halde Turkiye'nin en riskli bolgesi kaybolur.

Cikti:
  data_derived/features_fwi.parquet   (cell_id, day_idx anahtarli)
  _reports/features_fwi_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\join_fwi.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "fwi"
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
FRAME = DERIVED / "sample_frame.parquet"
OUT = DERIVED / "features_fwi.parquet"

START = pd.Timestamp("2016-01-01")
N_DAYS = 3653
YEARS = range(2016, 2026)

VARIABLES = {
    "fire_weather_index": "fwi",
    "fine_fuel_moisture_code": "ffmc",
    "duff_moisture_code": "dmc",
    "drought_code": "dc",
    "initial_fire_spread_index": "isi",
    "build_up_index": "bui",
}

MAX_SEARCH = 6          # kiyi duzeltmesinde en fazla kac hucre uzaga bakilir


def load_series(variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bir degiskenin 10 yilini tek dizide birlestirir -> (time, lat, lon)."""
    vdir = RAW / variable
    files = sorted(vdir.glob("*.nc"))
    if not files:
        raise SystemExit(f"HATA: {vdir} bos.")

    lat = lon = None
    big = None

    for fp in files:
        ds = xr.open_dataset(fp)
        name = list(ds.data_vars)[0]
        da = ds[name]

        la = ds["latitude"].values.astype("float64")
        lo = ds["longitude"].values.astype("float64")
        if lat is None:
            lat, lon = la, lo
            big = np.full((N_DAYS, la.size, lo.size), np.nan, dtype=np.float32)
        elif not (np.array_equal(la, lat) and np.array_equal(lo, lon)):
            raise SystemExit(f"HATA: {fp.name} grid farkli.")

        t = pd.to_datetime(ds["valid_time"].values)
        idx = (t - START).days.to_numpy()
        vals = np.squeeze(da.values).astype(np.float32)
        if vals.ndim == 2:
            vals = vals[None, ...]

        ok = (idx >= 0) & (idx < N_DAYS)
        big[idx[ok]] = vals[ok]
        ds.close()

    return big, lat, lon


def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Gecmise bakan w gunluk ortalama (bugun dahil)."""
    filled = np.nan_to_num(a, nan=0.0)
    mask = (~np.isnan(a)).astype(np.float32)
    cs = np.cumsum(filled, axis=0)
    cm = np.cumsum(mask, axis=0)
    out = np.empty_like(a)
    out[:w] = cs[:w] / np.maximum(cm[:w], 1)
    out[w:] = (cs[w:] - cs[:-w]) / np.maximum(cm[w:] - cm[:-w], 1)
    out[cm == 0] = np.nan
    return out


def rolling_max(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(a.shape[0]):
        lo = max(0, i - w + 1)
        out[i] = np.nanmax(a[lo:i + 1], axis=0)
    return out


def map_cells(lat_c: np.ndarray, lon_c: np.ndarray,
              lat: np.ndarray, lon: np.ndarray,
              valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hucre merkezlerini grid indekslerine esler, denize duseni kaydirir."""
    li = np.abs(lat[None, :] - lat_c[:, None]).argmin(axis=1)
    oi = np.abs(lon[None, :] - lon_c[:, None]).argmin(axis=1)

    shifted = np.zeros(len(lat_c), dtype=np.int8)
    bad = ~valid[li, oi]
    nlat, nlon = valid.shape

    for k in np.nonzero(bad)[0]:
        found = False
        for rad in range(1, MAX_SEARCH + 1):
            best, bestd = None, None
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    r, c = li[k] + dr, oi[k] + dc
                    if 0 <= r < nlat and 0 <= c < nlon and valid[r, c]:
                        d = (lat[r] - lat_c[k]) ** 2 + (lon[c] - lon_c[k]) ** 2
                        if bestd is None or d < bestd:
                            best, bestd = (r, c), d
            if best is not None:
                li[k], oi[k] = best
                shifted[k] = rad
                found = True
                break
        if not found:
            shifted[k] = -1
    return li, oi, shifted


def main() -> None:
    if not FRAME.exists():
        raise SystemExit(f"HATA: {FRAME} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(FRAME, columns=["cell_id", "day_idx",
                                            "lat_center", "lon_center"])
    print(f"Ornekem cercevesi: {len(frame):,} satir")

    cells = frame.drop_duplicates("cell_id")[
        ["cell_id", "lat_center", "lon_center"]].reset_index(drop=True)
    print(f"Benzersiz hucre: {len(cells):,}")

    # ---- ilk degiskeni yukleyip grid ve gecerlilik maskesini kur
    print(f"\n{'fire_weather_index':28s} yukleniyor...")
    arr, lat, lon = load_series("fire_weather_index")
    valid = np.isfinite(arr).mean(axis=0) > 0.5
    print(f"  grid {arr.shape}, gecerli kara noktasi: {valid.sum()}/{valid.size}")

    li, oi, shifted = map_cells(
        cells["lat_center"].to_numpy("float64"),
        cells["lon_center"].to_numpy("float64"),
        lat, lon, valid)

    n_shift = int((shifted > 0).sum())
    n_fail = int((shifted < 0).sum())
    print(f"  kiyi duzeltmesi: {n_shift:,} hucre kaydirildi, "
          f"{n_fail:,} hucre eslenemedi")

    cell_pos = pd.DataFrame({
        "cell_id": cells["cell_id"].to_numpy(),
        "_li": li, "_oi": oi, "_shift": shifted,
    })
    f = frame.merge(cell_pos, on="cell_id", how="left")
    t_idx = f["day_idx"].to_numpy()
    r_idx = f["_li"].to_numpy()
    c_idx = f["_oi"].to_numpy()

    out = pd.DataFrame({
        "cell_id": f["cell_id"].to_numpy(),
        "day_idx": t_idx,
        "fwi_grid_shift": f["_shift"].to_numpy(),
    })

    # ---- FWI ve gecikmeli turevleri
    out["fwi"] = arr[t_idx, r_idx, c_idx]
    print("  gecikmeli degiskenler...")
    out["fwi_mean7"] = rolling_mean(arr, 7)[t_idx, r_idx, c_idx]
    out["fwi_mean30"] = rolling_mean(arr, 30)[t_idx, r_idx, c_idx]
    out["fwi_max7"] = rolling_max(arr, 7)[t_idx, r_idx, c_idx]
    del arr

    # ---- kalan bes indeks
    for var, short in VARIABLES.items():
        if var == "fire_weather_index":
            continue
        print(f"{var:28s} yukleniyor...")
        a, la2, lo2 = load_series(var)
        if not (np.array_equal(la2, lat) and np.array_equal(lo2, lon)):
            raise SystemExit(f"HATA: {var} gridi farkli.")
        out[short] = a[t_idx, r_idx, c_idx]
        del a

    for c in out.columns:
        if c not in ("cell_id", "day_idx", "fwi_grid_shift"):
            out[c] = out[c].astype(np.float32)

    out.to_parquet(OUT, index=False)

    # ------------------------------------------------------------- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# FWI oznitelikleri - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Kaynak: CEMS cems-fire-historical-v1, system_version 4_1")
    log(f"- Grid: {lat.size} x {lon.size} @ 0.25 derece")
    log(f"- Satir: **{len(out):,}**")
    log()

    log("## Kiyi duzeltmesi")
    log()
    log(f"- Dogrudan eslenen hucre: **{int((shifted == 0).sum()):,}**")
    log(f"- Kaydirilan (denize dusen): **{n_shift:,}**")
    log(f"- Eslenemeyen: **{n_fail:,}**")
    log()
    log("> FWI deniz uzerinde tanimsizdir. Kaydirma yapilmasaydi Mugla ve")
    log("> Antalya kiyisindaki orman hucreleri bos kalacakti; bunlar Turkiye'nin")
    log("> en yangin riskli bolgesidir. `fwi_grid_shift` sutunu kac hucre")
    log("> kaydirildigini tutar, seffaflik icin veri setinde birakildi.")
    log()

    log("## Deger dagilimlari")
    log()
    log("| Degisken | Eksik | Min | Ortanca | %95 | Maks |")
    log("|---|---|---|---|---|---|")
    for c in ("fwi", "fwi_mean7", "fwi_mean30", "fwi_max7",
              "ffmc", "dmc", "dc", "isi", "bui"):
        s = out[c]
        q = s.quantile([0, .5, .95, 1])
        log(f"| {c} | {s.isna().mean():.2%} | {q.iloc[0]:.1f} | {q.iloc[1]:.1f} | "
            f"{q.iloc[2]:.1f} | {q.iloc[3]:.1f} |")
    log()

    # ---- ayirt edicilik testi
    lab = pd.read_parquet(FRAME, columns=["cell_id", "day_idx", "label"])
    j = out.merge(lab, on=["cell_id", "day_idx"], how="left")
    log("## Ayirt edicilik testi")
    log()
    log("| Degisken | Negatif ortalama | Pozitif ortalama | Fark |")
    log("|---|---|---|---|")
    for c in ("fwi", "fwi_mean7", "fwi_mean30", "fwi_max7",
              "ffmc", "dmc", "dc", "isi", "bui"):
        n = j.loc[j["label"] == 0, c].mean()
        p = j.loc[j["label"] == 1, c].mean()
        log(f"| {c} | {n:.1f} | {p:.1f} | **{p - n:+.1f}** |")
    log()
    log("> Pozitif satirlarda degerlerin belirgin yuksek olmasi beklenir.")
    log("> Fark yoksa ya esleme hatali ya da etiket sorunludur.")
    log()
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")

    (REPORT / "features_fwi_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'features_fwi_qa.md'}")


if __name__ == "__main__":
    main()
