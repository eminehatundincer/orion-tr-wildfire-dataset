"""
ORION-TR / Adim 8 - test dosyasi denetimi

Test indirmelerinin gercekten veri icerip icermedigini sinar.

Devir rehberi 8.4'teki uyari gecerli: urun '.zip' uzantisiyla gelse de
dogrudan NetCDF/HDF5 container olabilir. Bu yuzden uzantiya degil MAGIC BYTE'a
bakiyoruz ve iki bicimi de aciyoruz.

Kontroller:
  - gercek container bicimi (ZIP / HDF5 / NetCDF3 / GRIB)
  - boyutlar, koordinatlar, degisken adlari
  - Turkiye bbox kapsamasi ve grid araligi
  - deger araligi ve NaN orani (fiziksel olarak anlamli mi)

Kullanim:
    .venv\\Scripts\\python.exe scripts\\inspect_test_files.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data_derived" / "_test"

MAGIC = {
    b"PK\x03\x04": "ZIP",
    b"\x89HDF": "HDF5 / NetCDF4",
    b"CDF\x01": "NetCDF3 classic",
    b"CDF\x02": "NetCDF3 64-bit",
    b"GRIB": "GRIB",
}


def detect(path: Path) -> str:
    head = path.open("rb").read(8)
    for sig, name in MAGIC.items():
        if head.startswith(sig):
            return name
    return f"bilinmiyor ({head!r})"


def open_any(path: Path) -> tuple[xr.Dataset, str]:
    kind = detect(path)
    if kind == "ZIP":
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            print(f"    ZIP uyeleri: {members}")
            inner = [m for m in members if m.lower().endswith((".nc", ".nc4"))]
            if not inner:
                raise RuntimeError("ZIP icinde NetCDF yok.")
            out = path.parent / inner[0]
            with z.open(inner[0]) as src, out.open("wb") as dst:
                dst.write(src.read())
            return xr.open_dataset(out), f"ZIP -> {inner[0]}"
    return xr.open_dataset(path), kind


def report(path: Path, title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    if not path.exists():
        print(f"YOK: {path}")
        return
    print(f"Dosya  : {path.name}")
    print(f"Boyut  : {path.stat().st_size:,} bayt")
    print(f"Bicim  : {detect(path)}")

    try:
        ds, how = open_any(path)
    except Exception as e:
        print(f"ACILAMADI: {type(e).__name__}: {e}")
        return

    print(f"Acilis : {how}")
    print(f"\nBoyutlar   : {dict(ds.sizes)}")
    print(f"Koordinat  : {list(ds.coords)}")
    print(f"Degiskenler: {list(ds.data_vars)}")

    for cname in ("latitude", "lat", "longitude", "lon", "y", "x"):
        if cname in ds.coords:
            v = ds[cname].values
            if v.size > 1:
                step = float(abs(v[1] - v[0]))
                print(f"  {cname:10s}: {float(v.min()):.3f} .. {float(v.max()):.3f} "
                      f"| n={v.size} | adim={step:.4f}")
            else:
                print(f"  {cname:10s}: {v}")

    for tname in ("valid_time", "time", "forecast_reference_time"):
        if tname in ds.coords:
            print(f"  {tname:10s}: {ds[tname].values.ravel()[:5]}")
            break

    print("\nDeger kontrolu:")
    for name, da in ds.data_vars.items():
        try:
            arr = np.asarray(da.values, dtype="float64")
        except Exception:
            print(f"  {name}: sayisal degil, atlandi")
            continue
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            print(f"  {name}: TAMAMEN BOS (hic gecerli deger yok)  <-- SORUN")
            continue
        vals = arr[finite]
        print(f"  {name}: min={vals.min():.3f}  ort={vals.mean():.3f}  "
              f"maks={vals.max():.3f}  gecerli={finite.mean():.1%}  "
              f"birim={da.attrs.get('units', '?')}")

    ds.close()


def main() -> None:
    report(TMP / "test_era5.zip",
           "CDS / ERA5 - 2m sicaklik, gunluk maksimum, 2021-07-28")
    report(TMP / "test_fwi.zip",
           "EWDS / FWI - yangin hava indeksi, 2021-07-28")

    print(f"\n{'=' * 60}\nBEKLENEN DEGERLER\n{'=' * 60}")
    print("ERA5 2m_temperature : Kelvin, ~300-320 K (28 Temmuz, Turkiye yazi)")
    print("FWI                 : birimsiz, 0-100+; yangin gununde 30-90 arasi")
    print("Grid                : ERA5 ~0.25 derece, FWI 0.25 derece")
    print("Enlem/boylam        : 35.5-42.5 N, 25.5-45.0 E icinde olmali")


if __name__ == "__main__":
    main()
