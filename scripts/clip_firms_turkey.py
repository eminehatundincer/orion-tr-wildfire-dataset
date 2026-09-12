"""
ORION-TR / Adim 4a
FIRMS olay tablosunu resmi Turkiye idari sinirina kirpar.

Neden: indirme sinir kutusu (25.5, 35.5, 45.0, 42.5) Kuzey Suriye, Kuzey Irak,
Bati Iran, Kuzey Yunanistan ve Bulgaristan'in bir kismini kapsar. Bu bolgelerdeki
yogun tarimsal yakma ve petrol sahasi tespitleri etiketi bozar.

Ham veriye dokunulmaz; kirpma yalniz turetilmis katmanda yapilir.

Cikti:
  data_raw/boundary/tur_adm0.geojson          indirilen resmi sinir
  data_derived/firms_events_turkey.parquet    Turkiye ici olaylar
  data_derived/firms_events_outside.parquet   disarida kalanlar (izlenebilirlik)
  _reports/firms_clip_qa.md

Kullanim:
    .venv\\Scripts\\python.exe scripts\\clip_firms_turkey.py
    .venv\\Scripts\\python.exe scripts\\clip_firms_turkey.py --buffer-km 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
BOUNDARY_DIR = ROOT / "data_raw" / "boundary"
REPORT_DIR = ROOT / "_reports"

EVENTS = DERIVED / "firms_events.parquet"
BOUNDARY = BOUNDARY_DIR / "tur_adm0.geojson"

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/TUR/ADM0/"


def fetch_boundary() -> Path:
    """Turkiye ADM0 poligonunu geoBoundaries'ten indirir (CC BY 4.0)."""
    if BOUNDARY.exists() and BOUNDARY.stat().st_size > 0:
        print(f"Sinir dosyasi mevcut: {BOUNDARY}")
        return BOUNDARY

    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    print("Turkiye idari siniri indiriliyor (geoBoundaries gbOpen ADM0)...")

    meta = requests.get(GEOBOUNDARIES_API, timeout=60)
    meta.raise_for_status()
    info = meta.json()
    if isinstance(info, list):
        info = info[0]

    url = info.get("gjDownloadURL") or info.get("simplifiedGeometryGeoJSON")
    if not url:
        raise SystemExit("HATA: geoBoundaries yanitinda indirme baglantisi yok.")

    r = requests.get(url, timeout=180)
    r.raise_for_status()

    part = BOUNDARY.with_suffix(".geojson.part")
    part.write_bytes(r.content)
    gpd.read_file(part)          # dogrulama: acilabiliyor mu
    part.replace(BOUNDARY)

    (BOUNDARY_DIR / "source.json").write_text(json.dumps({
        "source": "geoBoundaries",
        "product": "gbOpen TUR ADM0",
        "license": info.get("licenseType", "CC BY 4.0"),
        "attribution": info.get("licenseDetail", ""),
        "source_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "crs": "EPSG:4326",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Indirildi: {BOUNDARY}")
    return BOUNDARY


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-km", type=float, default=0.0,
                   help="Sinir etkisi icin tampon (km). Varsayilan 0.")
    args = p.parse_args()

    if not EVENTS.exists():
        raise SystemExit(f"HATA: {EVENTS} yok. Once build_firms_events.py calistirin.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_boundary()

    tur = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    geom = tur.union_all() if hasattr(tur, "union_all") else tur.unary_union

    if args.buffer_km > 0:
        # metrik tampon icin once projeksiyon (Turkiye icin EPSG:5636 / LAEA yaklasimi)
        proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3035")
        geom = proj.buffer(args.buffer_km * 1000).to_crs("EPSG:4326").iloc[0]
        print(f"Tampon uygulandi: {args.buffer_km} km")

    df = pd.read_parquet(EVENTS)
    n_before = len(df)
    print(f"Girdi kayit: {n_before:,}")

    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    print("Nokta-poligon kesisimi hesaplaniyor...")
    inside_mask = pts.geometry.within(geom)

    inside = df[inside_mask.values].copy()
    outside = df[~inside_mask.values].copy()

    in_path = DERIVED / "firms_events_turkey.parquet"
    out_path = DERIVED / "firms_events_outside.parquet"
    inside.to_parquet(in_path, index=False)
    outside.to_parquet(out_path, index=False)

    # ------------------------------------------------------------- rapor
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# FIRMS Turkiye kirpma - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- Sinir kaynagi: geoBoundaries gbOpen TUR ADM0")
    log(f"- Tampon: {args.buffer_km} km")
    log()
    log(f"- Girdi (bbox, type=0): **{n_before:,}**")
    log(f"- Turkiye ici: **{len(inside):,}** ({len(inside) / n_before:.1%})")
    log(f"- Turkiye disi: **{len(outside):,}** ({len(outside) / n_before:.1%})")
    log()

    for name, frame in (("Turkiye ici", inside), ("Turkiye disi", outside)):
        if frame.empty:
            continue
        f = frame.copy()
        f["yil"] = pd.to_datetime(f["acq_date"]).dt.year
        f["ay"] = pd.to_datetime(f["acq_date"]).dt.month

        log(f"## {name} - yil bazinda")
        log()
        piv = pd.crosstab(f["yil"], f["source_sensor"])
        log("| Yil | " + " | ".join(piv.columns) + " | Toplam |")
        log("|---" * (len(piv.columns) + 2) + "|")
        for yil, row in piv.iterrows():
            log(f"| {yil} | " + " | ".join(f"{v:,}" for v in row.values) + f" | {row.sum():,} |")
        log()

        ay = f["ay"].value_counts().sort_index()
        log(f"## {name} - ay bazinda")
        log()
        log("| Ay | " + " | ".join(str(i) for i in range(1, 13)) + " |")
        log("|---" * 13 + "|")
        log("| Kayit | " + " | ".join(f"{ay.get(i, 0):,}" for i in range(1, 13)) + " |")
        log()

    log("## Cikti")
    log()
    log(f"- `{in_path.name}` ({in_path.stat().st_size / 1e6:.1f} MB)")
    log(f"- `{out_path.name}` ({out_path.stat().st_size / 1e6:.1f} MB)")
    log()
    log("> Ham bbox dosyalari degistirilmedi. Kirpma yalniz turetilmis katmandadir,")
    log("> bu nedenle tampon veya sinir kaynagi degisirse yeniden uretilebilir.")

    (REPORT_DIR / "firms_clip_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {REPORT_DIR / 'firms_clip_qa.md'}")


if __name__ == "__main__":
    main()
