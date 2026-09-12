"""
ORION-TR / Adim 15
Nihai veri setini kurar: butun katmanlari tek tabloda birlestirir, zaman
kodlamalarini ekler, bilgisiz sutunlari atar, veri karti ve sutun sozlugu yazar.

Birlestirilen katmanlar:
  sample_frame.parquet            anahtar, etiket, bolme, arazi ortusu oranlari
  grid_1km_terrain.parquet        topografya (hucre bazli, statik)
  grid_1km_human.parquet          insan faktoru (hucre bazli, statik)
  features_fwi.parquet            FWI ailesi + gecikmeli turevler (gunluk)
  features_era5_monthly.parquet   toprak nemi, LAI, yuzey sicakligi (aylik)

Atilan sutunlar:
  lai_high_anom, lai_low_anom  ERA5-Land LAI'yi sabit aylik iklimsel ortalama
                               olarak kullanir; yillar arasi degismez, dolayisiyla
                               anomali her zaman ~0'dir ve bilgi tasimaz.

Eklenen zaman kodlamalari:
  doy_sin, doy_cos  yilin gununun dairesel kodlamasi (31 Aralik ile 1 Ocak
                    arasindaki suregi korur; ham gun numarasi bunu bozar)

Cikti:
  data_derived/orion_tr_dataset.parquet
  _reports/dataset_card.md
  _reports/column_dictionary.csv
  _reports/dataset_sha256.csv

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_dataset.py
    .venv\\Scripts\\python.exe scripts\\build_dataset.py --drop-incomplete
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
OUT = DERIVED / "orion_tr_dataset.parquet"

FRAME = DERIVED / "sample_frame.parquet"
TERRAIN = DERIVED / "grid_1km_terrain.parquet"
HUMAN = DERIVED / "grid_1km_human.parquet"
FWI = DERIVED / "features_fwi.parquet"
ERA5 = DERIVED / "features_era5_monthly.parquet"

DROP = ["lai_high_anom", "lai_low_anom"]

DESCRIPTIONS = {
    "cell_id": "1 km grid hucre kimligi (satir*1950+sutun)",
    "day_idx": "2016-01-01'den itibaren gun sayisi",
    "date": "Gozlem gunu (t)",
    "label": "t+1..t+3 araliginda bu hucrede tutusma oldu mu (0/1)",
    "split": "Zamansal bolme: train 2016-2022, val 2023, test 2024-2025",
    "yil": "Yil", "ay": "Ay (1-12)",
    "lat_center": "Hucre merkezi enlem", "lon_center": "Hucre merkezi boylam",
    "doy_sin": "Yilin gununun dairesel kodlamasi (sinus)",
    "doy_cos": "Yilin gununun dairesel kodlamasi (kosinus)",
    "frac_agac": "Hucrede agac ortusu orani (WorldCover 10)",
    "frac_maki": "Calilik/maki orani (WorldCover 20)",
    "frac_otlak": "Otlak orani (WorldCover 30)",
    "frac_tarim": "Tarim alani orani (WorldCover 40)",
    "frac_yapili": "Yapili alan orani (WorldCover 50)",
    "frac_ciplak": "Ciplak/seyrek ortu orani (WorldCover 60)",
    "frac_dogal": "Dogal yanici ortu orani (agac + maki)",
    "elev_mean": "Ortalama yukseklik (m)",
    "elev_std": "Yukseklik standart sapmasi (arazi puruzlulugu, m)",
    "elev_range": "Hucre ici yukseklik farki (m)",
    "slope_mean": "Ortalama egim (derece)",
    "slope_max": "Maksimum egim (derece)",
    "aspect_sin": "Baki dogululuk bileseni (-1 bati, +1 dogu)",
    "aspect_cos": "Baki kuzeylik bileseni (-1 guney, +1 kuzey)",
    "dist_built_km": "En yakin yapili alana mesafe (km)",
    "dist_crop_km": "En yakin tarim alanina mesafe (km)",
    "dist_water_km": "En yakin su kutlesine mesafe (km)",
    "natural_5x5": "5 km cevrede dogal ortu orani (yakit surekliligi)",
    "built_5x5": "5 km cevrede yapili alan orani",
    "crop_5x5": "5 km cevrede tarim alani orani",
    "crop_edge": "Orman-tarim gecis kusaginda mi (0/1)",
    "fwi": "Fire Weather Index (t gunu)",
    "fwi_mean7": "FWI son 7 gun ortalamasi",
    "fwi_mean30": "FWI son 30 gun ortalamasi",
    "fwi_max7": "FWI son 7 gun maksimumu",
    "ffmc": "Fine Fuel Moisture Code (ince yakit nemi)",
    "dmc": "Duff Moisture Code (orta katman nemi)",
    "dc": "Drought Code (derin katman kurakligi)",
    "isi": "Initial Spread Index (baslangic yayilim hizi)",
    "bui": "Build Up Index (yanabilir yakit miktari)",
    "soil_w1": "Toprak nemi 0-7 cm (m3/m3, aylik)",
    "soil_w2": "Toprak nemi 7-28 cm", "soil_w3": "Toprak nemi 28-100 cm",
    "soil_w4": "Toprak nemi 100-289 cm",
    "soil_w1_anom": "Toprak nemi 0-7 cm, hucrenin kendi aylik normalinden sapma",
    "soil_w2_anom": "Toprak nemi 7-28 cm anomalisi",
    "soil_w3_anom": "Toprak nemi 28-100 cm anomalisi",
    "soil_w4_anom": "Toprak nemi 100-289 cm anomalisi",
    "lai_high": "Yaprak alan indeksi, yuksek bitki ortusu (iklimsel, aylik)",
    "lai_low": "Yaprak alan indeksi, alcak bitki ortusu (iklimsel, aylik)",
    "skin_temp": "Yuzey sicakligi (K, aylik)",
    "fwi_grid_shift": "FWI gridinde kiyi duzeltmesi icin kaydirma (hucre)",
    "era5_grid_shift": "ERA5 gridinde kiyi duzeltmesi icin kaydirma (hucre)",
}

GROUPS = {
    "anahtar": ["cell_id", "day_idx", "date", "yil", "ay", "split"],
    "etiket": ["label"],
    "konum_zaman": ["lat_center", "lon_center", "doy_sin", "doy_cos"],
    "arazi_ortusu": ["frac_agac", "frac_maki", "frac_otlak", "frac_tarim",
                     "frac_yapili", "frac_ciplak", "frac_dogal"],
    "topografya": ["elev_mean", "elev_std", "elev_range", "slope_mean",
                   "slope_max", "aspect_sin", "aspect_cos"],
    "insan_faktoru": ["dist_built_km", "dist_crop_km", "dist_water_km",
                      "natural_5x5", "built_5x5", "crop_5x5", "crop_edge"],
    "yangin_indeksleri": ["fwi", "fwi_mean7", "fwi_mean30", "fwi_max7",
                          "ffmc", "dmc", "dc", "isi", "bui"],
    "toprak_bitki": ["soil_w1", "soil_w2", "soil_w3", "soil_w4",
                     "soil_w1_anom", "soil_w2_anom", "soil_w3_anom",
                     "soil_w4_anom", "lai_high", "lai_low", "skin_temp"],
    "teknik": ["fwi_grid_shift", "era5_grid_shift"],
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-incomplete", action="store_true",
                    help="Topografyasi eksik satirlari cikar")
    a = ap.parse_args()

    for p in (FRAME, TERRAIN, HUMAN, FWI, ERA5):
        if not p.exists():
            raise SystemExit(f"HATA: {p} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    print("Katmanlar okunuyor...")
    df = pd.read_parquet(FRAME)
    n0 = len(df)
    print(f"  cerceve      : {n0:,} satir")

    for name, path, keys in (
        ("topografya", TERRAIN, ["cell_id"]),
        ("insan", HUMAN, ["cell_id"]),
        ("fwi", FWI, ["cell_id", "day_idx"]),
        ("era5", ERA5, ["cell_id", "day_idx"]),
    ):
        part = pd.read_parquet(path)
        before = len(df)
        df = df.merge(part, on=keys, how="left")
        print(f"  {name:12s} : +{len(part.columns) - len(keys)} sutun")
        if len(df) != before:
            raise SystemExit(f"HATA: {name} birlestirmesi satir sayisini degistirdi "
                             f"({before:,} -> {len(df):,}). Anahtar mukerrer olabilir.")

    df = df.drop(columns=[c for c in DROP if c in df.columns])

    # ---- zaman kodlamasi
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # ---- sutun sirasi
    ordered = [c for g in GROUPS.values() for c in g if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]

    # ---- eksik analizi
    feat_cols = [c for g, cols in GROUPS.items() if g not in ("anahtar", "etiket", "teknik")
                 for c in cols if c in df.columns]
    incomplete = df[feat_cols].isna().any(axis=1)
    n_inc = int(incomplete.sum())

    if a.drop_incomplete and n_inc:
        df = df[~incomplete].reset_index(drop=True)
        print(f"\nEksik satir cikarildi: {n_inc:,}")

    df.to_parquet(OUT, index=False)

    # ------------------------------------------------------------ raporlar
    dict_rows = []
    for grp, cols in GROUPS.items():
        for c in cols:
            if c in df.columns:
                dict_rows.append({
                    "sutun": c, "grup": grp,
                    "tip": str(df[c].dtype),
                    "eksik_oran": f"{df[c].isna().mean():.4f}",
                    "aciklama": DESCRIPTIONS.get(c, ""),
                })
    pd.DataFrame(dict_rows).to_csv(REPORT / "column_dictionary.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame([{"file": OUT.name, "bytes": OUT.stat().st_size,
                   "sha256": sha256_of(OUT)}]).to_csv(
        REPORT / "dataset_sha256.csv", index=False)

    L: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        L.append(s)

    log("# ORION-TR — Veri Karti")
    log()
    log(f"**Uretim (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ")
    log(f"**Satir:** {len(df):,}  |  **Sutun:** {len(df.columns)}  |  "
        f"**Boyut:** {OUT.stat().st_size / 1e6:.1f} MB")
    log()
    log("## Tanim")
    log()
    log("| Alan | Deger |")
    log("|---|---|")
    log("| Satir birimi | Bir 1 km hucre, bir gun |")
    log("| Etiket | `label` = t+1..t+3 araliginda bu hucrede orman yangini tutusmasi |")
    log("| Cografya | Turkiye, resmi idari sinir (geoBoundaries ADM0) |")
    log("| Grid | 0.01 derece (~1 km), EPSG:4326 |")
    log("| Donem | 2016-01-01 .. 2025-12-31 (3.653 gun) |")
    log("| Ornekleme | Tum pozitifler + 1:15 oraninda rastgele negatif |")
    log("| Maske | Yalniz dogal ortu orani >= %30 olan hucreler |")
    log()

    log("## Sinif dengesi ve bolme")
    log()
    log("| Bolme | Yillar | Satir | Pozitif | Oran |")
    log("|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        if d.empty:
            continue
        log(f"| {s} | {d['yil'].min()}-{d['yil'].max()} | {len(d):,} | "
            f"{int(d['label'].sum()):,} | {d['label'].mean():.2%} |")
    log()
    log("> Bolme zamansaldir, rastgele degildir. Ayni yanginin komsu gunleri hem")
    log("> egitimde hem testte olsaydi model kopya ceker, basari sahte cikardi.")
    log("> Bolmeler arasi pozitif orani farki gercek yillar arasi degiskenliktir.")
    log("> Degerlendirmede accuracy yerine PR-AUC ve recall kullaniniz.")
    log()

    log("## Oznitelik gruplari")
    log()
    log("| Grup | Sutun | Kaynak |")
    log("|---|---|---|")
    src = {
        "konum_zaman": "Turetilmis", "arazi_ortusu": "ESA WorldCover 10 m 2021",
        "topografya": "Copernicus DEM GLO-30", "insan_faktoru": "WorldCover turevi",
        "yangin_indeksleri": "CEMS cems-fire-historical-v1",
        "toprak_bitki": "ERA5-Land aylik", "teknik": "Turetilmis",
    }
    for g, cols in GROUPS.items():
        if g in ("anahtar", "etiket"):
            continue
        n = len([c for c in cols if c in df.columns])
        log(f"| {g} | {n} | {src.get(g, '')} |")
    log()

    log("## Eksik veri")
    log()
    log(f"- En az bir ozniteligi eksik satir: **{n_inc:,}** ({n_inc / n0:.2%})")
    miss = df[feat_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        log()
        log("| Sutun | Eksik oran |")
        log("|---|---|")
        for c, v in miss.items():
            log(f"| {c} | {v:.2%} |")
        log()
        log("> Eksiklik DEM karosu bulunmayan kiyi hucrelerinden gelir.")
        log("> `--drop-incomplete` ile cikarilabilir veya agac tabanli modellerde")
        log("> (LightGBM, XGBoost) oldugu gibi birakilabilir; bu modeller eksik")
        log("> degeri dogal olarak isler.")
    else:
        log("- Eksik deger yok.")
    log()

    log("## Bilinen sinirlar")
    log()
    log("1. **LAI iklimseldir.** ERA5-Land yaprak alan indeksini sabit aylik")
    log("   ortalama olarak kullanir; yillar arasi degismez. Bu nedenle bitki")
    log("   stresi bilgisi tasimaz ve anomali sutunlari atilmistir. Gercek bitki")
    log("   stresi icin MODIS/Sentinel NDVI gereklidir (v2).")
    log("2. **Toprak nemi ve LAI aylik cozunurluktedir.** Gunluk yangin dinamigi")
    log("   FWI ailesi tarafindan tasinir.")
    log("3. **Insan faktoru dolayli olcilmustur.** Gercek nufus yogunlugu (GHSL)")
    log("   ve yol agi (OSM) yerine yapili alana mesafe kullanilmistir (v2).")
    log("4. **Etiket uydu tespitine dayanir.** Yogun bulut altindaki veya cok")
    log("   kucuk yanginlar gorulmemis olabilir; 'tespit yok' 'yangin yok'")
    log("   anlamina gelmez.")
    log("5. **Sensor homojenligi icin** etikette yalniz VIIRS_SNPP_SP ve MODIS_SP")
    log("   kullanilmistir. VIIRS_NOAA20_SP 2018-04'te devreye girdigi icin")
    log("   kalibrasyon kirilimi olustururdu.")
    log("6. **Maki alt temsil edilmektedir.** WorldCover Akdeniz makisinin cogunu")
    log("   agac ortusu veya otlak olarak siniflandirir; CORINE'in `323")
    log("   Sclerophyllous vegetation` sinifi v2'de eklenmelidir.")
    log()

    log("## Kaynaklar ve lisanslar")
    log()
    log("| Kaynak | Urun | Lisans |")
    log("|---|---|---|")
    log("| NASA FIRMS | VIIRS_SNPP_SP, MODIS_SP, VIIRS_NOAA20_SP | NASA Open Data |")
    log("| geoBoundaries | gbOpen TUR ADM0 | CC BY 4.0 |")
    log("| ESA WorldCover | 10 m 2021 v200 | CC BY 4.0 |")
    log("| Copernicus DEM | GLO-30 Public | Copernicus lisansi |")
    log("| Copernicus EMS | cems-fire-historical-v1, sistem 4_1 | Copernicus lisansi |")
    log("| Copernicus C3S | reanalysis-era5-land-monthly-means | CC BY 4.0 |")
    log()
    log("Atif metinleri `data_raw/*/source.json` ve `*/metadata/` altindadir.")
    log()
    log("## Uretim zinciri")
    log()
    log("```")
    for i, s in enumerate([
        "download_firms.py         FIRMS arsivi (2.028 pencere dosyasi)",
        "build_firms_events.py     temizlik, type filtresi (1.699.297 -> 1.002.276)",
        "clip_firms_turkey.py      Turkiye sinirina kirpma (-> 614.720)",
        "assign_landcover.py       arazi ortusu sinifi atama",
        "build_grid.py             1 km grid + ortu oranlari (812.032 hucre)",
        "build_fire_celldays.py    tutusma/devam ayrimi (-> 18.764 tutusma)",
        "build_sample_frame.py     pozitif + negatif ornekleme (900.192 satir)",
        "build_terrain.py          Copernicus DEM -> topografya",
        "build_human_features.py   insan faktoru turetme",
        "download_fwi.py           CEMS FWI (60 dosya)",
        "join_fwi.py               FWI + gecikmeli degiskenler",
        "download_era5_monthly.py  ERA5-Land aylik",
        "join_era5_monthly.py      toprak nemi + LAI + anomaliler",
        "build_dataset.py          nihai birlestirme",
    ], 1):
        log(f"{i:2d}. {s}")
    log("```")
    log()
    log("## Cikti")
    log()
    log(f"- `{OUT.name}` ({OUT.stat().st_size / 1e6:.1f} MB)")
    log("- `_reports/column_dictionary.csv` — sutun sozlugu")
    log("- `_reports/dataset_sha256.csv` — butunluk kaydi")

    (REPORT / "dataset_card.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nVeri karti: {REPORT / 'dataset_card.md'}")


if __name__ == "__main__":
    main()
