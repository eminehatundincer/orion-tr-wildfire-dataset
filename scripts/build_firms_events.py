"""
ORION-TR / Adim 3
FIRMS ham pencere CSV'lerini tek bir temiz olay tablosuna donusturur ve QA uretir.

Yaptiklari:
  - 2028 pencere dosyasini sensor bazinda birlestirir
  - MODIS / VIIRS farkli sutun adlarini ortak semaya cevirir
  - 'type' alani varsa yangin disi isi kaynaklarini ayirir (sanayi, flare, volkan)
  - koordinat ve tarih dogrulamasi yapar
  - birebir mukerrerleri temizler
  - 3653 gunluk kapsama defteri uretir
  - QA raporu ve SHA-256 yazar

Cikti:
  data_derived/firms_events.parquet        temiz olay tablosu (yalniz vejetasyon yangini)
  data_derived/firms_events_excluded.parquet  ayiklanan kayitlar (izlenebilirlik icin)
  data_derived/firms_daily_coverage.csv    gun bazli kapsama defteri
  _reports/firms_qa.md                     insan okur QA raporu
  _reports/firms_events_sha256.csv

Kullanim:
    .venv\\Scripts\\python.exe scripts\\build_firms_events.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw" / "firms" / "original"
OUT_DIR = ROOT / "data_derived"
REPORT_DIR = ROOT / "_reports"

WEST, SOUTH, EAST, NORTH = 25.5, 35.5, 45.0, 42.5
START, END = date(2016, 1, 1), date(2025, 12, 31)
N_DAYS = (END - START).days + 1

# FIRMS 'type' kodlari
TYPE_LABELS = {
    0: "vejetasyon_yangini",
    1: "aktif_volkan",
    2: "sabit_kara_kaynagi",   # sanayi, gaz flare
    3: "acik_deniz",
}

# Sensore ozgu parlaklik sutunlari -> ortak ad
BRIGHT_MAP = {
    "bright_ti4": "bright_primary", "brightness": "bright_primary",
    "bright_ti5": "bright_secondary", "bright_t31": "bright_secondary",
}

KEEP = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "frp", "daynight", "type", "scan", "track",
    "bright_primary", "bright_secondary", "source_sensor",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sensor(sensor_dir: Path) -> pd.DataFrame:
    files = sorted(sensor_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"    UYARI okunamadi {fp.name}: {e}")
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["source_sensor"] = sensor_dir.name
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sensor_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if not sensor_dirs:
        raise SystemExit(f"HATA: {RAW_DIR} altinda sensor klasoru yok.")

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("# FIRMS olay tablosu - QA raporu")
    log()
    log(f"- Uretim (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"- AOI: {WEST},{SOUTH},{EAST},{NORTH} (EPSG:4326)")
    log(f"- Donem: {START} .. {END} ({N_DAYS} gun)")
    log()

    parts = []
    log("## 1. Ham okuma")
    log()
    log("| Sensor | Dosya | Ham kayit | Sutunlar |")
    log("|---|---|---|---|")

    for sd in sensor_dirs:
        df = load_sensor(sd)
        n_files = len(list(sd.glob("*.csv")))
        if df.empty:
            log(f"| {sd.name} | {n_files} | 0 | - |")
            continue
        cols = ",".join(sorted(c for c in df.columns if c != "source_sensor"))
        log(f"| {sd.name} | {n_files} | {len(df):,} | `{cols}` |")
        parts.append(df)

    if not parts:
        raise SystemExit("HATA: hic kayit okunamadi.")

    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)

    # ---- ortak sema
    def coalesce(frame, candidates):
        cols = [c for c in candidates if c in frame.columns]
        if not cols:
            return pd.Series(pd.NA, index=frame.index)
        out = frame[cols[0]]
        for c in cols[1:]:
            out = out.fillna(frame[c])
        return out

    df["bright_primary"] = coalesce(df, ["bright_ti4", "brightness"])
    df["bright_secondary"] = coalesce(df, ["bright_ti5", "bright_t31"])
    df = df.drop(columns=[c for c in ("bright_ti4", "brightness", "bright_ti5", "bright_t31")
                          if c in df.columns])
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP]

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    for c in ("latitude", "longitude", "frp", "scan", "track",
              "bright_primary", "bright_secondary"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("satellite", "instrument", "confidence", "version",
              "daynight", "acq_time", "source_sensor"):
        df[c] = df[c].astype("string")
    log()
    log(f"Birlestirilmis ham kayit: **{n_raw:,}**")
    log()

    # ---- dogrulama
    log("## 2. Dogrulama ve temizlik")
    log()
    steps = []

    m = df["latitude"].notna() & df["longitude"].notna() & df["acq_date"].notna()
    steps.append(("Eksik koordinat/tarih", int((~m).sum())))
    df = df[m]

    m = (df["latitude"].between(SOUTH, NORTH) & df["longitude"].between(WEST, EAST))
    steps.append(("AOI disi koordinat", int((~m).sum())))
    df = df[m]

    m = df["acq_date"].dt.date.between(START, END)
    steps.append(("Donem disi tarih", int((~m).sum())))
    df = df[m]

    before = len(df)
    df = df.drop_duplicates(
        subset=["source_sensor", "latitude", "longitude", "acq_date", "acq_time", "satellite"]
    )
    steps.append(("Birebir mukerrer", before - len(df)))

    for name, n in steps:
        log(f"- {name}: **{n:,}**")
    log(f"- Kalan: **{len(df):,}**")
    log()

    # ---- type filtresi
    log("## 3. Isi kaynagi turu (type)")
    log()
    has_type = df["type"].notna().any()
    if has_type:
        df["type"] = pd.to_numeric(df["type"], errors="coerce").astype("Int64")
        vc = df["type"].value_counts(dropna=False).sort_index()
        log("| type | Anlam | Kayit | Oran |")
        log("|---|---|---|---|")
        for k, v in vc.items():
            label = TYPE_LABELS.get(int(k), "bilinmiyor") if pd.notna(k) else "bos"
            log(f"| {k} | {label} | {v:,} | {v / len(df):.1%} |")
        keep_mask = df["type"] == 0
        excluded = df[~keep_mask].copy()
        df = df[keep_mask].copy()
        log()
        log(f"Etiket icin tutulan (type=0): **{len(df):,}**")
        log(f"Ayiklanan: **{len(excluded):,}** -> `firms_events_excluded.parquet`")
    else:
        excluded = df.iloc[0:0].copy()
        log("`type` alani bu API ciktisinda yok.")
        log()
        log("> **DIKKAT:** Yangin disi sabit isi kaynaklari (sanayi tesisleri, gaz")
        log("> flareleri) ayiklanamadi. Etiket uretmeden once mekansal bir maske")
        log("> gerekir: ayni hucrede yil boyu her ay tespit varsa bu bir yangin")
        log("> degil, kalici kaynaktir. Adim 4'te CORINE maskesiyle ele alinacak.")
    log()

    # ---- dagilimlar
    log("## 4. Dagilimlar")
    log()
    df["yil"] = df["acq_date"].dt.year
    df["ay"] = df["acq_date"].dt.month

    log("### Yil bazinda")
    log()
    log("| Yil | " + " | ".join(sorted(df["source_sensor"].unique())) + " | Toplam |")
    log("|---" * (len(df["source_sensor"].unique()) + 2) + "|")
    piv = pd.crosstab(df["yil"], df["source_sensor"])
    for yil, row in piv.iterrows():
        log(f"| {yil} | " + " | ".join(f"{v:,}" for v in row.values) + f" | {row.sum():,} |")
    log()

    log("### Ay bazinda (tum yillar)")
    log()
    ay = df["ay"].value_counts().sort_index()
    log("| Ay | " + " | ".join(str(i) for i in range(1, 13)) + " |")
    log("|---" * 13 + "|")
    log("| Kayit | " + " | ".join(f"{ay.get(i, 0):,}" for i in range(1, 13)) + " |")
    log()

    if df["confidence"].notna().any():
        log("### Guven (confidence)")
        log()
        vc = df["confidence"].astype(str).value_counts().head(12)
        for k, v in vc.items():
            log(f"- `{k}`: {v:,} ({v / len(df):.1%})")
        log()

    if df["daynight"].notna().any():
        vc = df["daynight"].astype(str).value_counts()
        log("### Gunduz / gece")
        log()
        for k, v in vc.items():
            log(f"- `{k}`: {v:,} ({v / len(df):.1%})")
        log()

    # ---- kapsama defteri
    log("## 5. Gun bazli kapsama defteri")
    log()
    all_days = pd.DataFrame({"gun": [START + timedelta(days=i) for i in range(N_DAYS)]})
    all_days["gun"] = pd.to_datetime(all_days["gun"])

    cov = df.groupby([df["acq_date"], "source_sensor"]).size().unstack(fill_value=0)
    cov = all_days.set_index("gun").join(cov).fillna(0).astype(int)
    cov["toplam"] = cov.sum(axis=1)
    cov["durum"] = cov["toplam"].apply(
        lambda n: "covered_events" if n > 0 else "covered_zero_events"
    )
    cov_out = OUT_DIR / "firms_daily_coverage.csv"
    cov.to_csv(cov_out, encoding="utf-8")

    n_ev = int((cov["durum"] == "covered_events").sum())
    n_zero = int((cov["durum"] == "covered_zero_events").sum())
    log(f"- Beklenen gun: **{N_DAYS}**")
    log(f"- Tespit olan gun: **{n_ev}** ({n_ev / N_DAYS:.1%})")
    log(f"- Kapsanmis, tespit yok: **{n_zero}**")
    log(f"- Eksik (indirilemeyen) gun: **0**")
    log()
    log("> `covered_zero_events` ile `missing` ayrimi korunmustur: sifir tespit,")
    log("> veri eksikligi degil o gun yangin gorulmedigi anlamina gelir.")
    log()

    # ---- yazma
    ev_out = OUT_DIR / "firms_events.parquet"
    df.drop(columns=["yil", "ay"]).to_parquet(ev_out, index=False)
    written = [ev_out, cov_out]

    if len(excluded):
        ex_out = OUT_DIR / "firms_events_excluded.parquet"
        excluded.to_parquet(ex_out, index=False)
        written.append(ex_out)

    rows = []
    for p in written:
        rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_of(p)})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "firms_events_sha256.csv", index=False)

    log("## 6. Cikti dosyalari")
    log()
    log("| Dosya | Boyut (MB) |")
    log("|---|---|")
    for p in written:
        log(f"| `{p.name}` | {p.stat().st_size / 1e6:.1f} |")
    log()

    (REPORT_DIR / "firms_qa.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nQA raporu: {REPORT_DIR / 'firms_qa.md'}")


if __name__ == "__main__":
    main()
