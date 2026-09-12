"""
ORION-TR / Adim 17
Iki dogrulama testi.

TEST 1 — Mekansal genelleme
  Temel modelde lat_center ve lon_center oznitelik oneminin %9.7'sini aldi.
  Model yanginin NEREDE ciktigini kismen ezberliyor olabilir. Bunu olcmek icin
  model Turkiye'nin batisinda egitilip DOGUSUNDA test edilir. Ayrica koordinat
  sutunlari tamamen cikarilmis bir varyant denenir.

  Performans cokmezse model gercekten fiziksel kosullardan ogreniyor ve yeni
  bolgelere genelleme yapiyor demektir.

TEST 2 — Vaka incelemesi
  Test doneminde (2024-2025) en buyuk yangin olaylari secilir ve modelin
  tutusmadan ONCEKI gunlerde o hucreye ne risk verdigi incelenir. Bu, PDR icin
  en ikna edici kanittir: somut bir yangin, somut bir erken uyari.

Cikti:
  _reports/validation_qa.md
  data_derived/case_studies.csv

Kullanim:
    .venv\\Scripts\\python.exe scripts\\validate_model.py
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
DATA = DERIVED / "orion_tr_dataset.parquet"
CELLDAYS = DERIVED / "fire_celldays.parquet"

EXCLUDE = {"cell_id", "day_idx", "date", "label", "split", "yil", "ay",
           "fwi_grid_shift", "era5_grid_shift"}
COORDS = ["lat_center", "lon_center"]

PARAMS = dict(objective="binary", metric="average_precision",
              learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              n_estimators=1200, verbose=-1, n_jobs=-1)

SPLIT_LON = 33.0     # Turkiye'yi kabaca ikiye bolen boylam


def fit_predict(tr, va, te, feats):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr["label"],
          eval_set=[(va[feats], va["label"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict_proba(te[feats])[:, 1]
    y = te["label"].to_numpy()
    return m, y, p


def capture(y, p, pct):
    k = max(1, int(len(p) * pct / 100))
    top = np.argpartition(-p, k - 1)[:k]
    return y[top].sum() / y.sum()


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"HATA: {DATA} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    feats_all = [c for c in df.columns if c not in EXCLUDE]
    feats_nocoord = [c for c in feats_all if c not in COORDS]

    L: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        L.append(s)

    log("# Model dogrulama — mekansal genelleme ve vaka incelemesi")
    log()
    log(f"**Uretim (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log()

    # ================================================== TEST 1a: koordinatsiz
    log("## Test 1a — Koordinat sutunlari cikarildiginda")
    log()
    print("Koordinatsiz model egitiliyor...")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    _, y_a, p_a = fit_predict(tr, va, te, feats_all)
    _, y_b, p_b = fit_predict(tr, va, te, feats_nocoord)

    log("| Model | PR-AUC | ROC-AUC | %5 yakalama |")
    log("|---|---|---|---|")
    log(f"| Koordinatli | {average_precision_score(y_a, p_a):.4f} | "
        f"{roc_auc_score(y_a, p_a):.4f} | %{capture(y_a, p_a, 5) * 100:.1f} |")
    log(f"| Koordinatsiz | {average_precision_score(y_b, p_b):.4f} | "
        f"{roc_auc_score(y_b, p_b):.4f} | %{capture(y_b, p_b, 5) * 100:.1f} |")
    log()
    dd = average_precision_score(y_a, p_a) - average_precision_score(y_b, p_b)
    log(f"Fark: **{dd:+.4f}**")
    log()
    if dd < 0.03:
        log("> Koordinatlar cikarildiginda performans belirgin dusmedi. Model")
        log("> konumu ezberlemiyor; risk bilgisi fiziksel degiskenlerde.")
    else:
        log("> Koordinatlar onemli katki sagliyor. Bu kismen mesru (bazi bolgeler")
        log("> gercekten daha riskli) ama modelin yeni bolgelere genellemesi")
        log("> sinirli olabilir. Uretimde koordinatsiz varyant tercih edilebilir.")
    log()

    # ============================================ TEST 1b: bati -> dogu
    log("## Test 1b — Batida egit, doguda test et")
    log()
    log(f"Ayirma boylami: {SPLIT_LON}° D. Egitim ve test **ayri cografyalarda**,")
    log("ayrica zaman bolmesi de korunuyor.")
    log()
    print("Mekansal ayirma testi...")

    west = df["lon_center"] < SPLIT_LON
    tr_w = df[west & (df["split"] == "train")]
    va_w = df[west & (df["split"] == "val")]
    te_e = df[~west & (df["split"] == "test")]
    te_w = df[west & (df["split"] == "test")]

    log(f"- Egitim (bati): {len(tr_w):,} satir, pozitif %{tr_w['label'].mean() * 100:.2f}")
    log(f"- Test (dogu)  : {len(te_e):,} satir, pozitif %{te_e['label'].mean() * 100:.2f}")
    log(f"- Test (bati)  : {len(te_w):,} satir, pozitif %{te_w['label'].mean() * 100:.2f}")
    log()

    _, y_e, p_e = fit_predict(tr_w, va_w, te_e, feats_nocoord)
    _, y_w, p_w = fit_predict(tr_w, va_w, te_w, feats_nocoord)

    log("| Test bolgesi | PR-AUC | Rastgele | Kat | %5 yakalama |")
    log("|---|---|---|---|---|")
    for name, yy, pp in (("Bati (tanidik)", y_w, p_w), ("Dogu (yeni)", y_e, p_e)):
        ap = average_precision_score(yy, pp)
        log(f"| {name} | **{ap:.4f}** | {yy.mean():.4f} | {ap / yy.mean():.1f}x | "
            f"%{capture(yy, pp, 5) * 100:.1f} |")
    log()
    ratio_e = average_precision_score(y_e, p_e) / y_e.mean()
    ratio_w = average_precision_score(y_w, p_w) / y_w.mean()
    log(f"Dogu/bati kazanim orani: **{ratio_e / ratio_w:.2f}**")
    log()
    if ratio_e / ratio_w > 0.6:
        log("> Model hic gormedigi cografyada da rastgeleye gore belirgin ustun.")
        log("> Ogrendigi sey bolgeye ozgu ezber degil, yangin fizigi. Bu, sistemin")
        log("> pilot bolgeden ulke geneline olceklenebilecegini gosterir.")
    else:
        log("> Yeni cografyada performans belirgin dusuyor. Model bolgeye ozgu")
        log("> oruntuler ogrenmis olabilir. Ulke geneline yayilmadan once her")
        log("> bolge icin ayri kalibrasyon gerekebilir; bu PDR'de durustce")
        log("> belirtilmelidir.")
    log()

    # ================================================== TEST 2: vaka incelemesi
    log("## Test 2 — Gercek yangin vakalari")
    log()
    print("Vaka incelemesi...")

    pred = pd.DataFrame({
        "cell_id": te["cell_id"].to_numpy(),
        "day_idx": te["day_idx"].to_numpy(),
        "date": te["date"].to_numpy(),
        "lat": te["lat_center"].to_numpy(),
        "lon": te["lon_center"].to_numpy(),
        "label": y_a, "risk": p_a,
    })
    pct = pred["risk"].rank(pct=True)
    pred["risk_pct"] = (pct * 100).astype(np.float32)

    cd = pd.read_parquet(CELLDAYS)
    cd["date"] = pd.to_datetime(cd["date"])
    big = cd[(cd["is_ignition"]) & (cd["date"].dt.year >= 2024)]
    big = big.sort_values("max_frp", ascending=False).head(400)

    rows = []
    for _, ev in big.iterrows():
        cid, d = int(ev["cell_id"]), int((ev["date"] - pd.Timestamp("2016-01-01")).days)
        # tutusmadan 1-3 gun onceki satirlar
        sel = pred[(pred["cell_id"] == cid) &
                   (pred["day_idx"].between(d - 3, d - 1))]
        if sel.empty:
            continue
        rows.append({
            "tarih": ev["date"].date(),
            "enlem": round(float(sel["lat"].iloc[0]), 3),
            "boylam": round(float(sel["lon"].iloc[0]), 3),
            "max_frp": round(float(ev["max_frp"]), 1),
            "tespit": int(ev["n_det"]),
            "risk_maks": round(float(sel["risk"].max()), 4),
            "risk_yuzdelik": round(float(sel["risk_pct"].max()), 1),
        })
        if len(rows) >= 25:
            break

    if rows:
        cases = pd.DataFrame(rows).sort_values("max_frp", ascending=False)
        cases.to_csv(DERIVED / "case_studies.csv", index=False,
                     encoding="utf-8-sig")

        top5 = float((cases["risk_yuzdelik"] >= 95).mean())
        top10 = float((cases["risk_yuzdelik"] >= 90).mean())
        top20 = float((cases["risk_yuzdelik"] >= 80).mean())

        log(f"Test doneminde en siddetli **{len(cases)}** yangin olayi incelendi.")
        log("Her biri icin tutusmadan **1-3 gun once** modelin verdigi en yuksek")
        log("risk skoru ve bu skorun tum test satirlari icindeki yuzdeligi:")
        log()
        log("| Erken uyari seviyesi | Vaka orani |")
        log("|---|---|")
        log(f"| En riskli %5 icinde | **%{top5 * 100:.0f}** |")
        log(f"| En riskli %10 icinde | **%{top10 * 100:.0f}** |")
        log(f"| En riskli %20 icinde | **%{top20 * 100:.0f}** |")
        log()
        log("### En siddetli 12 vaka")
        log()
        log("| Tarih | Enlem | Boylam | Maks FRP | Risk yuzdeligi |")
        log("|---|---|---|---|---|")
        for _, r in cases.head(12).iterrows():
            mark = " ✓" if r["risk_yuzdelik"] >= 90 else ""
            log(f"| {r['tarih']} | {r['enlem']} | {r['boylam']} | "
                f"{r['max_frp']} | %{r['risk_yuzdelik']:.0f}{mark} |")
        log()
        log("> FRP (Fire Radiative Power) yangin siddetinin olcusudur.")
        log("> ✓ isareti, modelin tutusmadan once o hucreyi en riskli %10'a")
        log("> koydugunu gosterir. Bu satirlar PDR'de somut erken uyari ornegi")
        log("> olarak kullanilabilir; koordinatlari haritada gostererek")
        log("> 'sistem su tarihte su bolgeyi isaretledi' denebilir.")
    else:
        log("Eslesen vaka bulunamadi.")
    log()

    log("## Cikti")
    log()
    log("- `data_derived/case_studies.csv` — vaka tablosu")
    log("- `data_derived/model_test_predictions.parquet` — harita icin tahminler")

    (REPORT / "validation_qa.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'validation_qa.md'}")


if __name__ == "__main__":
    main()
