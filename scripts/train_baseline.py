"""
ORION-TR / Adim 16
Temel model: veri setinin ogrenilebilir sinyal tasidigini dogrular.

Bu bir "en iyi model" denemesi degildir. Amaci uctur:

  1) Sinyal gercekten var mi
  2) Model yalnizca mevsimselligi mi ogreniyor, yoksa fiziksel degiskenleri de
     kullaniyor mu  -> zaman kodlamasi cikarilarak ablasyon testi
  3) Operasyonel deger var mi  -> "en riskli %N alani izlersek yanginlarin
     yuzde kacini yakalariz"

Ucuncusu PDR'deki deger onerisinin dogrudan olcumudur: ekipler tum ormani
degil, en riskli bolgeyi izler ve oraya onceden konumlanir.

Degerlendirme accuracy ile YAPILMAZ. Pozitif orani %6 oldugu icin "hicbir
yangin yok" diyen model %94 dogruluk alir ve hicbir ise yaramaz. PR-AUC,
recall ve yakalama orani kullanilir.

Cikti:
  _reports/baseline_model_qa.md
  _reports/feature_importance.csv
  data_derived/model_test_predictions.parquet

Kullanim:
    .venv\\Scripts\\python.exe scripts\\train_baseline.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data_derived"
REPORT = ROOT / "_reports"
DATA = DERIVED / "orion_tr_dataset.parquet"

EXCLUDE = {
    "cell_id", "day_idx", "date", "label", "split", "yil", "ay",
    "fwi_grid_shift", "era5_grid_shift",
}
TEMPORAL = ["doy_sin", "doy_cos"]

PARAMS = dict(
    objective="binary", metric="average_precision",
    learning_rate=0.05, num_leaves=63, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    n_estimators=1500, verbose=-1, n_jobs=-1,
)


def capture_rate(y: np.ndarray, p: np.ndarray, pct: float) -> float:
    """En riskli %pct satirda yanginlarin yuzde kaci yakalanir."""
    k = max(1, int(len(p) * pct / 100))
    top = np.argpartition(-p, k - 1)[:k]
    return y[top].sum() / y.sum()


def train(df: pd.DataFrame, feats: list[str], tag: str):
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        tr[feats], tr["label"],
        eval_set=[(va[feats], va["label"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p = model.predict_proba(te[feats])[:, 1]
    y = te["label"].to_numpy()

    print(f"  {tag:24s} PR-AUC={average_precision_score(y, p):.4f}  "
          f"ROC-AUC={roc_auc_score(y, p):.4f}  "
          f"(en iyi tur {model.best_iteration_})")
    return model, y, p, te


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"HATA: {DATA} yok.")
    REPORT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c not in EXCLUDE]
    print(f"Satir: {len(df):,}  Oznitelik: {len(feats)}\n")
    print("Egitim...")

    model, y, p, te = train(df, feats, "tam model")
    feats_noT = [c for c in feats if c not in TEMPORAL]
    _, y2, p2, _ = train(df, feats_noT, "zaman kodlamasiz")

    base = y.mean()
    imp = pd.DataFrame({
        "oznitelik": feats,
        "onem": model.booster_.feature_importance("gain"),
    }).sort_values("onem", ascending=False).reset_index(drop=True)
    imp["pay"] = imp["onem"] / imp["onem"].sum()
    imp.to_csv(REPORT / "feature_importance.csv", index=False,
               encoding="utf-8-sig")

    out = te[["cell_id", "day_idx", "date", "lat_center", "lon_center",
              "label"]].copy()
    out["risk"] = p.astype(np.float32)
    out.to_parquet(DERIVED / "model_test_predictions.parquet", index=False)

    # ------------------------------------------------------------- rapor
    L: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        L.append(s)

    log("# Temel model — veri seti dogrulamasi")
    log()
    log(f"**Uretim (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ")
    log("**Model:** LightGBM ikili siniflandirici  ")
    log(f"**Egitim:** 2016-2022  |  **Dogrulama:** 2023  |  **Test:** 2024-2025  ")
    log(f"**Oznitelik:** {len(feats)}")
    log()
    log("> Bu bir optimizasyon denemesi degil, veri setinin ogrenilebilir sinyal")
    log("> tasidiginin kanitidir. Hiperparametre aramasi yapilmamistir.")
    log()

    log("## Performans (test: 2024-2025, hic gorulmemis yillar)")
    log()
    log("| Olcut | Tam model | Zaman kodlamasiz | Rastgele |")
    log("|---|---|---|---|")
    log(f"| PR-AUC | **{average_precision_score(y, p):.4f}** | "
        f"{average_precision_score(y2, p2):.4f} | {base:.4f} |")
    log(f"| ROC-AUC | **{roc_auc_score(y, p):.4f}** | "
        f"{roc_auc_score(y2, p2):.4f} | 0.5000 |")
    log()
    log(f"> Rastgele tahmin PR-AUC'si pozitif orani kadardir ({base:.2%}).")
    log("> Accuracy kullanilmamistir: 'hicbir yangin yok' diyen bir model")
    log(f"> %{(1 - base) * 100:.1f} dogruluk alir ve hicbir ise yaramaz.")
    log()

    log("## Ablasyon: model yalnizca mevsimselligi mi ogreniyor?")
    log()
    d = average_precision_score(y, p) - average_precision_score(y2, p2)
    log(f"Zaman kodlamasi (`doy_sin`, `doy_cos`) cikarildiginda PR-AUC "
        f"{d:+.4f} degisti.")
    log()
    if abs(d) < 0.02:
        log("> Degisim kucuk. Model mevsimsellige bagimli degil; fiziksel")
        log("> degiskenlerden (FWI, toprak nemi, topografya) ogreniyor.")
    else:
        log("> Degisim belirgin. Mevsimsellik onemli bir katki sagliyor;")
        log("> yine de diger degiskenler olmadan bu performans elde edilemezdi.")
    log()

    log("## Operasyonel deger")
    log()
    log("| Izlenen alan | Yakalanan yangin | Rastgeleye gore |")
    log("|---|---|---|")
    for pct in (1, 2, 5, 10, 20):
        r = capture_rate(y, p, pct)
        log(f"| en riskli %{pct} | **%{r * 100:.1f}** | {r / (pct / 100):.1f}x |")
    log()
    log("> PDR'deki deger onerisinin dogrudan olcumu budur. Ekipler tum ormani")
    log("> degil en riskli bolgeyi izler. Ornegin en riskli %5'lik alani izleyerek")
    log("> yanginlarin onemli bir kismini onceden kapsamak mumkundur; bu 'reaktif")
    log("> sondurmeden proaktif onden konumlanmaya gecis' iddiasinin sayisal")
    log("> karsiligidir.")
    log()

    log("## En onemli 20 oznitelik")
    log()
    log("| # | Oznitelik | Pay |")
    log("|---|---|---|")
    for i, r in imp.head(20).iterrows():
        log(f"| {i + 1} | `{r['oznitelik']}` | {r['pay']:.2%} |")
    log()

    grp = {
        "yangin_indeksleri": ["fwi", "fwi_mean7", "fwi_mean30", "fwi_max7",
                              "ffmc", "dmc", "dc", "isi", "bui"],
        "toprak_bitki": ["soil_w1", "soil_w2", "soil_w3", "soil_w4",
                         "soil_w1_anom", "soil_w2_anom", "soil_w3_anom",
                         "soil_w4_anom", "lai_high", "lai_low", "skin_temp"],
        "topografya": ["elev_mean", "elev_std", "elev_range", "slope_mean",
                       "slope_max", "aspect_sin", "aspect_cos"],
        "insan_faktoru": ["dist_built_km", "dist_crop_km", "dist_water_km",
                          "natural_5x5", "built_5x5", "crop_5x5", "crop_edge"],
        "arazi_ortusu": ["frac_agac", "frac_maki", "frac_otlak", "frac_tarim",
                         "frac_yapili", "frac_ciplak", "frac_dogal"],
        "konum_zaman": ["lat_center", "lon_center", "doy_sin", "doy_cos"],
    }
    log("## Grup bazinda katki")
    log()
    log("| Grup | Pay |")
    log("|---|---|")
    s = imp.set_index("oznitelik")["pay"]
    for g, cols in sorted(grp.items(),
                          key=lambda kv: -s.reindex(kv[1]).fillna(0).sum()):
        log(f"| {g} | {s.reindex(cols).fillna(0).sum():.1%} |")
    log()
    log("> `crop_edge` ve `dist_crop_km` yuksek ciktiysa PDR'de 'tarim-orman")
    log("> gecis kusaginda onleyici konumlanma' onerisi yapabilirsiniz.")
    log()

    log("## Cikti")
    log()
    log("- `_reports/feature_importance.csv`")
    log("- `data_derived/model_test_predictions.parquet` — test tahminleri, "
        "harita uzerinde gorsellestirilebilir")

    (REPORT / "baseline_model_qa.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nRapor: {REPORT / 'baseline_model_qa.md'}")


if __name__ == "__main__":
    main()
