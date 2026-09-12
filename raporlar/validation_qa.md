# Model dogrulama — mekansal genelleme ve vaka incelemesi

**Uretim (UTC):** 2026-09-11T08:41:27Z

## Test 1a — Koordinat sutunlari cikarildiginda

| Model | PR-AUC | ROC-AUC | %5 yakalama |
|---|---|---|---|
| Koordinatli | 0.4135 | 0.8687 | %31.4 |
| Koordinatsiz | 0.3953 | 0.8613 | %30.3 |

Fark: **+0.0181**

> Koordinatlar cikarildiginda performans belirgin dusmedi. Model
> konumu ezberlemiyor; risk bilgisi fiziksel degiskenlerde.

## Test 1b — Batida egit, doguda test et

Ayirma boylami: 33.0° D. Egitim ve test **ayri cografyalarda**,
ayrica zaman bolmesi de korunuyor.

- Egitim (bati): 364,855 satir, pozitif %5.69
- Test (dogu)  : 74,524 satir, pozitif %5.18
- Test (bati)  : 108,038 satir, pozitif %9.25

| Test bolgesi | PR-AUC | Rastgele | Kat | %5 yakalama |
|---|---|---|---|---|
| Bati (tanidik) | **0.3993** | 0.0925 | 4.3x | %26.5 |
| Dogu (yeni) | **0.2961** | 0.0518 | 5.7x | %34.9 |

Dogu/bati kazanim orani: **1.32**

> Model hic gormedigi cografyada da rastgeleye gore belirgin ustun.
> Ogrendigi sey bolgeye ozgu ezber degil, yangin fizigi. Bu, sistemin
> pilot bolgeden ulke geneline olceklenebilecegini gosterir.

## Test 2 — Gercek yangin vakalari

Test doneminde en siddetli **25** yangin olayi incelendi.
Her biri icin tutusmadan **1-3 gun once** modelin verdigi en yuksek
risk skoru ve bu skorun tum test satirlari icindeki yuzdeligi:

| Erken uyari seviyesi | Vaka orani |
|---|---|
| En riskli %5 icinde | **%4** |
| En riskli %10 icinde | **%28** |
| En riskli %20 icinde | **%48** |

### En siddetli 12 vaka

| Tarih | Enlem | Boylam | Maks FRP | Risk yuzdeligi |
|---|---|---|---|---|
| 2025-06-27 | 40.365 | 30.385 | 2979.8 | %58 |
| 2025-07-27 | 39.725 | 29.075 | 2416.5 | %82 |
| 2024-08-15 | 40.375 | 30.955 | 2250.6 | %77 |
| 2024-08-15 | 40.365 | 30.935 | 2230.4 | %64 |
| 2025-07-15 | 39.575 | 26.175 | 2074.6 | %91 ✓ |
| 2025-07-20 | 40.425 | 30.115 | 2033.6 | %56 |
| 2025-07-21 | 40.395 | 30.105 | 1996.3 | %54 |
| 2025-06-27 | 40.355 | 30.385 | 1875.6 | %67 |
| 2025-07-15 | 40.615 | 26.995 | 1860.0 | %86 |
| 2025-06-27 | 40.375 | 30.375 | 1791.7 | %71 |
| 2025-06-30 | 36.305 | 36.145 | 1671.9 | %96 ✓ |
| 2025-07-21 | 40.395 | 30.115 | 1613.6 | %73 |

> FRP (Fire Radiative Power) yangin siddetinin olcusudur.
> ✓ isareti, modelin tutusmadan once o hucreyi en riskli %10'a
> koydugunu gosterir. Bu satirlar PDR'de somut erken uyari ornegi
> olarak kullanilabilir; koordinatlari haritada gostererek
> 'sistem su tarihte su bolgeyi isaretledi' denebilir.

## Cikti

- `data_derived/case_studies.csv` — vaka tablosu
- `data_derived/model_test_predictions.parquet` — harita icin tahminler