# Temel model — veri seti dogrulamasi

**Uretim (UTC):** 2026-09-11T08:36:44Z  
**Model:** LightGBM ikili siniflandirici  
**Egitim:** 2016-2022  |  **Dogrulama:** 2023  |  **Test:** 2024-2025  
**Oznitelik:** 45

> Bu bir optimizasyon denemesi degil, veri setinin ogrenilebilir sinyal
> tasidiginin kanitidir. Hiperparametre aramasi yapilmamistir.

## Performans (test: 2024-2025, hic gorulmemis yillar)

| Olcut | Tam model | Zaman kodlamasiz | Rastgele |
|---|---|---|---|
| PR-AUC | **0.4135** | 0.4082 | 0.0759 |
| ROC-AUC | **0.8687** | 0.8682 | 0.5000 |

> Rastgele tahmin PR-AUC'si pozitif orani kadardir (7.59%).
> Accuracy kullanilmamistir: 'hicbir yangin yok' diyen bir model
> %92.4 dogruluk alir ve hicbir ise yaramaz.

## Ablasyon: model yalnizca mevsimselligi mi ogreniyor?

Zaman kodlamasi (`doy_sin`, `doy_cos`) cikarildiginda PR-AUC +0.0053 degisti.

> Degisim kucuk. Model mevsimsellige bagimli degil; fiziksel
> degiskenlerden (FWI, toprak nemi, topografya) ogreniyor.

## Operasyonel deger

| Izlenen alan | Yakalanan yangin | Rastgeleye gore |
|---|---|---|
| en riskli %1 | **%9.9** | 9.9x |
| en riskli %2 | **%16.9** | 8.5x |
| en riskli %5 | **%31.4** | 6.3x |
| en riskli %10 | **%48.9** | 4.9x |
| en riskli %20 | **%70.7** | 3.5x |

> PDR'deki deger onerisinin dogrudan olcumu budur. Ekipler tum ormani
> degil en riskli bolgeyi izler. Ornegin en riskli %5'lik alani izleyerek
> yanginlarin onemli bir kismini onceden kapsamak mumkundur; bu 'reaktif
> sondurmeden proaktif onden konumlanmaya gecis' iddiasinin sayisal
> karsiligidir.

## En onemli 20 oznitelik

| # | Oznitelik | Pay |
|---|---|---|
| 1 | `skin_temp` | 10.47% |
| 2 | `elev_mean` | 8.67% |
| 3 | `lon_center` | 5.62% |
| 4 | `lat_center` | 4.04% |
| 5 | `elev_range` | 3.66% |
| 6 | `bui` | 3.29% |
| 7 | `built_5x5` | 3.14% |
| 8 | `doy_sin` | 2.85% |
| 9 | `doy_cos` | 2.70% |
| 10 | `frac_ciplak` | 2.64% |
| 11 | `frac_yapili` | 2.58% |
| 12 | `ffmc` | 2.49% |
| 13 | `dmc` | 2.30% |
| 14 | `crop_5x5` | 2.28% |
| 15 | `elev_std` | 2.24% |
| 16 | `soil_w4_anom` | 2.23% |
| 17 | `lai_high` | 1.95% |
| 18 | `frac_agac` | 1.89% |
| 19 | `soil_w4` | 1.89% |
| 20 | `soil_w2_anom` | 1.80% |

## Grup bazinda katki

| Grup | Pay |
|---|---|
| toprak_bitki | 26.9% |
| topografya | 20.0% |
| konum_zaman | 15.2% |
| yangin_indeksleri | 14.2% |
| arazi_ortusu | 12.7% |
| insan_faktoru | 11.0% |

> `crop_edge` ve `dist_crop_km` yuksek ciktiysa PDR'de 'tarim-orman
> gecis kusaginda onleyici konumlanma' onerisi yapabilirsiniz.

## Cikti

- `_reports/feature_importance.csv`
- `data_derived/model_test_predictions.parquet` — test tahminleri, harita uzerinde gorsellestirilebilir