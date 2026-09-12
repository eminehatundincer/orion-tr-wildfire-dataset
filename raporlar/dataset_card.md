# ORION-TR — Veri Karti

**Uretim (UTC):** 2026-09-11T08:28:59Z  
**Satir:** 900,192  |  **Sutun:** 54  |  **Boyut:** 103.3 MB

## Tanim

| Alan | Deger |
|---|---|
| Satir birimi | Bir 1 km hucre, bir gun |
| Etiket | `label` = t+1..t+3 araliginda bu hucrede orman yangini tutusmasi |
| Cografya | Turkiye, resmi idari sinir (geoBoundaries ADM0) |
| Grid | 0.01 derece (~1 km), EPSG:4326 |
| Donem | 2016-01-01 .. 2025-12-31 (3.653 gun) |
| Ornekleme | Tum pozitifler + 1:15 oraninda rastgele negatif |
| Maske | Yalniz dogal ortu orani >= %30 olan hucreler |

## Sinif dengesi ve bolme

| Bolme | Yillar | Satir | Pozitif | Oran |
|---|---|---|---|---|
| train | 2016-2022 | 629,609 | 38,534 | 6.12% |
| val | 2023-2023 | 88,021 | 3,868 | 4.39% |
| test | 2024-2025 | 182,562 | 13,860 | 7.59% |

> Bolme zamansaldir, rastgele degildir. Ayni yanginin komsu gunleri hem
> egitimde hem testte olsaydi model kopya ceker, basari sahte cikardi.
> Bolmeler arasi pozitif orani farki gercek yillar arasi degiskenliktir.
> Degerlendirmede accuracy yerine PR-AUC ve recall kullaniniz.

## Oznitelik gruplari

| Grup | Sutun | Kaynak |
|---|---|---|
| konum_zaman | 4 | Turetilmis |
| arazi_ortusu | 7 | ESA WorldCover 10 m 2021 |
| topografya | 7 | Copernicus DEM GLO-30 |
| insan_faktoru | 7 | WorldCover turevi |
| yangin_indeksleri | 9 | CEMS cems-fire-historical-v1 |
| toprak_bitki | 11 | ERA5-Land aylik |
| teknik | 2 | Turetilmis |

## Eksik veri

- En az bir ozniteligi eksik satir: **31,554** (3.51%)

| Sutun | Eksik oran |
|---|---|
| elev_mean | 3.51% |
| elev_std | 3.51% |
| slope_max | 3.51% |
| slope_mean | 3.51% |
| elev_range | 3.51% |
| aspect_sin | 3.51% |
| aspect_cos | 3.51% |

> Eksiklik DEM karosu bulunmayan kiyi hucrelerinden gelir.
> `--drop-incomplete` ile cikarilabilir veya agac tabanli modellerde
> (LightGBM, XGBoost) oldugu gibi birakilabilir; bu modeller eksik
> degeri dogal olarak isler.

## Bilinen sinirlar

1. **LAI iklimseldir.** ERA5-Land yaprak alan indeksini sabit aylik
   ortalama olarak kullanir; yillar arasi degismez. Bu nedenle bitki
   stresi bilgisi tasimaz ve anomali sutunlari atilmistir. Gercek bitki
   stresi icin MODIS/Sentinel NDVI gereklidir (v2).
2. **Toprak nemi ve LAI aylik cozunurluktedir.** Gunluk yangin dinamigi
   FWI ailesi tarafindan tasinir.
3. **Insan faktoru dolayli olcilmustur.** Gercek nufus yogunlugu (GHSL)
   ve yol agi (OSM) yerine yapili alana mesafe kullanilmistir (v2).
4. **Etiket uydu tespitine dayanir.** Yogun bulut altindaki veya cok
   kucuk yanginlar gorulmemis olabilir; 'tespit yok' 'yangin yok'
   anlamina gelmez.
5. **Sensor homojenligi icin** etikette yalniz VIIRS_SNPP_SP ve MODIS_SP
   kullanilmistir. VIIRS_NOAA20_SP 2018-04'te devreye girdigi icin
   kalibrasyon kirilimi olustururdu.
6. **Maki alt temsil edilmektedir.** WorldCover Akdeniz makisinin cogunu
   agac ortusu veya otlak olarak siniflandirir; CORINE'in `323
   Sclerophyllous vegetation` sinifi v2'de eklenmelidir.

## Kaynaklar ve lisanslar

| Kaynak | Urun | Lisans |
|---|---|---|
| NASA FIRMS | VIIRS_SNPP_SP, MODIS_SP, VIIRS_NOAA20_SP | NASA Open Data |
| geoBoundaries | gbOpen TUR ADM0 | CC BY 4.0 |
| ESA WorldCover | 10 m 2021 v200 | CC BY 4.0 |
| Copernicus DEM | GLO-30 Public | Copernicus lisansi |
| Copernicus EMS | cems-fire-historical-v1, sistem 4_1 | Copernicus lisansi |
| Copernicus C3S | reanalysis-era5-land-monthly-means | CC BY 4.0 |

Atif metinleri `data_raw/*/source.json` ve `*/metadata/` altindadir.

## Uretim zinciri

```
 1. download_firms.py         FIRMS arsivi (2.028 pencere dosyasi)
 2. build_firms_events.py     temizlik, type filtresi (1.699.297 -> 1.002.276)
 3. clip_firms_turkey.py      Turkiye sinirina kirpma (-> 614.720)
 4. assign_landcover.py       arazi ortusu sinifi atama
 5. build_grid.py             1 km grid + ortu oranlari (812.032 hucre)
 6. build_fire_celldays.py    tutusma/devam ayrimi (-> 18.764 tutusma)
 7. build_sample_frame.py     pozitif + negatif ornekleme (900.192 satir)
 8. build_terrain.py          Copernicus DEM -> topografya
 9. build_human_features.py   insan faktoru turetme
10. download_fwi.py           CEMS FWI (60 dosya)
11. join_fwi.py               FWI + gecikmeli degiskenler
12. download_era5_monthly.py  ERA5-Land aylik
13. join_era5_monthly.py      toprak nemi + LAI + anomaliler
14. build_dataset.py          nihai birlestirme
```

## Cikti

- `orion_tr_dataset.parquet` (103.3 MB)
- `_reports/column_dictionary.csv` — sutun sozlugu
- `_reports/dataset_sha256.csv` — butunluk kaydi