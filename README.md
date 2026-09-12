# ORION-TR — Türkiye Orman Yangını Risk Veri Seti
Ahmet tarafından test değişikliği
**Sürüm:** 1.0 · **Tarih:** 11 Eylül 2026 · **Kapsam:** Türkiye, 2016–2025

\---
\*\*Veri setini indir:\*\* \[ORION-TR v1.0 — orion\_tr\_dataset.parquet](https://github.com/eminehatundincer/orion-tr-wildfire-dataset/releases/tag/v1.0)

## 1\. Bu nedir?

Türkiye'de orman yangını riskini **önceden** tahmin etmek için sıfırdan üretilmiş,
etiketli bir makine öğrenmesi veri setidir. Hazır indirilmiş bir paket değildir;
altı açık veri kaynağından çekilip temizlenerek, doğrulanarak oluşturulmuştur.

**Bir satır = bir 1 km'lik hücre, bir gün.**

**Etiket (`label`):** O hücrede, sonraki 1–3 gün içinde orman yangını
tutuşması olup olmadığı (0 / 1).

Yani model şunu öğrenir: *bugünkü hava, kuraklık, bitki örtüsü ve arazi
koşullarına bakarak, önümüzdeki 72 saatte burada yangın çıkma olasılığı nedir?*

|||
|-|-|
|Satır sayısı|900.192|
|Sütun sayısı|54|
|Dosya boyutu|103 MB (parquet)|
|Grid|0,01° (\~1 km), EPSG:4326|
|Dönem|2016-01-01 – 2025-12-31 (3.653 gün)|
|Pozitif oran|%6,25 (56.262 pozitif satır)|
|Coğrafya|Türkiye resmî idari sınırı|

\---

## 2\. Hızlı başlangıç

```python
import pandas as pd

df = pd.read\\\_parquet("orion\\\_tr\\\_dataset.parquet")

# Zamansal bölme hazır: rastgele bölmeyin!
train = df\\\[df.split == "train"]   # 2016–2022
val   = df\\\[df.split == "val"]     # 2023
test  = df\\\[df.split == "test"]    # 2024–2025

# Modelde kullanılmayacak sütunlar
drop = \\\["cell\\\_id", "day\\\_idx", "date", "label", "split",
        "yil", "ay", "fwi\\\_grid\\\_shift", "era5\\\_grid\\\_shift"]
features = \\\[c for c in df.columns if c not in drop]
```

Parquet okumak için `pip install pandas pyarrow` yeterlidir.

**Önemli:** Bölmeyi rastgele yapmayın. Aynı yangının komşu günleri hem eğitimde
hem testte olursa model kopya çeker ve başarı sahte çıkar. `split` sütunu
zaten zamansal olarak ayrılmıştır.

**Önemli 2:** Değerlendirmede **accuracy kullanmayın**. Pozitif oranı %6
olduğu için "hiç yangın yok" diyen model %94 doğruluk alır ve hiçbir işe
yaramaz. **PR-AUC, recall** ve "en riskli %N alanda yakalanan yangın oranı"
kullanın.

\---

## 3\. Sütunlar

Tam açıklamalar `column\\\_dictionary.csv` dosyasındadır. Özet:

|Grup|Sütun|İçerik|
|-|-|-|
|Anahtar|6|`cell\\\_id`, `day\\\_idx`, `date`, `yil`, `ay`, `split`|
|Etiket|1|`label`|
|Konum \& zaman|4|Enlem, boylam, yılın gününün dairesel kodlaması|
|Arazi örtüsü|7|Hücredeki ağaç / maki / otlak / tarım / yapılı / çıplak oranları|
|Topografya|7|Yükseklik, eğim, bakı, arazi pürüzlülüğü|
|İnsan faktörü|7|Yerleşime, tarıma, suya mesafe; çevre yoğunlukları; orman-tarım sınırı|
|Yangın indeksleri|9|FWI, FFMC, DMC, DC, ISI, BUI + FWI'nin 7/30 günlük gecikmeleri|
|Toprak \& bitki|11|4 katman toprak nemi + anomalileri, LAI, yüzey sıcaklığı|

Yangın indeksleri Kanada FWI sisteminin standart bileşenleridir — Algerian
Forest Fires veri setindeki sütunların Türkiye karşılığı.

\---

## 4\. Veriler nereden geldi?

|Kaynak|Ürün|Kattığı|Lisans|
|-|-|-|-|
|**NASA FIRMS**|VIIRS\_SNPP\_SP, MODIS\_SP|Yangın etiketi|NASA Open Data|
|**geoBoundaries**|gbOpen TUR ADM0|Türkiye sınırı|CC BY 4.0|
|**ESA WorldCover**|10 m, 2021 v200|Arazi örtüsü|CC BY 4.0|
|**Copernicus DEM**|GLO-30 (\~30 m)|Topografya|Copernicus|
|**Copernicus EMS**|`cems-fire-historical-v1`, sistem 4\_1|FWI ailesi|Copernicus|
|**Copernicus C3S**|`reanalysis-era5-land-monthly-means`|Toprak nemi, LAI|CC BY 4.0|

Hepsi ücretsiz ve açık. Atıf metinleri `data\\\_raw/\\\*/source.json` altındadır.

\---

## 5\. Etiket nasıl temizlendi? (en önemli kısım)

Ham uydu verisi doğrudan kullanılamazdı. Dört aşamalı ayıklama yapıldı:

|Aşama|Kalan kayıt|Ne ayıklandı|
|-|-|-|
|Ham FIRMS indirmesi|1.699.297|—|
|`type` filtresi|1.002.276|**%40,6'sı sanayi tesisi ve gaz flaresi**|
|Türkiye sınırına kırpma|614.720|**%38,7'si Suriye/Irak sınırlarındaydı**|
|Arazi örtüsü maskesi|77.726|**%67'si tarımsal anız yakmaydı**|
|Tutuşma / devam ayrımı|**18.764 tutuşma**|Devam eden yangın günleri|

**Doğrulama:** Temizlik sonrası yıllık tutuşma sayıları 1.147–2.814 aralığına
oturdu; bu, OGM'nin yayımladığı yıllık orman yangını sayılarıyla aynı mertebede.
Ayrıca 2021 (Manavgat, Marmaris) ve 2025 (İzmir, Çanakkale) yılları veride
belirgin biçimde öne çıkıyor.

**Neden önemli:** Bu temizlik yapılmasaydı model Aliağa rafinerisini ve
Çukurova'daki pamuk anızını "orman yangını" sanarak öğrenecekti.

Ayrıca etikette **yalnızca VIIRS\_SNPP\_SP ve MODIS\_SP** kullanıldı. VIIRS\_NOAA20
2018 Nisan'da devreye girdiği için seriye eklenirse yapay bir artış yaratırdı.

\---

## 6\. Negatif örnekleme

Orman maskesinde 259.288 hücre × 3.653 gün = \~950 milyon aday var; hepsi
kullanılamaz. Yöntem:

* **Pozitifler:** Her tutuşma için t−1, t−2, t−3 satırları → 56.262 satır
* **Negatifler:** Orman hücrelerinden rastgele, 1:15 oranında → 843.930 satır
* **Dışlama:** Bir yangının 3×3 komşuluğunda ve ±7 gün içindeki hücre-günler
negatif olarak seçilmedi. Orası "yangın yok" değil, "emin değiliz" bölgesi.

\---

## 7\. Doğrulandı mı?

Evet. LightGBM ile temel model eğitildi (hiperparametre araması yapılmadan),
2024–2025 üzerinde test edildi:

|Ölçüt|Değer|Rastgele|
|-|-|-|
|PR-AUC|**0,4135**|0,0759|
|ROC-AUC|**0,8687**|0,5000|

**Operasyonel değer:**

|İzlenen alan|Yakalanan yangın|
|-|-|
|En riskli %5|%31,4|
|En riskli %10|%48,9|
|En riskli %20|%70,7|

**Genelleme testleri:**

* Koordinat sütunları çıkarıldığında PR-AUC yalnızca 0,018 düştü → model
konumu ezberlemiyor.
* Batı Türkiye'de eğitilip **hiç görmediği** Doğu Anadolu'da test edildiğinde
rastgeleye göre 5,7 kat kazanım → öğrendiği şey bölgesel ezber değil,
yangın fiziği.
* Zaman kodlaması çıkarıldığında performans neredeyse değişmedi → model
sadece "yazın yangın çıkar" demeyi öğrenmemiş.

En önemli değişken grupları: toprak \& bitki (%26,9), topografya (%20,0),
yangın indeksleri (%14,2).

\---

## 8\. Bilinen sınırlar

Bunları bilerek kullanın, raporda da yazın.

1. **Şiddet tahmin edilmiyor.** Sistem *tutuşma olasılığını* tahmin eder.
Tüm tutuşmaların %70,7'si en riskli %20'de öngörülürken, en şiddetli 25
yangında bu oran %48'e düşüyor. Bir yangının felakete dönüşmesi rüzgâr,
topografik kanal etkisi ve müdahale gecikmesine bağlı; bunlar veri setinde
günlük çözünürlükte yok.
2. **Günlük rüzgâr yok.** ERA5 günlük indirmesi CDS kuyruğunda tıkandığı için
yapılamadı. Rüzgâr bilgisi yalnızca ISI indeksi içinde dolaylı olarak var.
En değerli iyileştirme bu olur.
3. **LAI iklimseldir.** ERA5-Land yaprak alan indeksini sabit aylık ortalama
olarak kullanır, yıllar arası değişmez. Bitki stresi bilgisi taşımaz.
Gerçek stres için MODIS/Sentinel NDVI gerekir.
4. **Toprak nemi aylık çözünürlüktedir.** Günlük yangın dinamiğini FWI ailesi
taşır.
5. **İnsan faktörü dolaylıdır.** Gerçek nüfus yoğunluğu (GHSL) ve yol ağı (OSM)
yerine yapılı alana mesafe kullanıldı.
6. **Etiket uydu tespitine dayanır.** Yoğun bulut altındaki veya çok küçük
yangınlar görülmemiş olabilir. "Tespit yok" ≠ "yangın yok".
7. **Maki alt temsil edilmiştir.** WorldCover Akdeniz makisinin çoğunu ağaç
örtüsü veya otlak sayıyor. CORINE'in `323 Sclerophyllous vegetation` sınıfı
eklenmeli.
8. **Topografyada %3,51 eksik** var (DEM karosu olmayan kıyı hücreleri).
LightGBM/XGBoost bunu doğal işler, doldurmaya gerek yok.

\---

## 9\. Bundan sonra ne yapılabilir?

Öncelik sırasıyla:

1. **Günlük rüzgâr ekle.** ERA5'ten `10m\\\_u/v\\\_component\\\_of\\\_wind` ve varsa rüzgâr
hamlesi. Şiddet tahminindeki en büyük eksik bu.
2. **NDVI ekle.** MODIS MOD13Q1 veya Sentinel-2. LAI'nin yapamadığı şeyi yapar:
yıllar arası bitki stresini ölçer.
3. **Ayrı bir şiddet modeli.** "Yangın çıkarsa ne kadar büyür" sorusu için,
etiket olarak FRP veya yanan alan kullanılabilir.
4. **CORINE arazi örtüsü.** Maki sınıfını düzgün ayırır.
5. **Hiperparametre optimizasyonu.** Temel modelde hiç yapılmadı, buradan
kolay kazanım var.
6. **Gerçek nüfus ve yol verisi.** GHSL + OpenStreetMap Türkiye ekstresi.

\---

## 10\. Üretim zinciri

Veri seti 17 adımda, tekrar çalıştırılabilir betiklerle üretildi. Her adım
kendi QA raporunu, SHA-256 manifestini ve eksik veri kaydını yazar.

```
 1. download\\\_firms.py          FIRMS arşivi (2.028 pencere dosyası)
 2. build\\\_firms\\\_events.py      temizlik + type filtresi
 3. clip\\\_firms\\\_turkey.py       Türkiye sınırına kırpma
 4. assign\\\_landcover.py        arazi örtüsü sınıfı atama
 5. build\\\_grid.py              1 km grid + örtü oranları
 6. build\\\_fire\\\_celldays.py     tutuşma / devam ayrımı
 7. build\\\_sample\\\_frame.py      pozitif + negatif örnekleme
 8. build\\\_terrain.py           Copernicus DEM → topografya
 9. build\\\_human\\\_features.py    insan faktörü türetme
10. download\\\_fwi.py            CEMS FWI indirme
11. join\\\_fwi.py                FWI + gecikmeli değişkenler
12. download\\\_era5\\\_monthly.py   ERA5-Land aylık indirme
13. join\\\_era5\\\_monthly.py       toprak nemi + LAI + anomaliler
14. build\\\_dataset.py           nihai birleştirme
15. train\\\_baseline.py          temel model
16. validate\\\_model.py          genelleme testleri + vaka incelemesi
```

Betikler `scripts/` klasöründedir. Yeniden çalıştırmak için `config/secrets.env`
içine kendi API anahtarlarınızı koymanız gerekir (NASA FIRMS ve Copernicus CDS/EWDS
— hepsi ücretsiz).

\---

## 11\. Paketin içeriği

|Dosya|Ne işe yarar|
|-|-|
|`orion\\\_tr\\\_dataset.parquet`|**Veri setinin kendisi**|
|`column\\\_dictionary.csv`|Her sütunun Türkçe açıklaması|
|`dataset\\\_card.md`|Ayrıntılı veri kartı|
|`baseline\\\_model\\\_qa.md`|Temel model sonuçları|
|`validation\\\_qa.md`|Genelleme testleri ve vaka incelemesi|
|`feature\\\_importance.csv`|Değişken önem sıralaması|
|`case\\\_studies.csv`|Gerçek yangınlar ve önceden verilen risk skorları|
|`model\\\_test\\\_predictions.parquet`|Test tahminleri (haritada gösterilebilir)|
|`scripts/`|Üretim betikleri|

\---
## Yapılacaklar (v2)

\- Günlük rüzgar verisi eklenecek (ERA5)


*Bu veri seti açık kaynaklardan üretilmiştir. Kaynak lisanslarına ve atıf
şartlarına uyunuz.*

