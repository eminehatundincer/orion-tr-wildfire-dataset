"""
ORION-TR / Adim 8 - on test
CDS (ERA5) ve EWDS (FWI) baglantilarini tek gunluk minik isteklerle sinar.

Amac: saatler surecek indirmeleri baslatmadan once su uc seyi dogrulamak
  1) anahtar dogru okunuyor mu
  2) kimlik dogrulama geciyor mu
  3) urun kullanim kosullari kabul edilmis mi (kabul edilmemisse 403 doner)
ve API semasinin guncel olup olmadigini gormek.

Anahtar hicbir yere yazilmaz, hata mesajlarinda maskelenir.

Kullanim:
    .venv\\Scripts\\python.exe scripts\\test_cds_ewds.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "config" / "secrets.env"
TMP = ROOT / "data_derived" / "_test"

TURKEY_AREA = [42.5, 25.5, 35.5, 45.0]   # kuzey, bati, guney, dogu


def load_env() -> dict[str, str]:
    if not SECRETS.exists():
        sys.exit(f"HATA: {SECRETS} yok.")
    env: dict[str, str] = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def mask(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) > 4:
            text = text.replace(s, "<GIZLI>")
    return text


def try_one(name: str, url: str, key: str, dataset: str,
            request: dict, target: Path, secrets: list[str]) -> bool:
    import cdsapi

    print(f"\n--- {name} ---")
    print(f"Urun   : {dataset}")
    print("Istek  :", {k: v for k, v in request.items() if k != "day"})
    try:
        client = cdsapi.Client(url=url, key=key, quiet=True,
                               wait_until_complete=True, delete=False)
        client.retrieve(dataset, request, str(target))
    except Exception as e:
        msg = mask(f"{type(e).__name__}: {e}", secrets)
        print(f"BASARISIZ\n{msg}")
        low = msg.lower()
        if "403" in msg or "licence" in low or "license" in low:
            print("\n>>> Bu hata genellikle URUN KOSULLARININ KABUL EDILMEDIGINI gosterir.")
            print(">>> Ilgili urun sayfasinda 'Terms of use' bolumunu onaylayin.")
        elif "401" in msg or "authoriz" in low or "authentic" in low:
            print("\n>>> Anahtar hatali veya yanlis siteye ait olabilir.")
            print(">>> CDS anahtari EWDS'te, EWDS anahtari CDS'te CALISMAZ.")
        elif "invalid" in low and "request" in low:
            print("\n>>> API semasi degismis olabilir; urun sayfasindaki")
            print(">>> 'Show API request' ciktisiyla karsilastirin.")
        return False

    if target.exists() and target.stat().st_size > 0:
        print(f"BASARILI  -> {target.name} ({target.stat().st_size / 1e6:.2f} MB)")
        return True
    print("BASARISIZ: dosya olusmadi.")
    return False


def main() -> None:
    env = load_env()
    TMP.mkdir(parents=True, exist_ok=True)

    cds_url = env.get("CDS_URL", "").strip()
    cds_key = env.get("CDS_KEY", "").strip()
    ew_url = env.get("EWDS_URL", "").strip()
    ew_key = env.get("EWDS_KEY", "").strip()
    secrets = [cds_key, ew_key]

    print("secrets.env okundu:")
    for k, v in (("CDS_URL", cds_url), ("CDS_KEY", cds_key),
                 ("EWDS_URL", ew_url), ("EWDS_KEY", ew_key)):
        state = "bos" if not v else (f"tanimli ({len(v)} karakter)"
                                     if "KEY" in k else v)
        print(f"  {k:10s}: {state}")

    if cds_key and ":" in cds_key:
        print("\nUYARI: CDS_KEY icinde ':' var. Bu ESKI bicim (uid:apikey).")
        print("Yeni cdsapi yalniz token bekler; profil sayfasindan yeni PAT alin.")

    results: dict[str, bool] = {}

    if cds_url and cds_key:
        results["CDS / ERA5"] = try_one(
            "CDS - ERA5 gunluk istatistik", cds_url, cds_key,
            "derived-era5-single-levels-daily-statistics",
            {
                "product_type": "reanalysis",
                "variable": ["2m_temperature"],
                "year": "2021",
                "month": "07",
                "day": ["28"],
                "daily_statistic": "daily_maximum",
                "time_zone": "utc+03:00",
                "frequency": "1_hourly",
                "area": TURKEY_AREA,
            },
            TMP / "test_era5.zip", secrets,
        )
    else:
        print("\nCDS atlandi: CDS_URL veya CDS_KEY bos.")

    if ew_url and ew_key:
        results["EWDS / FWI"] = try_one(
            "EWDS - CEMS yangin tehlike indeksleri", ew_url, ew_key,
            "cems-fire-historical-v1",
            {
                "product_type": "reanalysis",
                "variable": ["fire_weather_index"],
                "dataset_type": "consolidated_dataset",
                "system_version": "4_1",
                "year": "2021",
                "month": "07",
                "day": ["28"],
                "data_format": "netcdf",     
                "grid": "0.25/0.25",
                "area": TURKEY_AREA,
            },
            TMP / "test_fwi.zip", secrets,
        )
    else:
        print("\nEWDS atlandi: EWDS_URL veya EWDS_KEY bos.")

    print("\n=== SONUC ===")
    if not results:
        print("Hicbir test calistirilamadi.")
        return
    for k, v in results.items():
        print(f"  {k:14s}: {'BASARILI' if v else 'BASARISIZ'}")
    if all(results.values()):
        print("\nIkisi de hazir. Tam indirmeye gecebiliriz.")
    else:
        print("\nYukaridaki hata mesajini paylasin; anahtar gorunmuyor.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
