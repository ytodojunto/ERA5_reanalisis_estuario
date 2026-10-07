"""
Descarga reanálisis ERA5 horario (presión a nivel del mar, viento a 10 m) en
puntos del Río de la Plata y la plataforma bonaerense, desde la API histórica
de Open-Meteo (models=era5, sin clave). Sirve para backtestear el modelo de
residuo de marea con 10 años de meteorología en vez de las pocas semanas de
viento pronosticado del SMN.

Salida: data/era5_<punto>.csv con columnas
    time_utc, pressure_msl_hpa, wind_speed_ms, wind_dir_deg   (dir = de dónde viene)
Es idempotente: baja por año y por punto, salta los años ya completos y siempre
refresca los últimos 45 días (ERA5 final llega con ~5 días de atraso).

Atribución (CC BY 4.0): datos de Open-Meteo.com; contiene información de
Copernicus Climate Change Service (ERA5). La API gratuita de Open-Meteo es para
uso NO comercial; si esto pasa a producto pago, bajar de Copernicus CDS directo.

NOTA: el formato de respuesta está escrito según la documentación de la API;
la primera corrida real valida las claves/unidades y falla con mensaje claro
si algo difiere.
"""
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path(__file__).parent / "data"
INICIO = date(2015, 11, 1)          # arranque del historial de CARP
ATRASO_DIAS = 6                      # ERA5 no está listo para los últimos días
REFRESCO_DIAS = 45
PAUSA_S = 1.5

# Puntos: estuario (donde se mide) + boca/plataforma (donde se genera la sudestada)
PUNTOS = {
    "norden": (-34.629, -57.927),
    "buenos_aires": (-34.561, -58.399),
    "san_clemente": (-36.355, -56.715),
    "punta_del_este": (-34.950, -54.950),
    "mar_del_plata": (-38.000, -57.500),
    "plataforma_norte": (-36.000, -54.000),
    "plataforma_sur": (-39.000, -56.000),
    "bahia_blanca": (-39.300, -61.500),
}
COLS = ["time_utc", "pressure_msl_hpa", "wind_speed_ms", "wind_dir_deg"]


def pedir(lat, lon, d0: date, d1: date) -> list[list]:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": d0.isoformat(), "end_date": d1.isoformat(),
        "hourly": "pressure_msl,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "timezone": "UTC", "models": "era5",
    }
    for intento in range(6):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429:
            espera = 60 * (intento + 1)
            print(f"    429 (límite de uso), espero {espera}s")
            time.sleep(espera)
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError("Open-Meteo siguió respondiendo 429 tras 6 intentos")
    j = r.json()
    if "hourly" not in j:
        raise RuntimeError(f"respuesta sin 'hourly': {str(j)[:200]}")
    h, u = j["hourly"], j.get("hourly_units", {})
    for k in ("time", "pressure_msl", "wind_speed_10m", "wind_direction_10m"):
        if k not in h:
            raise RuntimeError(f"falta la variable {k!r} en la respuesta (claves: {list(h)})")
    if u.get("pressure_msl") != "hPa" or u.get("wind_speed_10m") != "m/s":
        raise RuntimeError(f"unidades inesperadas: {u}")
    n = len(h["time"])
    if not all(len(h[k]) == n for k in h):
        raise RuntimeError("arreglos de distinto largo")
    return [[h["time"][i], h["pressure_msl"][i], h["wind_speed_10m"][i], h["wind_direction_10m"][i]] for i in range(n)]


def leer(path: Path) -> dict[str, list]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {fila[0]: fila for fila in list(csv.reader(f))[1:] if fila}


def escribir(path: Path, filas: dict[str, list]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for k in sorted(filas):
            w.writerow(filas[k])


def rango_completo(filas: dict, d0: date, d1: date) -> bool:
    """¿Ya están (casi) todas las horas entre d0 y d1 (inclusive)?"""
    a, b = f"{d0.isoformat()}T00:00", f"{d1.isoformat()}T23:00"
    n = sum(1 for k in filas if a <= k <= b and filas[k][1] not in ("", "None"))
    return n >= 0.98 * 24 * ((d1 - d0).days + 1)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    fin = date.today() - timedelta(days=ATRASO_DIAS)
    refresco_desde = fin - timedelta(days=REFRESCO_DIAS)
    errores = 0
    for nombre, (lat, lon) in PUNTOS.items():
        path = DATA_DIR / f"era5_{nombre}.csv"
        filas = leer(path)
        print(f"== {nombre} ({lat}, {lon}) — {len(filas)} filas existentes")
        for anio in range(INICIO.year, fin.year + 1):
            d0 = max(INICIO, date(anio, 1, 1))
            d1 = min(fin, date(anio, 12, 31))
            if rango_completo(filas, d0, d1):
                if d1 < refresco_desde:
                    continue                      # año ya completo y fuera de la ventana de refresco
                d0 = max(d0, refresco_desde)      # solo se refrescan los últimos días
            try:
                nuevas = pedir(lat, lon, d0, d1)
            except Exception as e:
                print(f"   {anio}: ERROR {e.__class__.__name__}: {e}")
                errores += 1
                continue
            ok = 0
            for t, p, w, d in nuevas:
                if p is None or w is None or d is None:
                    continue
                filas[t] = [t, p, w, d]
                ok += 1
            print(f"   {anio}: {ok} horas ({d0} → {d1})")
            escribir(path, filas)  # guarda de a un año, por si se corta
            time.sleep(PAUSA_S)
        escribir(path, filas)
    print(f"\nTerminado con {errores} errores")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
