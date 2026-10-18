from getpass import getpass
from io import StringIO
from pathlib import Path
from readline import redisplay

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_columns", 30)
print("Setup complete.")

MAP_KEY = getpass("Paste your FIRMS MAP_KEY, then press Enter: ").strip()

if not MAP_KEY:
    raise ValueError("A FIRMS MAP_KEY is required. Use the registration link in Lab 0.")

print("MAP_KEY loaded securely (value hidden).")

REGIONS = {
    "Los Angeles area": (-118.95, 33.60, -117.60, 34.85),
    "Southern California": (-121.0, 32.0, -114.0, 36.0),
    "California": (-124.5, 32.0, -114.0, 42.0),
}

#REGION_NAME = "Southern California"
REGION_NAME = "Los Angeles area"
BBOX = REGIONS[REGION_NAME]               # west, south, east, north
SOURCE = "VIIRS_NOAA21_NRT"
#DAYS = 1 
DAYS = 5                                 # valid recent range: 1 through 5

print("Region:", REGION_NAME)
print("Bounding box (W, S, E, N):", BBOX)
print("Source:", SOURCE)

def get_firms_data(map_key, bbox, source="VIIRS_NOAA21_NRT", days=1, date=None):
    # Retrieve NASA FIRMS Area API CSV data as a pandas DataFrame.
    if not 1 <= int(days) <= 5:
        raise ValueError("days must be between 1 and 5 for the Area API.")

    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be (west, south, east, north) in valid ranges.")

    area = f"{west},{south},{east},{north}"
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{source}/{area}/{int(days)}"
    )
    if date is not None:
        url += f"/{date}"

    response = requests.get(url, timeout=45)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        return pd.DataFrame()
    if text.lower().startswith(("error", "invalid", "map key")):
        raise RuntimeError(f"FIRMS returned an error: {text[:300]}")

    return pd.read_csv(StringIO(text))

print("Requesting recent FIRMS data...")

try:
    fires_raw = get_firms_data(
        MAP_KEY,
        BBOX,
        source=SOURCE,
        days=DAYS,
    )
    print(f"Received {len(fires_raw):,} detections.")
except requests.RequestException as exc:
    print("Network/API request failed:", exc)
    print("Check your connection, key, source name, and bounding box.")
    raise

fires_raw.head()

print("Shape:", fires_raw.shape)
print("Columns:")
print(list(fires_raw.columns))

useful_columns = [
    c for c in [
        "latitude", "longitude", "acq_date", "acq_time",
        "satellite", "instrument", "confidence", "frp", "daynight"
    ]
    if c in fires_raw.columns
]

fires_raw[useful_columns].head(10)

def prepare_firms_data(df):
    # Clean numeric fields and add UTC and California timestamps.
    result = df.copy()
    if result.empty:
        return result

    for column in ["latitude", "longitude", "frp"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    time_text = result["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    result["acq_time_hhmm"] = time_text
    result["timestamp_utc"] = pd.to_datetime(
        result["acq_date"].astype(str) + " " + time_text,
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )
    result["timestamp_california"] = result["timestamp_utc"].dt.tz_convert(
        "America/Los_Angeles"
    )
    return result


fires = prepare_firms_data(fires_raw)
fires[["acq_date", "acq_time_hhmm", "timestamp_utc", "timestamp_california"]].head()

if "confidence" in fires.columns:
    confidence_text = fires["confidence"].astype(str).str.lower()
    fires_filtered = fires[confidence_text.isin(["n", "h"])].copy()
else:
    fires_filtered = fires.copy()

print(f"All detections: {len(fires):,}")
print(f"Nominal/high confidence: {len(fires_filtered):,}")
fires_filtered[useful_columns].head(10)

if fires_filtered.empty:
    print("No matching detections to plot. Try DAYS=5 or a larger region.")
else:
    frp = fires_filtered["frp"].fillna(0).clip(lower=0)
    marker_sizes = 15 + 3 * np.sqrt(frp)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        fires_filtered["longitude"],
        fires_filtered["latitude"],
        s=marker_sizes,
        c=frp,
        cmap="hot_r",
        alpha=0.70,
        edgecolors="black",
        linewidths=0.25,
    )
    plt.colorbar(scatter, label="FRP (MW)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"NASA FIRMS Detections — {REGION_NAME}")
    plt.grid(alpha=0.25)
    plt.show()

def haversine_km(lat1, lon1, lat2, lon2):
    # Vectorized great-circle distance in kilometers.
    radius_km = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


STATION_NAME = "Wildfire Guardian Station"
STATION_LAT = 34.1478
STATION_LON = -118.1445

if not fires_filtered.empty:
    fires_filtered["distance_km"] = haversine_km(
        STATION_LAT,
        STATION_LON,
        fires_filtered["latitude"],
        fires_filtered["longitude"],
    )

nearest_columns = [
    c for c in ["latitude", "longitude", "timestamp_california", "confidence", "frp", "distance_km"]
    if c in fires_filtered.columns
]
fires_filtered.sort_values("distance_km")[nearest_columns].head(10) if not fires_filtered.empty else fires_filtered

#ALERT_RADIUS_KM = 50
ALERT_RADIUS_KM = 25

if fires_filtered.empty:
    nearby = fires_filtered.copy()
    print("No filtered detections were returned.")
else:
    nearby = fires_filtered[fires_filtered["distance_km"] <= ALERT_RADIUS_KM].copy()
    if nearby.empty:
        print(f"No FIRMS detections within {ALERT_RADIUS_KM} km of the station.")
    else:
        print(f"RESEARCH ALERT: {len(nearby)} detection(s) within {ALERT_RADIUS_KM} km.")
        # display(nearby.sort_values("distance_km")[nearest_columns])
        display(nearby.sort_values("distance_km")[nearest_columns])

def confidence_color(value):
    return {"h": "darkred", "n": "orange", "l": "gold"}.get(str(value).lower(), "red")


def build_fire_map(df, station_name, station_lat, station_lon, alert_radius_km=50):
    m = folium.Map(
        location=[station_lat, station_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.Marker(
        [station_lat, station_lon],
        tooltip=station_name,
        popup=f"<b>{station_name}</b>",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    folium.Circle(
        [station_lat, station_lon],
        radius=alert_radius_km * 1000,
        color="blue",
        weight=2,
        fill=False,
        tooltip=f"{alert_radius_km} km research-alert radius",
    ).add_to(m)

    for _, row in df.iterrows():
        confidence = row.get("confidence", "unknown")
        timestamp = row.get("timestamp_california", "unknown")
        frp = row.get("frp", np.nan)
        distance = row.get("distance_km", np.nan)
        satellite = row.get("satellite", "unknown")

        popup = folium.Popup(
            (
                "<b>NASA FIRMS thermal detection</b><br>"
                f"Satellite: {satellite}<br>"
                f"California time: {timestamp}<br>"
                f"Confidence: {confidence}<br>"
                f"FRP: {frp:.1f} MW<br>"
                f"Distance: {distance:.1f} km"
            ),
            max_width=360,
        )

        radius = float(np.clip(4 + np.sqrt(max(float(frp), 0)), 4, 14))
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=radius,
            color=confidence_color(confidence),
            weight=1,
            fill=True,
            fill_opacity=0.70,
            popup=popup,
            tooltip=f"FRP {frp:.1f} MW",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


fire_map = build_fire_map(
    fires_filtered,
    STATION_NAME,
    STATION_LAT,
    STATION_LON,
    ALERT_RADIUS_KM,
)
fire_map

OUTPUT_HTML = Path("wildfire_guardian_firms_map.html")
fire_map.save(OUTPUT_HTML)
print(f"Saved interactive webpage: {OUTPUT_HTML.resolve()}")


SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
frames = []

for product in SOURCES:
    frame = get_firms_data(MAP_KEY, BBOX, source=product, days=DAYS)
    frame["source_product"] = product
    frames.append(frame)

fires_both = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(fires_both["source_product"].value_counts() if not fires_both.empty else "No detections returned.")
fires_both.head()

historical = get_firms_data(
    MAP_KEY,
    BBOX,
    source="VIIRS_NOAA21_NRT",
    days=3,
    date="2026-08-20",
)
historical.head()

if not fires.empty and "confidence" in fires.columns:
    fires["confidence"].astype(str).str.lower().value_counts().plot.bar(
        color=["gold", "orange", "darkred"]
    )
    plt.xlabel("Confidence code")
    plt.ylabel("Number of detections")
    plt.title("FIRMS Detection Count by Confidence")
    plt.show()


if not fires_filtered.empty:
    fires_filtered["frp"].dropna().plot.hist(bins=20, edgecolor="black")
    plt.xlabel("FRP (MW)")
    plt.ylabel("Number of detections")
    plt.title("Distribution of Fire Radiative Power")
    plt.show()




