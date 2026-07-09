from pathlib import Path
import pandas as pd

# Minimal first-step asset builder.
# Put this script in your project root and run: python src/build_assets.py

RAW_CSV = Path("data/taylor_swift_spotify.csv")
OUT_DIR = Path("data")
DOCS_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RAW_CSV)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
for col in ["id", "uri"]:
    if col in df.columns:
        df = df.drop(columns=[col])

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["duration_min"] = df["duration_ms"] / 60000

audio_features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

def get_base_album(name: str) -> str:
    name = str(name).strip()
    name = name.replace(" (Taylor's Version)", " TV")
    name = name.replace(" [Deluxe]", "").replace(" (Deluxe)", "")
    name = name.replace(" (Deluxe Edition)", "").replace(" (Deluxe Package)", "")
    name = name.replace(" (Platinum Edition)", "").replace(" (International Version)", "")
    name = name.replace(" (The Til Dawn Edition)", "").replace(" (3am Edition)", "")
    name = name.replace(": THE ANTHOLOGY", "")
    mapping = {
        "folklore: the long pond studio sessions (from the Disney+ special) [deluxe edition]": "folklore (long pond)",
        "reputation Stadium Tour Surprise Song Playlist": "reputation (live)",
        "Speak Now World Tour Live": "Speak Now (live)",
        "Live From Clear Channel Stripped 2008": "Taylor Swift (live)",
        "1989 (Taylor's Version)": "1989 TV",
        "1989 (Taylor's Version) [Deluxe]": "1989 TV",
        "Fearless (Taylor's Version)": "Fearless TV",
        "Red (Taylor's Version)": "Red TV",
        "Speak Now (Taylor's Version)": "Speak Now TV",
    }
    return mapping.get(name, name)

def assign_era(year):
    if pd.isna(year):
        return "Unknown"
    year = int(year)
    if year <= 2008:
        return "Country (2006-2008)"
    if year <= 2012:
        return "Country-Pop (2010-2012)"
    if year <= 2017:
        return "Pop (2014-2017)"
    if year <= 2020:
        return "Indie/Folk (2020)"
    return "Pop Rock (2022-)"

def mood_tags(row):
    tags = []
    if row["acousticness"] >= 0.55: tags.append("acoustic")
    if row["acousticness"] <= 0.18: tags.append("electronic/produced")
    if row["energy"] >= 0.65: tags.append("high-energy")
    if row["energy"] <= 0.40: tags.append("low-energy")
    if row["danceability"] >= 0.65: tags.append("danceable")
    if row["valence"] >= 0.55: tags.append("bright")
    if row["valence"] <= 0.30: tags.append("melancholic")
    if row["tempo"] >= 130: tags.append("fast-tempo")
    if row["tempo"] <= 90: tags.append("slow-tempo")
    return ", ".join(tags) if tags else "balanced"

df["album_base"] = df["album"].apply(get_base_album)
df["era"] = df["release_year"].apply(assign_era)
df["is_taylor_version"] = df["album_base"].str.contains(r"\bTV\b", regex=True) | df["name"].str.contains("Taylor's Version|Taylor’s Version", regex=True)
df["mood_tags"] = df.apply(mood_tags, axis=1)

clean_cols = [
    "name", "album", "album_base", "release_date", "release_year", "era", "track_number",
    *audio_features, "popularity", "duration_min", "is_taylor_version", "mood_tags"
]
df[clean_cols].to_csv(OUT_DIR / "taylor_swift_spotify_clean.csv", index=False)

profiles = []
for _, row in df.iterrows():
    text = f"""Song: {row['name']}
Album: {row['album']}
Base album: {row['album_base']}
Release year: {row['release_year']}
Era: {row['era']}
Popularity: {row['popularity']}
Audio profile: acousticness={row['acousticness']:.3f}, danceability={row['danceability']:.3f}, energy={row['energy']:.3f}, valence={row['valence']:.3f}, tempo={row['tempo']:.1f}, loudness={row['loudness']:.2f}.
Mood tags: {row['mood_tags']}
Taylor's Version: {bool(row['is_taylor_version'])}
"""
    profiles.append({
        "doc_id": f"song::{row['album_base']}::{row['track_number']}::{row['name']}",
        "name": row["name"],
        "album": row["album"],
        "album_base": row["album_base"],
        "era": row["era"],
        "mood_tags": row["mood_tags"],
        "profile_text": text,
    })

pd.DataFrame(profiles).to_csv(OUT_DIR / "song_profiles.csv", index=False)

with open(DOCS_DIR / "song_profiles.md", "w", encoding="utf-8") as f:
    f.write("# Song Profiles\n\n")
    for p in profiles:
        f.write(f"## {p['name']}\n\n{p['profile_text']}\n---\n\n")

album_summary = df.groupby("album_base")[["acousticness", "danceability", "energy", "valence", "popularity"]].mean().round(3)
album_summary.to_csv(OUT_DIR / "album_summary.csv")

with open(DOCS_DIR / "dataset_overview.md", "w", encoding="utf-8") as f:
    f.write("# Dataset Overview\n\n")
    f.write(f"Songs: {len(df)}\n\n")
    f.write(f"Base albums: {df['album_base'].nunique()}\n\n")
    f.write(f"Release years: {int(df['release_year'].min())} to {int(df['release_year'].max())}\n\n")
    f.write("## Album Summary\n\n")
    f.write(album_summary.to_markdown())

print("Done. Generated clean CSV, song profiles, and docs.")
