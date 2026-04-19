from huggingface_hub import login, hf_hub_download, HfApi
import pandas as pd
import numpy as np
import os

login(token=os.environ.get("HF_TOKEN"))

# ─── Find all shards ───
api = HfApi()

cats_path = hf_hub_download(
    repo_id="foursquare/fsq-os-places",
    filename="release/dt=2026-04-14/categories/parquet/categories_000000.parquet",
    repo_type="dataset",
)

# ─── Build allowed category IDs ───
ALLOWED_L1 = [
    "Dining and Drinking",
    "Nightlife Spot",
    "Sports and Recreation",
    "Arts and Entertainment",
    "Retail",
]
ALLOWED_L2 = [
    "Library",
    "Park",
]

cats = pd.read_parquet(cats_path)

allowed_ids = set(
    cats[
        cats["level1_category_name"].isin(ALLOWED_L1)
        | cats["level2_category_name"].isin(ALLOWED_L2)
    ]["category_id"].tolist()
)

all_files = api.list_repo_tree(
    "foursquare/fsq-os-places",
    repo_type="dataset",
    path_in_repo="release/dt=2026-04-14/places/parquet",
)
shard_files = [f.path for f in all_files if f.path.endswith(".parquet")]
print(f"Found {len(shard_files)} shards\n")

# ─── Process each shard and save individually ───
os.makedirs("output/shards", exist_ok=True)
total_rows = 0

for i, filepath in enumerate(shard_files):
    try:
        path = hf_hub_download(
            repo_id="foursquare/fsq-os-places",
            filename=filepath,
            repo_type="dataset",
        )
        chunk = pd.read_parquet(path, columns=[
            "fsq_place_id", "name", "latitude", "longitude",
            "locality", "region", "postcode", "country",
            "date_created", "date_closed", "date_refreshed",
            "fsq_category_ids", "fsq_category_labels",
            "website", "tel", "email",
            "facebook_id", "instagram", "twitter",
            "unresolved_flags",
        ])

        # Clean
        chunk = chunk[chunk["date_closed"].isna()]
        chunk = chunk[chunk["unresolved_flags"].isna()]
        chunk = chunk[chunk["fsq_category_labels"].apply(
            lambda x: x is not None and isinstance(
                x, np.ndarray) and len(x) > 0
        )]
        chunk = chunk[chunk["fsq_category_ids"].apply(
            lambda x: isinstance(x, np.ndarray) and any(
                cid in allowed_ids for cid in x)
        )]
        chunk = chunk[chunk["region"].notna()]
        chunk = chunk[chunk["latitude"].notna() & chunk["longitude"].notna()]

        # Features
        chunk["primary_category"] = chunk["fsq_category_labels"].apply(
            lambda x: x[0] if isinstance(
                x, np.ndarray) and len(x) > 0 else None
        )
        chunk["digital_presence"] = (
            chunk["website"].notna().astype(int)
            + chunk["tel"].notna().astype(int)
            + chunk["email"].notna().astype(int)
            + chunk["facebook_id"].notna().astype(int)
            + chunk["instagram"].notna().astype(int)
            + chunk["twitter"].notna().astype(int)
        )
        chunk = chunk[chunk["digital_presence"] > 0]

        # Normalize region and locality to Title Case
        chunk["region"] = chunk["region"].str.strip().str.title()
        chunk["locality"] = chunk["locality"].str.strip().str.title()

        # Save this shard
        chunk.to_parquet(f"output/shards/clean_{i:04d}.parquet", index=False)
        total_rows += len(chunk)
        print(
            f"[{i+1:>3}/{len(shard_files)}] {len(chunk):>8,} rows  (total: {total_rows:,})")

        del chunk  # free memory

    except Exception as e:
        print(f"[{i+1:>3}/{len(shard_files)}] FAILED: {e}")

print(f"\nDone! {total_rows:,} total clean rows saved to output/shards/")
