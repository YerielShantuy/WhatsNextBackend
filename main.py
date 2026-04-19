"""
WhatsNext — Random destination itinerary generator
Powered by Foursquare OS Places (16M POIs)

Sections
--------
1. Imports & Logging
2. Constants & Configuration
3. Time-Slot Helpers
4. Preset Configs
5. Data Models (dataclasses + Pydantic)
6. Haversine Distance
7. Proximity Clustering
8. Route Ordering (Nearest-Neighbour TSP)
9. UserAuthManager
10. HistoryManager
11. FavoritesManager
12. PlacesDataStore (Dask)
13. Itinerary Generation Engine
14. FastAPI Application & Endpoints
15. Entry Point
"""

# ═══════════════════════════════════════════════
#  1. IMPORTS & LOGGING
# ═══════════════════════════════════════════════

import os
import math
import json
import logging
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

import dask.dataframe as dd
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import login, hf_hub_download, HfApi
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("whatsnext")


# ═══════════════════════════════════════════════
#  2. CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════

# Path to the pre-filtered shards — same directory as this script
SHARD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output", "shards", "*.parquet",
)

# Proximity thresholds (km)
MAX_CLUSTER_RADIUS_KM = 3.0
SHOP_CLUSTER_RADIUS_KM = 1.0

# Hour boundaries (24-hour clock, local time passed by client)
MORNING_START = 6    # 06:00 — cafes, parks open
AFTERNOON_START = 12   # 12:00 — lunch, daytime entertainment
EVENING_START = 17   # 17:00 — bars start, dinner opens
NIGHT_START = 20   # 20:00 — nightlife in full swing
LATE_NIGHT_END = 4    # 04:00 — after this, treat as early morning

# What counts as "evening or later" for nightlife inclusion
NIGHTLIFE_THRESHOLD = EVENING_START  # 17:00+

# ── LLM Provider Config for Place Tips ─────────
# Get your free key at https://console.groq.com (no credit card needed)
# Then:  export GROQ_API_KEY=gsk_abc123...

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

if GROQ_API_KEY:
    logger.info("Groq API key found — place tips enabled")
else:
    logger.warning("GROQ_API_KEY not set — place tips will be empty. "
                   "Get a free key at https://console.groq.com")


# ═══════════════════════════════════════════════
#  2b. PLACE-TIP GENERATOR (Groq)
# ═══════════════════════════════════════════════

def _build_tip_prompt(
    stops: list,
    preset: str,
    region: str,
    country: str,
    time_slot: str,
) -> str:
    """Build the prompt string for tip generation."""
    places_list = "\n".join(
        f"{i+1}. {s.name} — Category: {s.category}, Locality: {s.locality}"
        for i, s in enumerate(stops)
    )
    return (
        f'You are a local travel guide. The user has a "{preset}" itinerary '
        f"in {region}, {country} (time: {time_slot}).\n\n"
        f"Here are the stops:\n{places_list}\n\n"
        f"For EACH stop, write a 1–2 sentence tip that helps the user know "
        f"what to expect. Include practical advice like what to try, best time "
        f"to visit, or insider knowledge. Be concise and conversational.\n\n"
        f"Respond ONLY as a JSON object mapping each place name (exactly as "
        f'written above) to its tip string. No markdown, no backticks, no preamble. '
        f'Example:\n'
        f'{{"Place Name": "Tip text here", "Another Place": "Another tip"}}'
    )


async def generate_place_tips(
    stops: list,
    preset: str,
    region: str,
    country: str,
    time_slot: str = "",
) -> Dict[str, str]:
    """
    Generate a short tip/description for each itinerary stop via Groq.

    Uses Llama 3.3 70B on Groq's free tier (~1000 req/day, no credit card).
    Sends all stops in a single request for full itinerary context.
    Returns a dict mapping place name → tip string.
    Falls back to empty dict on any error (tips are non-critical).
    """
    if not GROQ_API_KEY:
        return {}

    prompt = _build_tip_prompt(stops, preset, region, country, time_slot)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       GROQ_MODEL,
                    "max_tokens":  1000,
                    "temperature": 0.7,
                    "messages":    [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()

        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]

        # Parse JSON — strip any accidental markdown fences
        raw_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        tips = json.loads(raw_text)

        logger.info(
            f"Generated tips for {len(tips)} places via Groq/{GROQ_MODEL}")
        return tips

    except httpx.TimeoutException:
        logger.warning("Groq API timed out — returning empty tips")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Groq tips JSON: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Groq tips API error: {e}")
        return {}


# ═══════════════════════════════════════════════
#  3. TIME-SLOT HELPERS
# ═══════════════════════════════════════════════

def get_time_slot(hour: int) -> str:
    """
    Classify an hour (0–23) into a named time slot.

    morning      → 06:00–11:59
    afternoon    → 12:00–16:59
    evening      → 17:00–19:59
    night        → 20:00–03:59 (wraps past midnight)
    early_morning→ 04:00–05:59
    """
    if LATE_NIGHT_END <= hour < MORNING_START:
        return "early_morning"
    elif MORNING_START <= hour < AFTERNOON_START:
        return "morning"
    elif AFTERNOON_START <= hour < EVENING_START:
        return "afternoon"
    elif EVENING_START <= hour < NIGHT_START:
        return "evening"
    else:
        return "night"


def is_nightlife_time(hour: int) -> bool:
    """True when nightlife venues are realistically open / appropriate."""
    return hour >= NIGHTLIFE_THRESHOLD or hour < LATE_NIGHT_END


# ═══════════════════════════════════════════════
#  4. PRESET CONFIGS
# ═══════════════════════════════════════════════

NIGHTLIFE_SLOT_EVENING = [
    "Nightlife Spot",
    "Bar",
    "Lounge",
    "Arts and Entertainment",
    "Dining and Drinking",
]

NIGHTLIFE_SLOT_DAYTIME = [
    "Arts and Entertainment",
    "Movie Theater",
    "Dining and Drinking",
]


def get_hangout_rules(hour: int) -> dict:
    """
    Return the full hangout preset config for a given hour.
    Slots 1–3 are fixed; slot 4 is time-aware.
    """
    nightlife_ok = is_nightlife_time(hour)
    slot4_cascade = NIGHTLIFE_SLOT_EVENING if nightlife_ok else NIGHTLIFE_SLOT_DAYTIME
    time_slot = get_time_slot(hour)

    return {
        "display_name":      "Hangout",
        "description":       "A mix of dining, sports, arts, and nightlife",
        "category_rules": [
            {"match": "Dining and Drinking",   "pick": 1},
            {"match": "Sports and Recreation", "pick": 1},
            {"match": "Arts and Entertainment", "pick": 1},
            {"match_cascade": slot4_cascade, "pick": 1, "slot": "nightlife"},
        ],
        "total_stops":       4,
        "cluster_radius_km": MAX_CLUSTER_RADIUS_KM,
        "time_slot":         time_slot,
        "nightlife_eligible": nightlife_ok,
    }


def get_date_rules(hour: int) -> dict:
    """
    Return the full date preset config for a given hour.
    Slots 1–3 are fixed; slot 4 is time-aware (Bar/Lounge at night).
    """
    nightlife_ok = is_nightlife_time(hour)
    slot4_cascade = [
        "Bar", "Lounge"] if nightlife_ok else NIGHTLIFE_SLOT_DAYTIME
    time_slot = get_time_slot(hour)

    return {
        "display_name":      "Date",
        "description":       "A mix of dining, sports, arts, and nightlife (Bar or Lounge)",
        "category_rules": [
            {"match": "Dining and Drinking",   "pick": 1},
            {"match": "Sports and Recreation", "pick": 1},
            {"match": "Arts and Entertainment", "pick": 1},
            {"match_cascade": slot4_cascade, "pick": 1, "slot": "nightlife"},
        ],
        "total_stops":       4,
        "cluster_radius_km": MAX_CLUSTER_RADIUS_KM,
        "time_slot":         time_slot,
        "nightlife_eligible": nightlife_ok,
    }


PRESET_CONFIGS = {
    "hangout": None,  # resolved dynamically — see get_hangout_rules()
    "date":    None,  # resolved dynamically — see get_date_rules()
    "study": {
        "display_name":      "Study",
        "description":       "Quiet spots — libraries, cafes, and parks",
        "category_rules": [
            {"match": "Library", "pick": 1},
            {"match": "Cafe",    "pick": 1},
            {"match": "Park",    "pick": 1},
        ],
        "total_stops":       3,
        "cluster_radius_km": MAX_CLUSTER_RADIUS_KM,
        "time_slot":         None,
        "nightlife_eligible": False,
    },
    "shop": {
        "display_name":      "Shop",
        "description":       "4 retail spots close together for a shopping spree",
        "category_rules": [
            {"match": "Retail", "pick": 4},
        ],
        "total_stops":       4,
        "cluster_radius_km": SHOP_CLUSTER_RADIUS_KM,
        "time_slot":         None,
        "nightlife_eligible": False,
    },
}


# ═══════════════════════════════════════════════
#  5. DATA MODELS
# ═══════════════════════════════════════════════

# ── Dataclasses (internal) ─────────────────────

@dataclass
class ItineraryStop:
    name:             str
    category:         str
    latitude:         float
    longitude:        float
    address:          str = ""
    locality:         str = ""
    website:          str = ""
    tel:              str = ""
    email:            str = ""
    facebook_id:      Union[str, int, float] = ""
    instagram:        str = ""
    twitter:          str = ""
    digital_presence: int = 0
    tip:              str = ""
    order:            int = 0


@dataclass
class Itinerary:
    preset:            str
    preset_display:    str
    region:            str
    country:           str
    locality:          str = ""
    user_id:           str = ""
    stops:             list = field(default_factory=list)
    total_distance_km: float = 0.0
    ai_summary:        str = ""


# ── Pydantic models (API request bodies) ──────

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str
    role:     str = "user"


class LoginRequest(BaseModel):
    username_or_email: str
    password:          str


class RoleUpdateRequest(BaseModel):
    role: str  # "user" | "admin"


class ActiveUpdateRequest(BaseModel):
    is_active: bool


class AddFavoriteRequest(BaseModel):
    name:          str
    latitude:      float
    longitude:     float
    fsq_place_id:  str = ""
    category:      str = ""
    locality:      str = ""
    region:        str = ""
    country:       str = ""
    website:       str = ""
    tel:           str = ""
    note:          str = ""


class UpdateNoteRequest(BaseModel):
    note: str


# ═══════════════════════════════════════════════
#  6. HAVERSINE DISTANCE
# ═══════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine for numpy arrays — used for fast filtering."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ═══════════════════════════════════════════════
#  7. PROXIMITY CLUSTERING
# ═══════════════════════════════════════════════

def filter_by_cascade(df: pd.DataFrame, cascade: list) -> tuple:
    """
    Try each category string in `cascade` in order.
    Returns (matched_category, filtered_df) for the first non-empty match.
    If nothing matches, returns the last category tried and an empty DataFrame.
    """
    for match_str in cascade:
        mask = df["primary_category"].str.contains(
            match_str, case=False, na=False)
        result = df[mask]
        logger.info(f"  Cascade '{match_str}': {len(result):,} matches")
        if not result.empty:
            return match_str, result

    logger.warning(f"  Cascade exhausted {cascade}, no matches found")
    return cascade[-1], df.iloc[0:0]


def find_nearby_cluster(
    candidates_by_slot: dict[str, pd.DataFrame],
    radius_km: float,
    max_attempts: int = 300,
    preference_weights: Dict[str, float] = None,
    exclude_names: set = None,
) -> list[dict]:
    """
    Find one place per slot where ALL places are within radius_km of each other.

    Parameters
    ----------
    preference_weights : optional dict mapping category substrings → float weights.
        When provided, sampling is biased toward places whose primary_category
        matches a preferred category. Falls back to uniform if weights don't
        cover the pool.
    exclude_names : optional set of place names to skip (e.g. previously visited).

    Algorithm (randomised anchor):
      1. Sort slots by pool size (smallest = most constrained → anchor)
      2. Sample random anchor from smallest pool
      3. For each other slot, filter to places within radius_km of anchor
      4. Sample one from each filtered pool (weighted if preferences exist)
      5. Verify all pairwise distances ≤ radius_km
      6. Retry up to max_attempts times
    """
    # Remove empty pools
    candidates_by_slot = {k: v for k,
                          v in candidates_by_slot.items() if len(v) > 0}
    if not candidates_by_slot:
        return []

    # Pre-filter: exclude previously visited places by name
    if exclude_names:
        candidates_by_slot = {
            k: v[~v["name"].isin(exclude_names)]
            for k, v in candidates_by_slot.items()
        }
        # Remove pools that became empty after exclusion
        candidates_by_slot = {k: v for k,
                              v in candidates_by_slot.items() if len(v) > 0}
        if not candidates_by_slot:
            return []

    def _compute_weights(pool: pd.DataFrame) -> Optional[np.ndarray]:
        """Build per-row sampling weights from preference_weights dict."""
        if not preference_weights:
            return None
        weights = np.ones(len(pool), dtype=float)
        for cat_substr, w in preference_weights.items():
            mask = pool["primary_category"].str.contains(
                cat_substr, case=False, na=False)
            weights[mask.values] += w * 10  # boost preferred categories
        total = weights.sum()
        if total == 0:
            return None
        return weights / total

    # Sort by pool size (smallest first)
    sorted_slots = sorted(candidates_by_slot.items(), key=lambda x: len(x[1]))

    for attempt in range(max_attempts):
        anchor_label, anchor_df = sorted_slots[0]
        anchor_weights = _compute_weights(anchor_df)
        anchor = anchor_df.sample(1, weights=anchor_weights).iloc[0]
        a_lat, a_lon = anchor["latitude"], anchor["longitude"]

        cluster = [anchor.to_dict()]
        used_ids = {anchor["fsq_place_id"]}
        failed = False

        for slot_label, slot_df in sorted_slots[1:]:
            # Exclude already-picked places
            pool = slot_df[~slot_df["fsq_place_id"].isin(used_ids)]
            if pool.empty:
                failed = True
                break

            # Vectorized distance from anchor
            dists = haversine_vectorized(
                a_lat, a_lon, pool["latitude"].values, pool["longitude"].values)
            nearby_mask = dists <= radius_km
            nearby = pool[nearby_mask]

            if nearby.empty:
                failed = True
                break

            nearby_weights = _compute_weights(nearby)
            pick = nearby.sample(1, weights=nearby_weights).iloc[0]
            cluster.append(pick.to_dict())
            used_ids.add(pick["fsq_place_id"])

        if failed:
            continue

        # Verify ALL pairwise distances
        all_ok = True
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                d = haversine_km(
                    cluster[i]["latitude"], cluster[i]["longitude"],
                    cluster[j]["latitude"], cluster[j]["longitude"],
                )
                if d > radius_km:
                    all_ok = False
                    break
            if not all_ok:
                break

        if all_ok:
            logger.info(f"  Cluster found in {attempt + 1} attempts")
            return cluster

    # Fallback: best-effort (picks without strict proximity)
    logger.warning(
        f"Could not find tight cluster after {max_attempts} attempts, returning best-effort")
    fallback = []
    used_ids = set()
    for slot_label, slot_df in sorted_slots:
        pool = slot_df[~slot_df["fsq_place_id"].isin(used_ids)]
        if not pool.empty:
            pool_weights = _compute_weights(pool)
            pick = pool.sample(1, weights=pool_weights).iloc[0]
            fallback.append(pick.to_dict())
            used_ids.add(pick["fsq_place_id"])
    return fallback


# ═══════════════════════════════════════════════
#  8. ROUTE ORDERING (Nearest-Neighbour TSP)
# ═══════════════════════════════════════════════

def order_by_nearest_neighbor(stops: list[ItineraryStop]) -> list[ItineraryStop]:
    """Greedy nearest-neighbour for walking-route optimisation."""
    if len(stops) <= 2:
        for i, s in enumerate(stops):
            s.order = i + 1
        return stops

    remaining = list(stops)
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: haversine_km(
                last.latitude, last.longitude,
                remaining[i].latitude, remaining[i].longitude,
            ),
        )
        ordered.append(remaining.pop(nearest_idx))

    for i, s in enumerate(ordered):
        s.order = i + 1
    return ordered


# ═══════════════════════════════════════════════
#  9. USER AUTH MANAGER
# ═══════════════════════════════════════════════

class UserAuthManager:
    """
    Handles user registration, login, session tokens, and role management.

    Tables
    ------
    users     — id, user_id, username, email, password_hash/salt, role, is_active, timestamps
    sessions  — id, session_token, user_id, created_at, expires_at, is_revoked
    """

    HASH_ITERATIONS = 260_000  # PBKDF2 iteration count (OWASP 2024 rec.)
    TOKEN_BYTES = 64
    TOKEN_TTL_DAYS = 30

    def __init__(self, db_path: str = "itinerary_history.db"):
        self.db_path = db_path
        self._init_db()

    # ── Schema ─────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id       TEXT    NOT NULL UNIQUE,
                    username      TEXT    NOT NULL UNIQUE,
                    email         TEXT    NOT NULL UNIQUE,
                    password_hash TEXT    NOT NULL,
                    password_salt TEXT    NOT NULL,
                    role          TEXT    NOT NULL DEFAULT 'user',
                    is_active     INTEGER NOT NULL DEFAULT 1,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login    TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT    NOT NULL UNIQUE,
                    user_id       TEXT    NOT NULL,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at    TIMESTAMP NOT NULL,
                    is_revoked    INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_token
                ON sessions (session_token)
            """)
            conn.commit()

    # ── Password helpers ───────────────────────

    def _hash_password(self, password: str, salt: str) -> str:
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            self.HASH_ITERATIONS,
        )
        return dk.hex()

    def _make_salt(self) -> str:
        return secrets.token_hex(32)

    def _make_user_id(self) -> str:
        return "usr_" + secrets.token_hex(8)

    def _make_token(self) -> str:
        return secrets.token_hex(self.TOKEN_BYTES)

    # ── Public API ─────────────────────────────

    def register(self, username: str, email: str, password: str, role: str = "user") -> dict:
        """Create a new user account. Returns the public profile dict."""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if len(username) < 2:
            raise ValueError("Username must be at least 2 characters.")

        salt = self._make_salt()
        pw_hash = self._hash_password(password, salt)
        user_id = self._make_user_id()

        if role not in ("user", "admin"):
            role = "user"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users
                        (user_id, username, email, password_hash, password_salt, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, username.strip(), email.strip().lower(), pw_hash, salt, role))
                conn.commit()
        except sqlite3.IntegrityError as exc:
            msg = str(exc)
            if "username" in msg:
                raise ValueError("Username already taken.")
            if "email" in msg:
                raise ValueError("Email already registered.")
            raise ValueError("Registration failed.")

        logger.info(
            f"New user registered: {username} ({user_id}), role={role}")
        return self._public_profile(user_id)

    def login(self, username_or_email: str, password: str) -> dict:
        """Verify credentials and create a session token."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, password_hash, password_salt, is_active, role
                FROM users
                WHERE username = ? OR email = ?
            """, (username_or_email, username_or_email.lower()))
            row = cursor.fetchone()

        if not row:
            raise ValueError("Invalid credentials.")

        user_id, stored_hash, salt, is_active, role = row

        if not is_active:
            raise ValueError("Account is disabled. Contact support.")

        attempt_hash = self._hash_password(password, salt)
        if not hmac.compare_digest(attempt_hash, stored_hash):
            raise ValueError("Invalid credentials.")

        token = self._make_token()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_token, user_id, expires_at)
                VALUES (?, ?, datetime('now', '+30 days'))
            """, (token, user_id))
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
            """, (user_id,))
            conn.commit()

        logger.info(f"User logged in: {user_id}")
        return {"session_token": token, "user": self._public_profile(user_id)}

    def logout(self, session_token: str) -> bool:
        """Revoke a session token. Returns True if token was valid."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET is_revoked = 1
                WHERE session_token = ? AND is_revoked = 0
            """, (session_token,))
            conn.commit()
            return cursor.rowcount > 0

    def get_user_from_token(self, session_token: str) -> Optional[dict]:
        """Resolve a session token → user profile dict (None if invalid)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.user_id
                FROM sessions s
                WHERE s.session_token = ?
                  AND s.is_revoked    = 0
                  AND s.expires_at    > CURRENT_TIMESTAMP
            """, (session_token,))
            row = cursor.fetchone()

        if not row:
            return None
        return self._public_profile(row[0])

    def require_auth(self, session_token: Optional[str]) -> dict:
        """Get user or raise HTTP 401."""
        if not session_token:
            raise HTTPException(
                status_code=401, detail="Missing session token.")
        user = self.get_user_from_token(session_token)
        if not user:
            raise HTTPException(
                status_code=401, detail="Invalid or expired token.")
        return user

    def require_admin(self, session_token: Optional[str]) -> dict:
        """Like require_auth but also asserts role == 'admin'."""
        user = self.require_auth(session_token)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403, detail="Admin access required.")
        return user

    # ── Admin helpers ──────────────────────────

    def list_all_users(self, skip: int = 0, limit: int = 100) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, email, role, is_active, created_at, last_login
                FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, skip))
            rows = cursor.fetchall()

        return [
            {
                "user_id":    r[0], "username":  r[1], "email":      r[2],
                "role":       r[3], "is_active": bool(r[4]),
                "created_at": r[5], "last_login": r[6],
            }
            for r in rows
        ]

    def set_user_role(self, target_user_id: str, new_role: str) -> bool:
        if new_role not in ("user", "admin"):
            raise ValueError("Role must be 'user' or 'admin'.")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET role = ? WHERE user_id = ?", (new_role, target_user_id))
            conn.commit()
            return cursor.rowcount > 0

    def set_user_active(self, target_user_id: str, is_active: bool) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = ? WHERE user_id = ?", (int(
                is_active), target_user_id))
            conn.commit()
            return cursor.rowcount > 0

    # ── Private ────────────────────────────────

    def _public_profile(self, user_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, email, role, is_active, created_at, last_login
                FROM users WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()
        if not row:
            return {}
        return {
            "user_id":    row[0], "username":   row[1], "email":      row[2],
            "role":       row[3], "is_active":  bool(row[4]),
            "created_at": row[5], "last_login": row[6],
        }


# ═══════════════════════════════════════════════
#  10. HISTORY MANAGER
# ═══════════════════════════════════════════════

class HistoryManager:
    """Stores completed itineraries in SQLite for predictive modelling."""

    def __init__(self, db_path: str = "itinerary_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS itineraries (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id           TEXT,
                    preset            TEXT,
                    preset_display    TEXT,
                    country           TEXT,
                    region            TEXT,
                    locality          TEXT,
                    total_stops       INTEGER,
                    total_distance_km REAL,
                    categories        TEXT,
                    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS itinerary_stops (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    itinerary_id     INTEGER,
                    stop_order       INTEGER,
                    name             TEXT,
                    category         TEXT,
                    latitude         REAL,
                    longitude        REAL,
                    locality         TEXT,
                    website          TEXT,
                    tel              TEXT,
                    email            TEXT,
                    facebook_id      TEXT,
                    instagram        TEXT,
                    twitter          TEXT,
                    digital_presence INTEGER DEFAULT 0,
                    tip              TEXT,
                    FOREIGN KEY (itinerary_id) REFERENCES itineraries (id)
                )
            """)
            conn.commit()

            # ── Migrate existing databases ─────────────
            # Add new columns if they don't exist yet (safe to run repeatedly)
            migrations = [
                ("itineraries",    "preset_display",  "TEXT"),
                ("itinerary_stops", "locality",        "TEXT"),
                ("itinerary_stops", "website",         "TEXT"),
                ("itinerary_stops", "tel",             "TEXT"),
                ("itinerary_stops", "email",           "TEXT"),
                ("itinerary_stops", "facebook_id",     "TEXT"),
                ("itinerary_stops", "instagram",       "TEXT"),
                ("itinerary_stops", "twitter",         "TEXT"),
                ("itinerary_stops", "digital_presence", "INTEGER DEFAULT 0"),
                ("itinerary_stops", "tip",             "TEXT"),
            ]
            for table, col, col_type in migrations:
                try:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()

    def save_itinerary(self, user_id: str, itinerary: Itinerary) -> int:
        """Save a completed itinerary. Returns the itinerary ID."""
        categories = list(set(stop.category for stop in itinerary.stops))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO itineraries
                    (user_id, preset, preset_display, country, region, locality,
                     total_stops, total_distance_km, categories)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, itinerary.preset, itinerary.preset_display,
                itinerary.country, itinerary.region,
                getattr(itinerary, "locality", ""),
                len(itinerary.stops), itinerary.total_distance_km,
                json.dumps(categories),
            ))
            itinerary_id = cursor.lastrowid

            for stop in itinerary.stops:
                cursor.execute("""
                    INSERT INTO itinerary_stops
                        (itinerary_id, stop_order, name, category,
                         latitude, longitude, locality, website, tel, email,
                         facebook_id, instagram, twitter, digital_presence, tip)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    itinerary_id, stop.order, stop.name, stop.category,
                    stop.latitude, stop.longitude,
                    stop.locality, stop.website, stop.tel, stop.email,
                    str(stop.facebook_id) if stop.facebook_id else "",
                    stop.instagram, stop.twitter,
                    stop.digital_presence, stop.tip,
                ))

            conn.commit()
            return itinerary_id

    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Fetch itinerary summaries
            cursor.execute("""
                SELECT id, preset, preset_display, country, region, locality,
                       total_stops, total_distance_km, categories, created_at
                FROM itineraries
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit))
            itinerary_rows = cursor.fetchall()

            if not itinerary_rows:
                return []

            # Fetch all stops for these itineraries in one query
            itinerary_ids = [r[0] for r in itinerary_rows]
            placeholders = ",".join("?" * len(itinerary_ids))
            cursor.execute(f"""
                SELECT itinerary_id, stop_order, name, category,
                       latitude, longitude, locality, website, tel, email,
                       facebook_id, instagram, twitter, digital_presence, tip
                FROM itinerary_stops
                WHERE itinerary_id IN ({placeholders})
                ORDER BY itinerary_id, stop_order
            """, itinerary_ids)
            stop_rows = cursor.fetchall()

        # Group stops by itinerary_id
        stops_by_itinerary: Dict[int, list] = {}
        for s in stop_rows:
            itin_id = s[0]
            if itin_id not in stops_by_itinerary:
                stops_by_itinerary[itin_id] = []
            stops_by_itinerary[itin_id].append({
                "order":            s[1],
                "name":             s[2],
                "category":         s[3],
                "latitude":         s[4],
                "longitude":        s[5],
                "locality":         s[6] or "",
                "website":          s[7] or "",
                "tel":              s[8] or "",
                "email":            s[9] or "",
                "facebook_id":      s[10] or "",
                "instagram":        s[11] or "",
                "twitter":          s[12] or "",
                "digital_presence": s[13] or 0,
                "tip":              s[14] or "",
            })

        return [
            {
                "id":                r[0],
                "preset":            r[1],
                "preset_display":    r[2] or "",
                "country":           r[3],
                "region":            r[4],
                "locality":          r[5],
                "total_stops":       r[6],
                "total_distance_km": r[7],
                "categories":        json.loads(r[8]) if r[8] else [],
                "created_at":        r[9],
                "stops":             stops_by_itinerary.get(r[0], []),
            }
            for r in itinerary_rows
        ]

    def get_popular_categories(self, user_id: str = None, days: int = 30) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("""
                    SELECT categories FROM itineraries
                    WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
                """.format(days), (user_id,))
            else:
                cursor.execute("""
                    SELECT categories FROM itineraries
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days))

            rows = cursor.fetchall()
            category_counts: Dict[str, int] = {}
            for row in rows:
                if row[0]:
                    for cat in json.loads(row[0]):
                        category_counts[cat] = category_counts.get(cat, 0) + 1

        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))

    def get_popular_stats(self, days: int = 30) -> Dict:
        """
        Build a comprehensive popularity report across all users.
        Returns top categories, presets, regions, countries, and total itinerary count.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT preset, country, region, locality, categories
                FROM itineraries
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days))
            rows = cursor.fetchall()

        if not rows:
            return {
                "total_itineraries": 0,
                "top_categories": [],
                "top_presets": [],
                "top_countries": [],
                "top_regions": [],
                "top_localities": [],
                "days": days,
            }

        presets:    Dict[str, int] = {}
        countries:  Dict[str, int] = {}
        regions:    Dict[str, int] = {}
        localities: Dict[str, int] = {}
        categories: Dict[str, int] = {}

        for row in rows:
            preset, country, region, locality, cats_json = row
            if preset:
                presets[preset] = presets.get(preset, 0) + 1
            if country:
                countries[country] = countries.get(country, 0) + 1
            if region:
                regions[region] = regions.get(region, 0) + 1
            if locality:
                localities[locality] = localities.get(locality, 0) + 1
            if cats_json:
                for cat in json.loads(cats_json):
                    categories[cat] = categories.get(cat, 0) + 1

        def _top(d: dict, n: int = 10) -> list:
            return [
                {"name": k, "count": v}
                for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
            ]

        return {
            "total_itineraries": len(rows),
            "top_categories":    _top(categories),
            "top_presets":       _top(presets),
            "top_countries":     _top(countries),
            "top_regions":       _top(regions, 15),
            "top_localities":    _top(localities, 15),
            "days":              days,
        }

    def get_user_preferences(self, user_id: str) -> Dict:
        history = self.get_user_history(user_id, limit=100)
        if not history:
            return {}

        presets:    Dict[str, int] = {}
        countries:  Dict[str, int] = {}
        regions:    Dict[str, int] = {}
        categories: Dict[str, int] = {}

        for item in history:
            presets[item["preset"]] = presets.get(item["preset"], 0) + 1
            countries[item["country"]] = countries.get(item["country"], 0) + 1
            regions[item["region"]] = regions.get(item["region"], 0) + 1
            for cat in item["categories"]:
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "favorite_preset":      max(presets.items(),   key=lambda x: x[1])[0] if presets else None,
            "favorite_country":     max(countries.items(), key=lambda x: x[1])[0] if countries else None,
            "favorite_region":      max(regions.items(),   key=lambda x: x[1])[0] if regions else None,
            "preferred_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5],
            "total_itineraries":    len(history),
        }

    def get_visited_place_names(self, user_id: str, limit: int = 200) -> set:
        """
        Return a set of place names the user has visited in recent itineraries.
        Used to avoid recommending the same places on reshuffles.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT s.name
                FROM itinerary_stops s
                JOIN itineraries i ON s.itinerary_id = i.id
                WHERE i.user_id = ?
                ORDER BY i.created_at DESC
                LIMIT ?
            """, (user_id, limit))
            return {row[0] for row in cursor.fetchall()}

    def get_category_weights(self, user_id: str) -> Dict[str, float]:
        """
        Build a category → weight mapping from the user's history.
        More frequently visited categories get higher weights.
        Returns normalised weights (0.0–1.0) suitable for pandas sampling.
        If no history, returns empty dict (caller falls back to uniform).
        """
        cat_counts = self.get_popular_categories(user_id=user_id, days=90)
        if not cat_counts:
            return {}

        total = sum(cat_counts.values())
        return {cat: count / total for cat, count in cat_counts.items()}


# ═══════════════════════════════════════════════
#  11. FAVORITES MANAGER
# ═══════════════════════════════════════════════

class FavoritesManager:
    """Per-user favorite stops (UNIQUE on user_id + fsq_place_id)."""

    def __init__(self, db_path: str = "itinerary_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      TEXT    NOT NULL,
                    fsq_place_id TEXT,
                    name         TEXT    NOT NULL,
                    category     TEXT,
                    latitude     REAL    NOT NULL,
                    longitude    REAL    NOT NULL,
                    locality     TEXT,
                    region       TEXT,
                    country      TEXT,
                    website      TEXT,
                    tel          TEXT,
                    note         TEXT,
                    saved_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE (user_id, fsq_place_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_favorites_user
                ON favorites (user_id)
            """)
            conn.commit()

    # ── Core actions ───────────────────────────

    def add_favorite(
        self,
        user_id: str, name: str, latitude: float, longitude: float,
        fsq_place_id: str = "", category: str = "",
        locality: str = "", region: str = "", country: str = "",
        website: str = "", tel: str = "", note: str = "",
    ) -> dict:
        fsq_id_val = fsq_place_id.strip() if fsq_place_id and fsq_place_id.strip() else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO favorites
                        (user_id, fsq_place_id, name, category,
                         latitude, longitude, locality, region, country,
                         website, tel, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, fsq_id_val, name.strip(), category,
                      latitude, longitude, locality, region, country,
                      website, tel, note))
                conn.commit()
                fav_id = cursor.lastrowid
            logger.info(f"Favorite saved: '{name}' for user {user_id}")
            return self.get_favorite(fav_id)
        except sqlite3.IntegrityError:
            raise ValueError(f"'{name}' is already in your favorites.")

    def remove_favorite(self, user_id: str, favorite_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM favorites WHERE id = ? AND user_id = ?", (favorite_id, user_id))
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Favorite {favorite_id} removed for user {user_id}")
        return deleted

    def update_note(self, user_id: str, favorite_id: int, note: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE favorites SET note = ? WHERE id = ? AND user_id = ?", (note, favorite_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    # ── Retrieval ──────────────────────────────

    def get_favorite(self, favorite_id: int) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, fsq_place_id, name, category,
                       latitude, longitude, locality, region, country,
                       website, tel, note, saved_at
                FROM favorites WHERE id = ?
            """, (favorite_id,))
            row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_user_favorites(
        self,
        user_id: str,
        category: str = None, country: str = None, region: str = None,
        skip: int = 0, limit: int = 100,
    ) -> List[dict]:
        query = """
            SELECT id, user_id, fsq_place_id, name, category,
                   latitude, longitude, locality, region, country,
                   website, tel, note, saved_at
            FROM favorites WHERE user_id = ?
        """
        params: list = [user_id]

        if category:
            query += " AND category LIKE ?"
            params.append(f"%{category}%")
        if country:
            query += " AND LOWER(country) = ?"
            params.append(country.lower())
        if region:
            query += " AND LOWER(region) = ?"
            params.append(region.lower())

        query += " ORDER BY saved_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_dict(r) for r in rows]

    def is_favorited(self, user_id: str, fsq_place_id: str) -> bool:
        if not fsq_place_id:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM favorites WHERE user_id = ? AND fsq_place_id = ? LIMIT 1",
                (user_id, fsq_place_id),
            )
            return cursor.fetchone() is not None

    def get_favorites_count(self, user_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM favorites WHERE user_id = ?", (user_id,))
            return cursor.fetchone()[0]

    # ── Private ────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id":           row[0],  "user_id":      row[1],
            "fsq_place_id": row[2],  "name":         row[3],
            "category":     row[4],  "latitude":     row[5],
            "longitude":    row[6],  "locality":     row[7],
            "region":       row[8],  "country":      row[9],
            "website":      row[10], "tel":          row[11],
            "note":         row[12], "saved_at":     row[13],
        }


# ═══════════════════════════════════════════════
#  12. PLACES DATA STORE (Dask)
# ═══════════════════════════════════════════════

class PlacesDataStore:
    """
    Loads the 16M pre-filtered dataset via Dask.
    Data stays lazy until the user makes a request
    (region + category filter first).
    """

    PARQUET_COLUMNS = [
        "fsq_place_id", "name", "latitude", "longitude",
        "primary_category", "locality", "region", "country",
        "fsq_category_labels", "website", "tel", "email",
        "facebook_id", "instagram", "twitter", "digital_presence",
    ]

    def __init__(self, shard_path: str):
        self.shard_path = shard_path
        self.ddf: Optional[dd.DataFrame] = None
        self._regions_cache:    Optional[dict] = None
        self._localities_cache: Optional[dict] = None
        self._country_mappings: Optional[dict] = None

    # ── Loading ────────────────────────────────

    def load(self):
        """Load the Dask dataframe (lazy — no data in memory yet)."""
        logger.info(f"Loading Dask dataframe from {self.shard_path}")
        try:
            self.ddf = dd.read_parquet(
                self.shard_path, columns=self.PARQUET_COLUMNS)
            total = self.ddf.shape[0].compute()
            logger.info(f"Loaded {total:,} places (lazy via Dask)")
        except Exception as e:
            logger.error(f"Failed to load parquet shards: {e}")
            raise

        self.load_country_mappings()

    def load_country_mappings(self):
        """Load country code ↔ name mappings from country_code.csv."""
        if self._country_mappings is not None:
            return self._country_mappings

        csv_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "country_code.csv")
        try:
            country_df = pd.read_csv(csv_path, usecols=["name", "alpha-2"])
            name_to_code: Dict[str, str] = {}
            code_to_name: Dict[str, str] = {}

            for _, row in country_df.iterrows():
                name, code = row["name"], row["alpha-2"]
                if pd.isna(name) or pd.isna(code):
                    continue
                name_to_code[name.upper()] = code
                code_to_name[code.upper()] = name

            self._country_mappings = {
                "name_to_code": name_to_code, "code_to_name": code_to_name}
            logger.info(f"Loaded {len(name_to_code)} country mappings")
        except Exception as e:
            logger.error(f"Failed to load country mappings: {e}")
            self._country_mappings = {"name_to_code": {}, "code_to_name": {}}

        return self._country_mappings

    # ── Region / locality lookups ──────────────

    def get_regions_by_country(self) -> dict:
        """Returns {country_full_name: [regions]} for the UI dropdown. Cached."""
        if self._regions_cache is not None:
            return self._regions_cache

        logger.info("Computing available regions (one-time)...")
        region_df = self.ddf[["country", "region"]].drop_duplicates().compute()
        region_df = region_df.dropna(subset=["country", "region"])
        region_df = region_df[region_df["region"].str.strip().str.len() > 0]

        result = {}
        for country_code, group in region_df.groupby("country"):
            regions = sorted(group["region"].unique().tolist())
            if regions:
                country_name = self._country_mappings["code_to_name"].get(
                    country_code, country_code)
                result[country_name] = regions

        self._regions_cache = result
        logger.info(
            f"Found {len(result)} countries, {sum(len(v) for v in result.values())} regions")
        return result

    def get_localities_by_country(self) -> dict:
        """Returns {country_full_name: [localities]} for the UI dropdown. Cached."""
        if self._localities_cache is not None:
            return self._localities_cache

        logger.info("Computing available localities (one-time)...")
        locality_df = self.ddf[["country", "locality"]
                               ].drop_duplicates().compute()
        locality_df = locality_df.dropna(subset=["country", "locality"])
        locality_df = locality_df[locality_df["locality"].str.strip(
        ).str.len() > 0]

        result = {}
        for country_code, group in locality_df.groupby("country"):
            localities = sorted(group["locality"].unique().tolist())
            if localities:
                country_name = self._country_mappings["code_to_name"].get(
                    country_code, country_code)
                result[country_name] = localities

        self._localities_cache = result
        logger.info(
            f"Found {len(result)} countries, {sum(len(v) for v in result.values())} localities")
        return result

    def get_localities_by_region(self, region: str, country_name: str) -> list:
        """Returns list of localities for a specific region + country."""
        country_code = self._country_mappings["name_to_code"].get(
            country_name.upper(), country_name.upper(),
        )
        filtered = self.ddf[
            (self.ddf["region"].str.lower() == region.lower().strip())
            & (self.ddf["country"].str.upper() == country_code.upper().strip())
        ]
        localities_df = filtered[["locality"]].drop_duplicates().compute()
        localities_df = localities_df.dropna(subset=["locality"])
        localities_df = localities_df[localities_df["locality"].str.strip(
        ).str.len() > 0]

        localities = sorted(localities_df["locality"].unique().tolist())
        logger.info(
            f"Found {len(localities)} localities in {region}, {country_name}")
        return localities

    # ── Query helpers ──────────────────────────

    def query_region(self, region: str, country_name: str) -> pd.DataFrame:
        """Pull all places for a region + country into Pandas (16M → ~tens of thousands)."""
        country_code = self._country_mappings["name_to_code"].get(
            country_name.upper(), country_name.upper(),
        )
        filtered = self.ddf[
            (self.ddf["region"].str.lower() == region.lower().strip())
            & (self.ddf["country"].str.upper() == country_code.upper().strip())
        ]
        df = filtered.compute()
        logger.info(
            f"Region filter [{region}, {country_name} ({country_code})]: {len(df):,} places")
        return df

    def filter_by_category(self, df: pd.DataFrame, match: str) -> pd.DataFrame:
        """Filter a Pandas DataFrame by category substring on primary_category."""
        mask = df["primary_category"].str.contains(match, case=False, na=False)
        result = df[mask]
        logger.info(f"  Category '{match}': {len(result):,} matches")
        return result


# ═══════════════════════════════════════════════
#  13. ITINERARY GENERATION ENGINE
# ═══════════════════════════════════════════════

def generate_itinerary(
    store: PlacesDataStore,
    preset: str,
    region: str,
    country_name: str,
    locality: str = None,
    user_id: str = "",
    hour: int = None,
    preference_weights: Dict[str, float] = None,
    exclude_names: set = None,
) -> Itinerary:
    """
    Full pipeline:
      1. Dask query → pull region data into Pandas (16M → ~50K)
      2. Optional: filter by locality
      3. Category filter per preset rule (supports cascade rules)
      4. Proximity clustering (find N nearby places, preference-weighted)
      5. Route ordering (nearest-neighbour)
      6. Build itinerary object

    Parameters
    ----------
    preference_weights : optional category → weight dict from user history.
        Biases sampling toward categories the user has enjoyed before.
    exclude_names : optional set of place names to avoid (previously visited).
    """
    # ── Resolve config ─────────────────────────
    preset_key = preset.lower()
    if preset_key not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from: {list(PRESET_CONFIGS.keys())}")

    if hour is None:
        hour = datetime.now().hour

    if preset_key == "hangout":
        config = get_hangout_rules(hour)
    elif preset_key == "date":
        config = get_date_rules(hour)
    else:
        config = PRESET_CONFIGS[preset_key]

    time_slot = config.get("time_slot") or get_time_slot(hour)
    logger.info(
        f"Generating [{preset}] | {region}, {country_name} | "
        f"hour={hour} slot={time_slot} nightlife={config.get('nightlife_eligible')}"
    )

    # ── Step 1: Region filter ──────────────────
    regional_df = store.query_region(region, country_name)
    if regional_df.empty:
        raise ValueError(
            f"No places found in region='{region}', country='{country_name}'")

    # ── Step 2: Optional locality filter ───────
    if locality:
        regional_df = regional_df[regional_df["locality"].str.lower(
        ) == locality.lower().strip()]
        if regional_df.empty:
            raise ValueError(
                f"No places found in locality='{locality}', region='{region}', country='{country_name}'"
            )
        logger.info(
            f"Locality filter [{locality}]: {len(regional_df):,} places")

    # ── Step 3: Category filter → slot pools ───
    candidates_by_slot:  Dict[str, pd.DataFrame] = {}
    resolved_categories: Dict[str, str] = {}

    for rule in config["category_rules"]:
        pick_count = rule["pick"]

        if "match_cascade" in rule:
            slot_label = rule.get("slot", "cascade")
            matched_cat, cat_df = filter_by_cascade(
                regional_df, rule["match_cascade"])
            resolved_categories[slot_label] = matched_cat

            if cat_df.empty:
                logger.warning(
                    f"  No places for any cascade category in {region}")

            if pick_count > 1:
                for i in range(pick_count):
                    candidates_by_slot[f"{slot_label}_{i}"] = cat_df
            else:
                candidates_by_slot[slot_label] = cat_df

        else:
            match_str = rule["match"]
            cat_df = store.filter_by_category(regional_df, match_str)
            resolved_categories[match_str] = match_str

            if cat_df.empty:
                logger.warning(
                    f"  No places for category '{match_str}' in {region}")

            if pick_count > 1:
                for i in range(pick_count):
                    candidates_by_slot[f"{match_str}_{i}"] = cat_df
            else:
                candidates_by_slot[match_str] = cat_df

    # ── Step 4: Proximity clustering ───────────
    radius = config["cluster_radius_km"]
    cluster = find_nearby_cluster(
        candidates_by_slot, radius,
        preference_weights=preference_weights,
        exclude_names=exclude_names,
    )

    if not cluster:
        # If exclusion made it impossible, retry without exclusions
        if exclude_names:
            logger.info("  Retrying clustering without exclusions...")
            cluster = find_nearby_cluster(
                candidates_by_slot, radius,
                preference_weights=preference_weights,
                exclude_names=None,
            )

    if not cluster:
        raise ValueError(
            f"Could not find enough matching places in {region}. "
            f"Try a different region or preset."
        )

    # ── Step 5: Build stop objects ─────────────
    stops = []
    for place in cluster:
        cat = place.get("primary_category", "")
        stops.append(ItineraryStop(
            name=place.get("name", "Unknown"),
            category=cat if cat else "",
            latitude=place["latitude"],
            longitude=place["longitude"],
            locality=str(place.get("locality", "") or ""),
            website=str(place.get("website", "") or ""),
            tel=str(place.get("tel", "") or ""),
            email=str(place.get("email", "") or ""),
            facebook_id=place.get("facebook_id") if pd.notna(
                place.get("facebook_id")) else "",
            instagram=str(place.get("instagram", "") or ""),
            twitter=str(place.get("twitter", "") or ""),
            digital_presence=int(place.get("digital_presence", 0)),
        ))

    # ── Step 6: Route ordering ─────────────────
    stops = order_by_nearest_neighbor(stops)

    total_dist = 0.0
    for i in range(len(stops) - 1):
        total_dist += haversine_km(
            stops[i].latitude, stops[i].longitude,
            stops[i + 1].latitude, stops[i + 1].longitude,
        )

    itinerary = Itinerary(
        preset=preset,
        preset_display=config["display_name"],
        region=region,
        country=country_name,
        locality=locality or "",
        user_id=user_id,
        stops=stops,
        total_distance_km=round(total_dist, 2),
    )

    # Attach time metadata for the endpoint response
    itinerary._time_slot = time_slot
    itinerary._hour = hour
    itinerary._nightlife_eligible = config.get("nightlife_eligible", False)
    itinerary._resolved_categories = resolved_categories

    return itinerary


# ═══════════════════════════════════════════════
#  14. FASTAPI APPLICATION & ENDPOINTS
# ═══════════════════════════════════════════════

# ── Global singletons ─────────────────────────

store = PlacesDataStore(SHARD_PATH)
history_manager = HistoryManager()
auth_manager = UserAuthManager()
favorites_manager = FavoritesManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Dask dataframe on startup, cleanup on shutdown."""
    store.load()
    yield


app = FastAPI(
    title="WhatsNext",
    description="Random destination itinerary generator powered by Foursquare OS Places (16M POIs)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache: stores the last generated Itinerary per user.
# Key = user_id (str), Value = Itinerary object.
# Overwritten on every /api/generate call for that user.
_last_generation_cache: Dict[str, Itinerary] = {}

# Tracks how many times a user has reshuffled (generated without saving).
# Key = user_id (str), Value = int count.
# Reset when the user saves via /api/save or changes preset/region/country.
_reshuffle_count: Dict[str, int] = {}

# Tracks the last generation params per user so we can detect reshuffles vs new queries.
# Key = user_id, Value = (preset, region, country, locality).
_last_gen_params: Dict[str, tuple] = {}


# ── Discovery endpoints ───────────────────────

@app.get("/api/presets")
async def get_presets():
    """Available activity presets."""
    result = {}
    now_hour = datetime.now().hour
    for k, v in PRESET_CONFIGS.items():
        if k == "hangout":
            cfg = get_hangout_rules(now_hour)
        elif k == "date":
            cfg = get_date_rules(now_hour)
        else:
            cfg = v
        result[k] = {
            "display_name":      cfg["display_name"],
            "description":       cfg["description"],
            "total_stops":       cfg["total_stops"],
            "cluster_radius_km": cfg["cluster_radius_km"],
        }
    return result


@app.get("/api/countries")
async def get_countries():
    """Get all available countries."""
    all_regions = store.get_regions_by_country()
    return {"countries": sorted(list(all_regions.keys()))}


@app.get("/api/regions")
async def get_regions(country: str = None):
    """Available regions grouped by country. Pass ?country=Australia to filter."""
    all_regions = store.get_regions_by_country()
    if country:
        for key in all_regions:
            if key.upper() == country.upper():
                return {key: all_regions[key]}
        return {}
    return all_regions


@app.get("/api/localities")
async def get_localities(
    country: str = Query(..., description="Full country name"),
    region:  str = Query(..., description="Region/state name"),
):
    """Get localities for a specific region."""
    try:
        localities = store.get_localities_by_region(region, country)
        return {"country": country, "region": region, "localities": localities}
    except Exception as e:
        logger.error(f"Error fetching localities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Dataset statistics."""
    try:
        total = store.ddf.shape[0].compute()
        countries = store.ddf["country"].nunique().compute()
        return {"total_places": int(total), "countries": int(countries)}
    except Exception:
        return {"total_places": 0, "countries": 0}


# ── Generation endpoint ───────────────────────

@app.get("/api/generate")
async def generate(
    preset:   str = Query(...,
                          description="Preset: hangout, date, study, or shop"),
    region:   str = Query(...,  description="Region/state name"),
    country:  str = Query(...,  description="Full country name"),
    locality: str = Query(None, description="Optional locality/city name"),
    hour:     int = Query(
        None, ge=0, le=23, description="Local hour (0–23) for time-aware presets"),
    x_session_token: Optional[str] = Header(None),
):
    """
    Generate a proximity-clustered itinerary.

    **Authentication (optional):**
    - Pass `X-Session-Token` header to enable preference-aware generation.
    - If logged in, the user is auto-detected — no need to pass user_id manually.
    - Works without auth too — just generates a random itinerary as anonymous.

    **Preference-aware generation (logged-in users):**
    - First generation is biased toward their preferred categories (weighted sampling).
    - Previously visited places are excluded so reshuffles feel fresh.
    - After multiple reshuffles (same preset + region), preference weighting
      fades out and generation becomes fully random — giving the user variety.

    **Reshuffle behaviour:**
    - Calling /api/generate repeatedly with the same params = reshuffle.
    - Each reshuffle excludes stops from all previous unsaved generations.
    - After 3+ reshuffles, preferences are dropped and it goes fully random.
    - Saving via /api/save resets the reshuffle counter.

    The `hour` parameter controls the hangout/date preset's 4th slot:
      - 17:00–04:00 → nightlife cascade
      - 06:00–16:59 → daytime entertainment cascade
    """
    try:
        # ── Auto-detect user from session token ───
        user_id = ""
        if x_session_token:
            user = auth_manager.get_user_from_token(x_session_token)
            if user:
                user_id = user["user_id"]

        cache_key = user_id or "anonymous"
        gen_params = (preset.lower(), region.lower(),
                      country.lower(), (locality or "").lower())

        # ── Detect reshuffle vs new query ──────────
        prev_params = _last_gen_params.get(cache_key)
        if prev_params == gen_params:
            _reshuffle_count[cache_key] = _reshuffle_count.get(
                cache_key, 0) + 1
        else:
            # New query — reset reshuffle state
            _reshuffle_count[cache_key] = 0
        _last_gen_params[cache_key] = gen_params

        reshuffle_num = _reshuffle_count.get(cache_key, 0)

        # ── Build preference weights from history ──
        preference_weights = None
        exclude_names = set()
        weight_source = "none"

        if user_id:
            # Collect names from the current cached itinerary (if reshuffling)
            prev_itinerary = _last_generation_cache.get(cache_key)
            if prev_itinerary:
                for stop in prev_itinerary.stops:
                    exclude_names.add(stop.name)

            # Also exclude places from saved history
            visited = history_manager.get_visited_place_names(
                user_id, limit=100)
            exclude_names.update(visited)

            # Use user preference weighting for the first few generations
            if reshuffle_num < 3:
                preference_weights = history_manager.get_category_weights(
                    user_id)
                if preference_weights:
                    weight_source = "user_history"
                    logger.info(
                        f"Using user preference weights for {user_id} "
                        f"(reshuffle #{reshuffle_num}, {len(preference_weights)} categories)"
                    )

        # If no user preferences available (anonymous, new user, or reshuffled past threshold),
        # fall back to global popular categories so generation isn't fully random.
        # After 6+ reshuffles, skip popular too — go fully random for maximum variety.
        if not preference_weights and reshuffle_num < 6:
            popular = history_manager.get_popular_categories(days=30)
            if popular:
                total = sum(popular.values())
                preference_weights = {cat: count /
                                      total for cat, count in popular.items()}
                weight_source = "global_popular"
                logger.info(
                    f"Using global popular weights for '{cache_key}' "
                    f"({len(preference_weights)} categories)"
                )
            else:
                logger.info(
                    f"No popular data available for '{cache_key}' — fully random")
        elif reshuffle_num >= 6:
            preference_weights = None
            weight_source = "random"
            logger.info(
                f"Reshuffle #{reshuffle_num} for '{cache_key}' — fully random")

        # ── Generate ───────────────────────────────
        itinerary = generate_itinerary(
            store, preset, region, country, locality, user_id, hour=hour,
            preference_weights=preference_weights,
            exclude_names=exclude_names if exclude_names else None,
        )

        # ── Generate place tips via Claude ─────────
        tips = await generate_place_tips(
            stops=itinerary.stops,
            preset=preset,
            region=region,
            country=country,
            time_slot=getattr(itinerary, "_time_slot", ""),
        )
        # Attach tips to each stop
        for stop in itinerary.stops:
            stop.tip = tips.get(stop.name, "")

        # Cache for /api/save
        _last_generation_cache[cache_key] = itinerary
        logger.info(
            f"Cached itinerary for '{cache_key}' (reshuffle #{reshuffle_num})")

        return {
            "preset":            itinerary.preset,
            "preset_display":    itinerary.preset_display,
            "region":            itinerary.region,
            "country":           itinerary.country,
            "locality":          itinerary.locality,
            "user_id":           itinerary.user_id,
            "stops":             [asdict(s) for s in itinerary.stops],
            "total_distance_km": itinerary.total_distance_km,
            "time_context": {
                "hour":                itinerary._hour,
                "time_slot":           itinerary._time_slot,
                "nightlife_eligible":  itinerary._nightlife_eligible,
                "resolved_categories": itinerary._resolved_categories,
            },
            "generation_meta": {
                "reshuffle_number":    reshuffle_num,
                "preference_applied":  preference_weights is not None and len(preference_weights) > 0,
                "weight_source":       weight_source,
                "excluded_place_count": len(exclude_names),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to generate itinerary. Check server logs.")


# ── Save last generation to history ────────────

@app.post("/api/save")
async def save_last_generation(
    x_session_token: Optional[str] = Header(None),
):
    """
    Save the most recently generated itinerary to the user's history.

    Requires `X-Session-Token` header — the user must be logged in to save.
    Looks up the cached itinerary from the last /api/generate call.

    Returns the saved itinerary ID and summary.
    """
    # Require login
    user = auth_manager.require_auth(x_session_token)
    resolved_user_id = user["user_id"]

    # Look up the cached itinerary
    cache_key = resolved_user_id
    itinerary = _last_generation_cache.get(cache_key)

    # Also check "anonymous" cache if user generated before logging in
    if not itinerary and cache_key != "anonymous":
        itinerary = _last_generation_cache.get("anonymous")

    if not itinerary:
        raise HTTPException(
            status_code=404,
            detail="No cached itinerary found. Generate one first via /api/generate.",
        )

    # Save to history
    try:
        history_id = history_manager.save_itinerary(
            resolved_user_id, itinerary)
        logger.info(
            f"Saved itinerary {history_id} to history for user {resolved_user_id}")

        # Clear the cache entry after saving and reset reshuffle counter
        _last_generation_cache.pop(cache_key, None)
        _last_generation_cache.pop("anonymous", None)
        _reshuffle_count.pop(cache_key, None)
        _reshuffle_count.pop("anonymous", None)
        _last_gen_params.pop(cache_key, None)
        _last_gen_params.pop("anonymous", None)

        return {
            "detail":       "Itinerary saved to history.",
            "history_id":   history_id,
            "user_id":      resolved_user_id,
            "preset":       itinerary.preset,
            "region":       itinerary.region,
            "country":      itinerary.country,
            "locality":     itinerary.locality,
            "total_stops":  len(itinerary.stops),
            "total_distance_km": itinerary.total_distance_km,
        }
    except Exception as e:
        logger.error(f"Save error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to save itinerary.")


# ── Reshuffle single stop ─────────────────────

@app.post("/api/reshuffle-stop")
async def reshuffle_single_stop(
    index: int = Query(..., ge=0,
                       description="Index of the stop to replace (0-based)"),
    x_session_token: Optional[str] = Header(None),
):
    """
    Replace a single stop in the cached itinerary with a new one.

    Keeps all other stops unchanged. The replacement is:
      1. Same category as the original stop
      2. Within the cluster radius of the remaining stops
      3. Not a duplicate of any existing stop

    The cached itinerary is updated in-place.
    A new Groq tip is generated for the replacement stop only.

    Parameters
    ----------
    index : 0-based index of the stop to replace
    X-Session-Token : optional, for user-specific cache
    """
    # Resolve user
    user_id = ""
    if x_session_token:
        user = auth_manager.get_user_from_token(x_session_token)
        if user:
            user_id = user["user_id"]
    cache_key = user_id or "anonymous"

    # Get cached itinerary
    itinerary = _last_generation_cache.get(cache_key)
    if not itinerary:
        raise HTTPException(
            status_code=404, detail="No cached itinerary. Generate one first.")

    if index >= len(itinerary.stops):
        raise HTTPException(
            status_code=400, detail=f"Index {index} out of range. Itinerary has {len(itinerary.stops)} stops.")

    old_stop = itinerary.stops[index]
    logger.info(
        f"Reshuffling stop #{index} '{old_stop.name}' (category: {old_stop.category})")

    try:
        # Query the region data
        regional_df = store.query_region(itinerary.region, itinerary.country)
        if itinerary.locality:
            regional_df = regional_df[
                regional_df["locality"].str.lower(
                ) == itinerary.locality.lower().strip()
            ]

        # Filter to same category as the old stop
        cat_df = store.filter_by_category(regional_df, old_stop.category)
        if cat_df.empty:
            raise ValueError(
                f"No alternative places found for category '{old_stop.category}'")

        # Exclude all current stops by name
        current_names = {s.name for s in itinerary.stops}
        cat_df = cat_df[~cat_df["name"].isin(current_names)]
        if cat_df.empty:
            raise ValueError(
                f"No alternative places left for category '{old_stop.category}' (all used)")

        # Get coordinates of the other stops (not the one being replaced)
        other_stops = [s for i, s in enumerate(itinerary.stops) if i != index]

        # Filter to places within cluster radius of all other stops
        radius = MAX_CLUSTER_RADIUS_KM
        for other in other_stops:
            dists = haversine_vectorized(
                other.latitude, other.longitude,
                cat_df["latitude"].values, cat_df["longitude"].values,
            )
            cat_df = cat_df[dists <= radius]
            if cat_df.empty:
                break

        if cat_df.empty:
            # Fallback: just pick from same category without proximity constraint
            cat_df = store.filter_by_category(regional_df, old_stop.category)
            cat_df = cat_df[~cat_df["name"].isin(current_names)]
            if cat_df.empty:
                raise ValueError("No alternatives available")
            logger.warning(
                "  Proximity constraint relaxed for single reshuffle")

        # Pick a random replacement
        pick = cat_df.sample(1).iloc[0]

        # Build new stop
        def _safe(val, default=""):
            """Safely convert pandas value to string, handling NA/NaN."""
            if pd.isna(val):
                return default
            return str(val)

        new_stop = ItineraryStop(
            name=_safe(pick.get("name"), "Unknown"),
            category=_safe(pick.get("primary_category")),
            latitude=pick["latitude"],
            longitude=pick["longitude"],
            locality=_safe(pick.get("locality")),
            website=_safe(pick.get("website")),
            tel=_safe(pick.get("tel")),
            email=_safe(pick.get("email")),
            facebook_id=_safe(pick.get("facebook_id")),
            instagram=_safe(pick.get("instagram")),
            twitter=_safe(pick.get("twitter")),
            digital_presence=int(pick.get("digital_presence", 0)) if pd.notna(
                pick.get("digital_presence")) else 0,
            order=old_stop.order,
        )

        # Generate tip for the new stop only
        tips = await generate_place_tips(
            stops=[new_stop],
            preset=itinerary.preset,
            region=itinerary.region,
            country=itinerary.country,
            time_slot=getattr(itinerary, "_time_slot", ""),
        )
        new_stop.tip = tips.get(new_stop.name, "")

        # Replace in itinerary
        itinerary.stops[index] = new_stop

        # Recalculate total distance
        total_dist = 0.0
        for i in range(len(itinerary.stops) - 1):
            total_dist += haversine_km(
                itinerary.stops[i].latitude, itinerary.stops[i].longitude,
                itinerary.stops[i +
                                1].latitude, itinerary.stops[i + 1].longitude,
            )
        itinerary.total_distance_km = round(total_dist, 2)

        # Update cache
        _last_generation_cache[cache_key] = itinerary

        logger.info(f"  Replaced with '{new_stop.name}'")

        return {
            "preset":            itinerary.preset,
            "preset_display":    itinerary.preset_display,
            "region":            itinerary.region,
            "country":           itinerary.country,
            "locality":          itinerary.locality,
            "user_id":           itinerary.user_id,
            "stops":             [asdict(s) for s in itinerary.stops],
            "total_distance_km": itinerary.total_distance_km,
            "time_context": {
                "hour":                getattr(itinerary, "_hour", 0),
                "time_slot":           getattr(itinerary, "_time_slot", ""),
                "nightlife_eligible":  getattr(itinerary, "_nightlife_eligible", False),
                "resolved_categories": getattr(itinerary, "_resolved_categories", {}),
            },
            "generation_meta": {
                "reshuffle_number":    _reshuffle_count.get(cache_key, 0),
                "preference_applied":  False,
                "weight_source":       "reshuffle_single",
                "excluded_place_count": len(current_names),
            },
            "reshuffled_index":  index,
            "reshuffled_stop":   asdict(new_stop),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Single reshuffle error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to reshuffle stop.")


# ── History & analytics endpoints ──────────────

@app.get("/api/history")
async def get_my_history(
    x_session_token: Optional[str] = Header(None),
):
    """Get the logged-in user's itinerary history (auto-detected from session token)."""
    user = auth_manager.require_auth(x_session_token)
    try:
        history = history_manager.get_user_history(user["user_id"])
        return {"user_id": user["user_id"], "history": history}
    except Exception as e:
        logger.error(f"History retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve history.")


@app.get("/api/history/{user_id}")
async def get_user_history_by_id(user_id: str):
    """Get a specific user's itinerary history by user_id."""
    try:
        history = history_manager.get_user_history(user_id)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        logger.error(f"History retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve history.")


@app.get("/api/analytics/preferences")
async def get_my_preferences(
    x_session_token: Optional[str] = Header(None),
):
    """Get the logged-in user's preferences (auto-detected from session token)."""
    user = auth_manager.require_auth(x_session_token)
    try:
        preferences = history_manager.get_user_preferences(user["user_id"])
        return {"user_id": user["user_id"], "preferences": preferences}
    except Exception as e:
        logger.error(f"Preferences retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve preferences.")


@app.get("/api/analytics/preferences/{user_id}")
async def get_user_preferences_by_id(user_id: str):
    """Get a specific user's preferences by user_id."""
    try:
        preferences = history_manager.get_user_preferences(user_id)
        return {"user_id": user_id, "preferences": preferences}
    except Exception as e:
        logger.error(f"Preferences retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve preferences.")


@app.get("/api/analytics/popular")
async def get_popular():
    """
    Global popularity stats across all users.

    Returns the most popular categories, presets, countries, regions,
    and localities from the last N days (default 30).

    No auth required — this is public data used to power the
    global_popular fallback in the generation engine.
    """
    try:
        days = 30
        stats = history_manager.get_popular_stats(days=days)
        return stats
    except Exception as e:
        logger.error(f"Popular stats retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve popular stats.")


# ── Auth endpoints ─────────────────────────────

@app.post("/api/auth/register", status_code=201)
async def register(username: str, email: str, password: str, role: str = "user"):
    """Register a new user."""
    try:
        user = auth_manager.register(
            username=username, email=email, password=password, role=role)
        return {"user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login(username_or_email: str, password: str):
    """Login with username or email + password."""
    try:
        result = auth_manager.login(username_or_email, password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/logout")
async def logout(x_session_token: Optional[str] = Header(None)):
    """Invalidate the current session token."""
    if not x_session_token or not auth_manager.logout(x_session_token):
        raise HTTPException(
            status_code=400, detail="Token not found or already revoked.")
    return {"detail": "Logged out successfully."}


@app.get("/api/auth/me")
async def me(x_session_token: Optional[str] = Header(None)):
    """Return the current authenticated user's profile."""
    user = auth_manager.require_auth(x_session_token)
    return {"user": user}


# ── Admin endpoints ────────────────────────────

@app.get("/api/admin/users")
async def admin_list_users(
    skip:  int = 0,
    limit: int = 100,
    x_session_token: Optional[str] = Header(None),
):
    """[ADMIN] List all registered users (paginated)."""
    auth_manager.require_admin(x_session_token)
    users = auth_manager.list_all_users(skip=skip, limit=limit)
    return {"users": users, "count": len(users)}


@app.get("/api/admin/users/{target_user_id}/history")
async def admin_user_history(
    target_user_id: str,
    x_session_token: Optional[str] = Header(None),
):
    """[ADMIN] View any user's itinerary history."""
    auth_manager.require_admin(x_session_token)
    history = history_manager.get_user_history(target_user_id)
    return {"user_id": target_user_id, "history": history}


@app.patch("/api/admin/users/{target_user_id}/role")
async def admin_set_role(
    target_user_id: str,
    body: RoleUpdateRequest,
    x_session_token: Optional[str] = Header(None),
):
    """[ADMIN] Change a user's role (user ↔ admin)."""
    auth_manager.require_admin(x_session_token)
    try:
        found = auth_manager.set_user_role(target_user_id, body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not found:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"detail": f"Role updated to '{body.role}'."}


@app.patch("/api/admin/users/{target_user_id}/active")
async def admin_set_active(
    target_user_id: str,
    body: ActiveUpdateRequest,
    x_session_token: Optional[str] = Header(None),
):
    """[ADMIN] Ban or re-enable a user account."""
    auth_manager.require_admin(x_session_token)
    found = auth_manager.set_user_active(target_user_id, body.is_active)
    if not found:
        raise HTTPException(status_code=404, detail="User not found.")
    state = "enabled" if body.is_active else "disabled"
    return {"detail": f"Account {state}."}


# ── Favorites endpoints ────────────────────────

@app.get("/api/favorites")
async def list_favorites(
    category: str = Query(None, description="Filter by category substring"),
    country:  str = Query(None, description="Filter by country (exact)"),
    region:   str = Query(None, description="Filter by region (exact)"),
    skip:     int = Query(0,    ge=0),
    limit:    int = Query(50,   ge=1, le=200),
    x_session_token: Optional[str] = Header(None),
):
    """Get the current user's saved favorites."""
    user = auth_manager.require_auth(x_session_token)
    favorites = favorites_manager.get_user_favorites(
        user_id=user["user_id"], category=category,
        country=country, region=region, skip=skip, limit=limit,
    )
    total = favorites_manager.get_favorites_count(user["user_id"])
    return {"user_id": user["user_id"], "total": total, "favorites": favorites}


@app.post("/api/favorites", status_code=201)
async def add_favorite(
    body: AddFavoriteRequest,
    x_session_token: Optional[str] = Header(None),
):
    """Save a stop to favorites."""
    user = auth_manager.require_auth(x_session_token)
    try:
        fav = favorites_manager.add_favorite(
            user_id=user["user_id"],
            name=body.name, latitude=body.latitude, longitude=body.longitude,
            fsq_place_id=body.fsq_place_id, category=body.category,
            locality=body.locality, region=body.region, country=body.country,
            website=body.website, tel=body.tel, note=body.note,
        )
        return {"favorite": fav}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.delete("/api/favorites/{favorite_id}", status_code=200)
async def remove_favorite(
    favorite_id: int,
    x_session_token: Optional[str] = Header(None),
):
    """Remove a stop from favorites by its ID."""
    user = auth_manager.require_auth(x_session_token)
    removed = favorites_manager.remove_favorite(user["user_id"], favorite_id)
    if not removed:
        raise HTTPException(
            status_code=404, detail="Favorite not found or does not belong to you.")
    return {"detail": "Removed from favorites."}


@app.patch("/api/favorites/{favorite_id}/note")
async def update_favorite_note(
    favorite_id: int,
    body: UpdateNoteRequest,
    x_session_token: Optional[str] = Header(None),
):
    """Add or update a personal note on a saved favorite."""
    user = auth_manager.require_auth(x_session_token)
    updated = favorites_manager.update_note(
        user["user_id"], favorite_id, body.note)
    if not updated:
        raise HTTPException(
            status_code=404, detail="Favorite not found or does not belong to you.")
    return {"detail": "Note updated."}


@app.get("/api/favorites/check")
async def check_favorited(
    fsq_place_id: str = Query(..., description="Foursquare place ID to check"),
    x_session_token: Optional[str] = Header(None),
):
    """Check whether a specific place is already in the user's favorites."""
    user = auth_manager.require_auth(x_session_token)
    favorited = favorites_manager.is_favorited(user["user_id"], fsq_place_id)
    return {"fsq_place_id": fsq_place_id, "favorited": favorited}


# ═══════════════════════════════════════════════
#  15. ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
