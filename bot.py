import re
import logging
import psycopg2
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

# NOTE:
# This project uses the Ethiopian *clock* system (≈ 6-hour offset from international clock),
# not just a timezone conversion. Example: 1:20 PM international ≈ 7:20 Ethiopian clock.
# We still keep TZ_ETHIOPIA for future use, but shift logic is based on the Ethiopian clock offset.
TZ_ETHIOPIA = ZoneInfo("Africa/Addis_Ababa")

# Ethiopian clock offset: EthiopianClock = InternationalClock - 6 hours
ETHIOPIAN_CLOCK_OFFSET = timedelta(hours=-6)


def to_ethiopian_clock(dt: datetime) -> datetime:
    """Convert international (PC) clock datetime to Ethiopian clock datetime (subtract 6 hours)."""
    return dt + ETHIOPIAN_CLOCK_OFFSET


def ethiopian_clock_time_to_pc_time(t: time) -> time:
    """Convert an Ethiopian clock 'time' to the PC/international clock 'time' (add 6 hours, wrap 24h)."""
    total = (t.hour * 60 + t.minute + 6 * 60) % (24 * 60)
    return time(total // 60, total % 60)


from telegram import Update, BotCommand
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from dotenv import load_dotenv
import os

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    Defaults,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
from groq import Groq
import asyncio

# ---------------- CONFIG ----------------
from dotenv import load_dotenv

# Load the .env file first
load_dotenv()

# Then access the variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = int(os.getenv("GROUP_CHAT_ID"))
EFFICIENCY_LIMIT = 75.0

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

# ---------------- AI CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-oss-120b")

ai_client = Groq(api_key=GROQ_API_KEY)

AI_SYSTEM_PROMPT = """
You are a senior production audit AI for a beverage bottling plant.

Your role:
- Detect mechanical, electrical, process, and operator risks.
- Identify repeated faults, chronic failures, and abnormal downtime.
- Identify root-cause risk signals from downtime and operator notes.
- Ask ONLY audit-grade diagnostic questions when risk exists.

Rules:
- Ask questions ONLY if downtime exists, efficiency < 75%, repeated machine faults appear,
  or pre-summary messages lack necessary details.
- Focus on: blower, molds, conveyors, bearings, sensors, alarms, rejects, VOS, power stability.
- Questions must be short, professional, numbered, and investigation-oriented.
- Do NOT summarize.
- Do NOT provide solutions.
- Treat operator comments as evidence.

Stopping rule (MANDATORY):
- If root cause is identified AND
  corrective action is completed or scheduled AND
  current status is "ready", "normal", or "no further issue",
  respond with exactly: STOP
- Do NOT ask questions about future shifts, plans, or targets.
- Do NOT continue questioning once STOP is reached.

Scope rule:
- Audit applies ONLY to the reported shift.
- Do NOT ask about next shifts, production targets, or planning.
"""

SUMMARY_SYSTEM_PROMPT = """
You are the OFFICIAL production summary AI.

Rules:
- Never invent data
- Never assume missing values
- If any required data is missing, clearly state:
  DATA INCOMPLETE – specify what is missing

Output format:

STATUS:
COMPLETE or DATA INCOMPLETE

SUMMARY:
(one professional paragraph)

PRODUCTION:
- Product:
- Plan:
- Actual:
- Efficiency:

DOWNTIME:
- Key causes

REJECTS:
- Summary

AUDIT STATUS:
- CLOSED / FOLLOW-UP REQUIRED
"""

MULTI_SHIFT_SUMMARY_SYSTEM_PROMPT = """
You are the OFFICIAL multi-shift production summary AI.

Rules:
- Never invent data
- Never assume equal plans
- Never multiply plans
- Always sum plan values exactly as provided per shift
- Always sum actual, downtime, rejects, shrink loss, and available time
- Never average efficiencies
- Always recalculate efficiency from total actual ÷ total plan
- If any required data is missing:
  DATA INCOMPLETE – specify which shift and field

Output format:

STATUS:
COMPLETE or DATA INCOMPLETE

SUMMARY:
(one professional executive paragraph analyzing combined shifts)

PRODUCTION:
- Product:
- Total Plan:
- Total Actual:
- Total Available Time:
- Aggregated Efficiency:

DOWNTIME:
- Total Downtime:
- Downtime Ratio:
- Key causes

REJECTS:
- Total Rejects:
- Category breakdown
- Shrink Loss:

AUDIT STATUS:
- CLOSED / FOLLOW-UP REQUIRED
"""

# Legacy schedule dict (no longer used for shift boundaries).
# We keep it to avoid breaking older references, but all scheduling is now derived from
# Ethiopian clock times converted to PC time via ethiopian_clock_time_to_pc_time().
SHIFT_SCHEDULE = {
    1: {"plan_time": time(0, 5), "report_time": time(6, 55)},  # Ethiopian clock
    2: {"plan_time": time(7, 5), "report_time": time(13, 55)},  # Ethiopian clock
    3: {"plan_time": time(14, 5), "report_time": time(23, 55)},  # Ethiopian clock
}
# ---------------- AI SUMMARY EVIDENCE ----------------
ai_shift_evidence = {
    1: [],
    2: [],
    3: []
}

# Store AI text summaries per shift for full-day aggregation
daily_ai_shift_summaries = {
    1: None,
    2: None,
    3: None,
}
# ---------------- SHIFT / REMINDER STATE ----------------
current_shift = 1  # starts at shift 1
shift_closed = {
    1: False,
    2: False,
    3: False
}

# Line / sanitation / AI reminder gating
LINE_STATE_RUNNING = "running"
LINE_STATE_OFF = "line_off"
LINE_STATE_SANITATION = "sanitation"

line_state = LINE_STATE_RUNNING
ai_reminder_block = False  # True while deep AI audit is active
pending_reminders = []  # queued reminders while muted
daily_plan_last_date = None  # date of last daily production plan reminder sent
# At the top of your file, with your other globals
active_validation_session_key: str | None = None
# Track shift plan reminders sent per shift per day
shift_plan_sent_today = {
    1: None,  # date when sent, or None
    2: None,
    3: None,
}

# Track when line went off (for calculating partial hours)
line_off_since = None  # datetime when line went off

# Suppression state for line-off / sanitation-on behavior:
# After line goes OFF or sanitation starts, allow exactly ONE more scheduled reminder,
# then suppress all remaining hourly reminders until line is ON again.
line_off_next_reminder_allowed = True  # True = next reminder may fire; False = suppress
line_off_one_reminder_fired = False  # True = the one allowed reminder already fired
shift_had_production = {1: False, 2: False, 3: False}  # per shift, any production before OFF?

# ---------------- PRODUCTION VALIDATION STATE ----------------
# Tracks per-user validation sessions that BLOCK summaries until resolved
# Key: user_id or "shift_{shift}" or "hourly_{shift}_{hour}"
# Value: dict with validation state
validation_sessions = {}

MAX_VALIDATION_ROUNDS = 4  # Max back-and-forth before forcing verdict

VALIDATION_ACCEPT_KEYWORDS = [
    "accepted", "approved", "validated", "convincing",
    "explanation accepted", "justified",
]

VALIDATION_STATE_PENDING = "pending"        # Questions asked, waiting for answer
VALIDATION_STATE_APPROVED = "approved"      # Operator's answer was convincing
VALIDATION_STATE_REJECTED = "rejected"      # Operator failed to justify — unaccounted loss
VALIDATION_STATE_FOLLOWUP = "followup"      # AI asking follow-up questions

def format_date_time_12h(dt: datetime) -> str:
    """Format as dd/mm/yyyy, h:mm AM/PM (12-hour). Converts to Ethiopia if timezone-aware."""
    if dt.tzinfo:
        dt = dt.astimezone(TZ_ETHIOPIA)
    date_str = dt.strftime("%d/%m/%Y")
    hour_12 = dt.hour % 12 or 12
    am_pm = "AM" if dt.hour < 12 else "PM"
    time_str = f"{hour_12}:{dt.minute:02d} {am_pm}"
    return f"{date_str}, {time_str}"


def format_hour_range_12h(start_hour: int) -> str:
    """Format hour range as 12-hour AM/PM (e.g., '12:00 AM–1:00 AM')."""
    end_hour = (start_hour + 1) % 24
    start_12 = start_hour % 12 or 12
    start_am_pm = "AM" if start_hour < 12 else "PM"
    end_12 = end_hour % 12 or 12
    end_am_pm = "AM" if end_hour < 12 else "PM"
    return f"{start_12}:00 {start_am_pm}–{end_12}:00 {end_am_pm}"


def get_shift_duration_minutes(shift: int) -> int:
    """Get default shift duration in minutes based on shift number."""
    if shift == 1:  # 1st shift = 5 hours
        return 5 * 60
    elif shift == 2:  # 2nd shift = 7 hours
        return 7 * 60
    else:  # Shift 3: 3rd shift = 10 hours
        return 10 * 60


def get_default_production_hours(report_type: str, shift: int = None) -> float:
    """Get default production hours based on report type."""
    if report_type == "hourly":
        return 1.0  # 1 hour for hourly summaries
    elif report_type == "multi_shift":
        return 22.0  # 22 hours total for multi-shift (2 hours for PM)
    elif report_type == "shift" and shift:
        if shift == 1:
            return 5.0  # 5 hours for shift 1
        elif shift == 2:
            return 7.0  # 7 hours for shift 2
        else:  # shift 3
            return 10.0  # 10 hours for shift 3
    else:
        return 1.0  # Default to 1 hour


# ---------------- LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------------- DATABASE ----------------
def get_db_connection():
    """Get a fresh database connection"""
    return psycopg2.connect(**DB_CONFIG)


def get_clean_db_connection():
    """Get a clean database connection, ensuring no aborted transactions"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # Test if connection is clean by running a simple query
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return conn
    except Exception:
        # If connection is in aborted state, close and create new one
        try:
            conn.close()
        except:
            pass
        return psycopg2.connect(**DB_CONFIG)


def _ensure_bot_state_table():
    """Create bot_state table if it does not exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Could not ensure bot_state table: {e}")
    finally:
        cur.close()
        conn.close()


def bot_state_get(key: str) -> str | None:
    """Get value for a bot_state key. Returns None if not set."""
    _ensure_bot_state_table()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT value FROM bot_state WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"bot_state_get failed: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def bot_state_set(key: str, value: str) -> None:
    """Set value for a bot_state key."""
    _ensure_bot_state_table()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO bot_state (key, value, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
        """, (key, value))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"bot_state_set failed: {e}")
    finally:
        cur.close()
        conn.close()


def load_bot_state_from_db() -> None:
    """Load daily_plan_last_date, shift_plan_sent_today, line_state, line_off_since from DB."""
    global daily_plan_last_date, shift_plan_sent_today, line_state, line_off_since
    try:
        v = bot_state_get("daily_plan_last_date")
        if v:
            daily_plan_last_date = datetime.strptime(v, "%Y-%m-%d").date()
        for i in (1, 2, 3):
            v = bot_state_get(f"shift_plan_sent_{i}")
            if v:
                shift_plan_sent_today[i] = datetime.strptime(v, "%Y-%m-%d").date()
        v = bot_state_get("line_state")
        if v and v in (LINE_STATE_RUNNING, LINE_STATE_OFF, LINE_STATE_SANITATION):
            line_state = v
        # line_off_since is not loaded (stays None after reboot) so partial-hour logic only uses current session
        logger.info("Loaded bot state from database")
    except Exception as e:
        logger.warning(f"load_bot_state_from_db: {e}")


def parse_vos(text: str):
    """
    Extracts VOS line like:
    vos=line cleaning=40'
    Returns cleaned string or None.
    """
    if not text:
        return None

    # Case-insensitive match
    match = re.search(r"vos\s*=\s*(.+)", text, re.IGNORECASE)

    if not match:
        return None

    vos_line = match.group(1).strip()

    # Remove trailing quote if exists
    vos_line = vos_line.replace("'", "").strip()

    return vos_line if vos_line else None


def save_to_database(data, downtime, rejects, vos_info=None, shift_override: int | None = None):
    """Save production data. Uses shift_override if provided (for /shift_summary_N)."""
    conn = get_db_connection()
    cur = conn.cursor()
    shift = shift_override if shift_override is not None else data["shift"]

    SHIFT_DEFAULT_MINUTES = {1: 300, 2: 420, 3: 600}
    available_time = data.get("available_time")
    if available_time is None:
        available_time = SHIFT_DEFAULT_MINUTES.get(shift, 420)

    try:
        cur.execute("""
            INSERT INTO production
            (date, shift, product_type, shift_plan_pack, actual_output_pack, vos_info, available_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, shift) DO UPDATE SET
                product_type = EXCLUDED.product_type,
                shift_plan_pack = EXCLUDED.shift_plan_pack,
                actual_output_pack = EXCLUDED.actual_output_pack,
                vos_info = EXCLUDED.vos_info,
                available_time = EXCLUDED.available_time
            RETURNING id
        """, (
            data["date"],
            shift,
            data["product_type"],
            data["plan"],
            data["actual"],
            vos_info,
            available_time,
        ))
        production_id = cur.fetchone()[0]

        cur.execute("DELETE FROM downtime_events WHERE production_id = %s", (production_id,))
        cur.execute("DELETE FROM rejects WHERE production_id = %s", (production_id,))

        for d in downtime:
            cur.execute("""
                INSERT INTO downtime_events
                (production_id, description, duration_min, category)
                VALUES (%s, %s, %s, %s)
            """, (
                production_id,
                d["description"],
                d["duration"],
                d.get("category", "MECHANICAL"),
            ))

        cur.execute("""
            INSERT INTO rejects
            (production_id, preform, bottle, cap, label, shrink)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            production_id,
            rejects["preform"],
            rejects["bottle"],
            rejects["cap"],
            rejects["label"],
            rejects["shrink"],
        ))

        conn.commit()
        logger.info(
            f"Saved shift {shift} to DB — available_time={available_time} min "
            f"({'default' if data.get('available_time') is None else 'provided'})"
        )
        return production_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def _ensure_hourly_production_table():
    """Create hourly_production + related tables if they don't exist.
    Also migrates old schemas (drops stale NOT NULL columns the code doesn't use)."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # 1. Create tables if brand new
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hourly_production (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                shift INTEGER NOT NULL,
                hour_number INTEGER NOT NULL,
                product_type TEXT,
                hour_plan INTEGER DEFAULT 0,
                actual_output INTEGER DEFAULT 0,
                available_time_minutes INTEGER DEFAULT 60,
                vos_info TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, shift, hour_number)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hourly_downtime_events (
                id SERIAL PRIMARY KEY,
                hourly_production_id INTEGER REFERENCES hourly_production(id) ON DELETE CASCADE,
                description TEXT,
                duration_min INTEGER,
                category TEXT DEFAULT 'MECHANICAL'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hourly_rejects (
                id SERIAL PRIMARY KEY,
                hourly_production_id INTEGER REFERENCES hourly_production(id) ON DELETE CASCADE,
                preform INTEGER DEFAULT 0,
                bottle INTEGER DEFAULT 0,
                cap INTEGER DEFAULT 0,
                label INTEGER DEFAULT 0,
                shrink REAL DEFAULT 0
            )
        """)

        # 2. Schema migration: fix old columns that the code doesn't populate
        #    If hour_start or hour_end exist with NOT NULL, make them nullable
        #    so INSERT doesn't fail when we don't provide values.
        cur.execute("""
            SELECT column_name, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'hourly_production'
              AND column_name IN ('hour_start', 'hour_end')
        """)
        old_columns = cur.fetchall()

        for col_name, is_nullable in old_columns:
            if is_nullable == 'NO':
                cur.execute(
                    f"ALTER TABLE hourly_production ALTER COLUMN {col_name} DROP NOT NULL"
                )
                logger.info(f"Schema migration: made hourly_production.{col_name} nullable")

        # 3. Ensure hour_number column exists (old schema might not have it)
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'hourly_production' AND column_name = 'hour_number'
        """)
        if not cur.fetchone():
            cur.execute("""
                ALTER TABLE hourly_production ADD COLUMN hour_number INTEGER
            """)
            logger.info("Schema migration: added hour_number column to hourly_production")

        # 4. Ensure UNIQUE constraint on (date, shift, hour_number) exists
        cur.execute("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'hourly_production'
              AND constraint_type = 'UNIQUE'
        """)
        constraints = [row[0] for row in cur.fetchall()]

        # Check if any unique constraint covers (date, shift, hour_number)
        has_correct_unique = False
        for cname in constraints:
            cur.execute("""
                SELECT column_name
                FROM information_schema.constraint_column_usage
                WHERE constraint_name = %s
            """, (cname,))
            cols = {row[0] for row in cur.fetchall()}
            if cols == {'date', 'shift', 'hour_number'}:
                has_correct_unique = True
                break

        if not has_correct_unique:
            # Drop old unique constraints that might conflict
            for cname in constraints:
                try:
                    cur.execute(f"ALTER TABLE hourly_production DROP CONSTRAINT IF EXISTS {cname}")
                    logger.info(f"Schema migration: dropped old constraint {cname}")
                except Exception:
                    pass

            try:
                cur.execute("""
                    ALTER TABLE hourly_production
                    ADD CONSTRAINT hourly_production_date_shift_hour_number_key
                    UNIQUE (date, shift, hour_number)
                """)
                logger.info("Schema migration: added UNIQUE(date, shift, hour_number)")
            except Exception as e:
                logger.warning(f"Could not add unique constraint (may already exist): {e}")

        conn.commit()
        logger.info("Hourly production tables ensured (with schema migration)")
    except Exception as e:
        conn.rollback()
        logger.error(f"_ensure_hourly_production_table failed: {e}", exc_info=True)
    finally:
        cur.close()
        conn.close()


def save_hourly_to_database(data: dict, downtime: list, rejects: dict,
                            hour_number: int, vos_info: str = None,
                            shift_override: int = None) -> int | None:
    """
    Save ONE HOUR of production data to hourly_production table.
    Same pattern as save_to_database but for hourly granularity.
    """
    _ensure_hourly_production_table()
    conn = get_clean_db_connection()
    cur = conn.cursor()
    shift = shift_override if shift_override is not None else data["shift"]
    available_time = data.get("available_time") or 60

    try:
        cur.execute("""
            INSERT INTO hourly_production
            (date, shift, hour_number, product_type, hour_plan, actual_output,
             available_time_minutes, vos_info)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, shift, hour_number) DO UPDATE SET
                product_type = EXCLUDED.product_type,
                hour_plan = EXCLUDED.hour_plan,
                actual_output = EXCLUDED.actual_output,
                available_time_minutes = EXCLUDED.available_time_minutes,
                vos_info = EXCLUDED.vos_info
            RETURNING id
        """, (
            data["date"], shift, hour_number,
            data["product_type"], data["plan"], data["actual"],
            available_time, vos_info,
        ))
        hourly_id = cur.fetchone()[0]

        cur.execute("DELETE FROM hourly_downtime_events WHERE hourly_production_id = %s", (hourly_id,))
        cur.execute("DELETE FROM hourly_rejects WHERE hourly_production_id = %s", (hourly_id,))

        for d in downtime:
            cur.execute("""
                INSERT INTO hourly_downtime_events
                (hourly_production_id, description, duration_min, category)
                VALUES (%s, %s, %s, %s)
            """, (hourly_id, d["description"], d["duration"], d.get("category", "MECHANICAL")))

        cur.execute("""
            INSERT INTO hourly_rejects
            (hourly_production_id, preform, bottle, cap, label, shrink)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            hourly_id,
            rejects.get("preform", 0), rejects.get("bottle", 0),
            rejects.get("cap", 0), rejects.get("label", 0),
            rejects.get("shrink", 0),
        ))

        conn.commit()
        logger.info(f"Saved hourly data: shift={shift} hour={hour_number} to DB (id={hourly_id})")
        return hourly_id
    except Exception as e:
        conn.rollback()
        logger.error(f"save_hourly_to_database failed: {e}")
        return None
    finally:
        cur.close()
        conn.close()


# ---------------- PARSING ----------------
def parse_report(text: str):
    t = text.lower()
    date_match = re.search(r"date\s*(\d{1,2}/\d{1,2}/\d{2,4})", t)
    date = datetime.strptime(date_match.group(1), "%d/%m/%y").date() if date_match else now_ethiopia().date()
    shift_match = re.search(r"shift\s*(?:=)?\s*(1st|2nd|3rd)", t)
    if not shift_match:
        raise ValueError("Shift not found")
    shift = {"1st": 1, "2nd": 2, "3rd": 3}[shift_match.group(1)]
    product_match = re.search(r"product type\s*(.+)", t)
    product_type = product_match.group(1).strip() if product_match else None
    plan_match = re.search(r"shift plan\s*=\s*([\d,]+)", t)
    if not plan_match:
        raise ValueError("Shift plan missing")
    plan = int(plan_match.group(1).replace(",", ""))
    actual_match = re.search(r"actual(?:\s+output)?\s*=\s*([\d,]+)", t)
    if not actual_match:
        raise ValueError("Actual output missing")
    actual = int(actual_match.group(1).replace(",", ""))

    # Parse Available Time (machine active time in minutes)
    # Look for patterns like "available time = 420" or "available = 420 min" or "avail = 420"
    available_time_match = re.search(r"available(?:\s+time)?\s*=\s*(\d+)", t)
    available_time = None
    if available_time_match:
        available_time = int(available_time_match.group(1))
    # If not found, return None - calling functions will set their own defaults
    # Inside parse_report(), before the return statement:

    # Calculate efficiency based on available time
    efficiency = round((actual / plan) * 100, 1) if plan else 0
    vos_match = re.search(r"vos\s*=\s*([\d,]+)", t)
    vos = int(vos_match.group(1).replace(",", "")) if vos_match else None
    return {
        "date": date,
        "shift": shift,
        "product_type": product_type,
        "plan": plan,
        "actual": actual,
        "efficiency": efficiency,
        "available_time": available_time,
        "vos": vos
    }


import re

import re


def parse_downtime(text: str):
    events = []
    t = text.lower()

    # Lines that are never downtime — skip them regardless of content
    SKIP_PREFIXES = (
        "shift plan", "actual output", "actual =", "actual=",
        "product type", "shift ", "date ",
        "preform =", "preform=",
        "bottle =", "bottle=",
        "cap =", "cap=",
        "label =", "label=",
        "shrink =", "shrink=",
        "available time =", "available time=", "available =", "available=",
        "vos =", "vos=",
        "efficiency =", "efficiency=",
    )

    lines = t.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip known non-downtime field lines
        if any(line.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue

        # Skip pure date lines (e.g., 24/02/26 or 24-02-2026)
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', line):
            continue

        # Skip lines that are ONLY a number (standalone values like "3120")
        if re.match(r'^\d+(\.\d+)?$', line):
            continue

        # Only match lines with an explicit time unit (min/minutes/') — never bare numbers
        # This prevents "preform 121" or "actual 3120" from being treated as downtime
        duration_patterns = [
            r"(\d+)\s*(?:min|minutes?|')\s*$",  # ends with min/minutes/'
            r"(\d+)\s*(?:min|minutes?|')\s",  # min/minutes/' in middle
        ]

        for pattern in duration_patterns:
            match = re.search(pattern, line)
            if match:
                duration = int(match.group(1))
                # Extract description by removing the duration part
                desc = re.sub(r"\d+\s*(?:min|minutes?|')\s*$", "", line).strip()
                desc = desc.replace("vos", "").replace("=", "").strip()
                if len(desc) > 3:
                    events.append({"description": desc, "duration": duration})
                break

    return events


def parse_vos(text: str):
    """Parse vos (line off) information separately"""
    t = text.lower()
    vos_info = None

    # Look for vos line
    vos_match = re.search(r"vos\s*=\s*(.+)", t)
    if vos_match:
        vos_info = vos_match.group(1).strip()

    return vos_info


def parse_rejects(text: str):
    t = text.lower()

    def int_val(pattern):
        m = re.search(pattern, t)
        return int(m.group(1).replace(",", "")) if m else 0

    def float_val(pattern):
        m = re.search(pattern, t)
        return float(m.group(1)) if m else 0

    return {
        "preform": int_val(r"preform\s*=\s*([\d,]+)"),
        "bottle": int_val(r"bottle\s*=\s*([\d,]+)"),
        "cap": int_val(r"cap\s*=\s*([\d,]+)"),
        "label": int_val(r"(?:label|lable)\s*=\s*([\d,]+)"),
        "shrink": float_val(r"shrink\s*=\s*([\d.]+)")
    }


def parse_operator_notes(text: str):
    t = text.lower()
    keywords = ["going on", "still", "replacement", "repair", "adjustment",
                "alarm", "closing pb", "orbit alarm", "seal", "clutch",
                "bearing", "blower", "mold", "conveyor"]
    return [line.strip() for line in t.split("\n") if any(k in line for k in keywords)]


def detect_repeated_faults(downtime, operator_notes):
    combined = " ".join(d["description"] for d in downtime) + " " + " ".join(operator_notes)
    machines = ["blower", "mold", "conveyor", "clutch", "bearing"]
    return [f"{m} repeated {combined.count(m)} times" for m in machines if combined.count(m) >= 2]


# ---------------- KPI CALCULATION ENGINE ----------------
def compute_kpis(
        plan: int,
        actual: int,
        downtime_minutes: int,
        production_hours: float,
        rejects: dict
) -> dict:
    """
    Deterministic KPI calculation engine.
    Returns standardized KPI metrics for all report types.
    """
    # 1️⃣ Performance % - (Actual Production / Planned Production) * 100
    performance = round((actual / plan) * 100, 1) if plan > 0 else 0.0

    # 2️⃣ Machine Availability % - ((Planned Production hrs - Machine Downtime) * 100) / Production hrs
    downtime_hours = downtime_minutes / 60
    availability = round(((production_hours - downtime_hours) / production_hours) * 100,
                         1) if production_hours > 0 else 0.0

    # 3️⃣ Quality %
    defective_qty = rejects.get("preform", 0) + rejects.get("bottle", 0)
    quality = round((actual / (actual + defective_qty)) * 100, 1) if (actual + defective_qty) > 0 else 0.0

    # 4️⃣ OEE
    oee = round((performance * availability * quality) / 10000, 2)

    # 5️⃣ Reject Percentages
    reject_percentages = {}
    for category in ["preform", "bottle", "cap", "label", "shrink"]:
        reject_qty = rejects.get(category, 0)
        reject_pct = round((reject_qty / (reject_qty + actual)) * 100, 1) if (reject_qty + actual) > 0 else 0.0
        reject_percentages[category] = reject_pct

    return {
        "performance": performance,
        "availability": availability,
        "quality": quality,
        "oee": oee,
        "defective_qty": defective_qty,
        "downtime_hours": downtime_hours,
        "reject_percentages": reject_percentages
    }


# ---------------- AI SESSION ----------------
user_ai_sessions = {}
active_users = set()
MAX_AI_QUESTIONS = 6
user_audit_state = {}
READY_KEYWORDS = ["all are ready", "ready to produce", "production ready", "normal", "completed",
                  "issue resolved", "replacement completed", "we are ready", "no further issue"]


def audit_should_stop(user_id: int, message_text: str) -> bool:
    text = message_text.lower()
    if user_id not in user_audit_state:
        user_audit_state[user_id] = {"questions": 0, "completed": False}
    if any(k in text for k in READY_KEYWORDS):
        user_audit_state[user_id]["completed"] = True
        return True
    if user_audit_state[user_id]["questions"] >= MAX_AI_QUESTIONS:
        user_audit_state[user_id]["completed"] = True
        return True
    return False


async def generate_ai_questions_for_message(user_id, message_text):
    if user_id not in user_ai_sessions:
        user_ai_sessions[user_id] = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
    prompt = f"""
Operator message:
{message_text}

Rules:
- Ask only ONE concise, numbered audit-grade diagnostic question if details are missing.
- Do not summarize, do not give solutions.
- Focus on potential risks, repeated faults, abnormal conditions.
- Limit strictly to 1 question.
"""
    user_ai_sessions[user_id].append({"role": "user", "content": prompt})
    try:
        response = ai_client.chat.completions.create(model=AI_MODEL, messages=user_ai_sessions[user_id])
        ai_msg = response.choices[0].message.content.strip()
        if ai_msg.upper().strip() == "STOP":
            user_ai_sessions[user_id].append({"role": "assistant", "content": "STOP"})
            return "STOP"
        first_question = ""
        for line in ai_msg.split("\n"):
            if re.match(r"^\d+\.", line.strip()):
                first_question = line.strip()
                break
        if not first_question and ai_msg:
            first_question = ai_msg
        user_ai_sessions[user_id].append({"role": "assistant", "content": first_question})
        return first_question
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return None


# ---------------- SHIFT CALCULATION (BY CLOCK) ----------------
def now_ethiopia() -> datetime:
    """
    Current time in Ethiopia (Africa/Addis_Ababa) for shift logic and scheduling.
    Ensures bot_status, reminders, and job queue all use the same clock.
    """
    return datetime.now(TZ_ETHIOPIA)


def get_shift_for_time(dt: datetime | None = None) -> int:
    """
    Map PC/international wall-clock time to shift number using Ethiopian clock shift model.

    Ethiopian shifts (Ethiopian clock):
    - Shift 1: 12:00 – 7:00  (7 hours)
    - Shift 2: 7:00  – 2:00  (7 hours)
    - Shift 3: 2:00  – 12:00 (10 hours)

    Converted to PC/international clock (add 6 hours):
    - Shift 1: 06:00 – 13:00
    - Shift 2: 13:00 – 20:00
    - Shift 3: 20:00 – 06:00 (next day)
    """
    if dt is None:
        dt = now_ethiopia()
    t = dt.time()
    if time(6, 0) <= t < time(13, 0):
        return 1
    if time(13, 0) <= t < time(20, 0):
        return 2
    return 3


async def send_or_queue_reminder(
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        parse_mode: str | None = "Markdown",
) -> str:
    """
    Central dispatch for all scheduled reminders.

    Suppression rules (line OFF / sanitation ON):
    ─────────────────────────────────────────────
    CASE 1 — Line was OFF or sanitation ON at shift start, never changed:
        Never send any reminder (entire shift ignored).
        Detected by: line state non-running AND shift_had_production[shift] is False.

    CASE 2 — Line turned OFF during shift:
        Allow exactly ONE next scheduled reminder after the OFF event.
        After that reminder fires, suppress all remaining hourly reminders.
        Shift summary is NEVER suppressed (production occurred).

    CASE 3 — OFF near shift end (final hour has both hourly summary + shift summary):
        Both must execute. Shift summary is never suppressed.

    CASE 4 — Line ON: all reminders fire normally.

    AI audit block: queues ALL reminders (flush on audit end).
    """
    global pending_reminders, line_off_next_reminder_allowed, line_off_one_reminder_fired

    now = now_ethiopia()
    shift_now = get_shift_for_time(now)
    date_now = now.date()

    # ── Classify reminder type ──────────────────────────────────────────────
    is_shift_summary = (
            "Summary Report Reminder" in text and "Hourly" not in text
    )
    is_hourly_summary = "Hourly Summary Reminder" in text
    is_any_summary = is_shift_summary or is_hourly_summary

    is_planning_reminder = (
            "Daily Production Plan Reminder" in text
            or ("Plan Reminder" in text and "Hourly" not in text)
            or "Hourly Plan Reminder" in text
    )

    # ── Line state checks ───────────────────────────────────────────────────
    line_is_inactive = line_state != LINE_STATE_RUNNING

    if line_is_inactive:
        # CASE 1: Line was OFF/sanitation ON the ENTIRE shift (no production at all).
        # Suppress everything including summaries.
        if not shift_had_production.get(shift_now, False):
            logger.info(
                f"[SUPPRESS-CASE1] Entire shift inactive, no production — "
                f"suppressing: Shift {shift_now} | {text[:60]}"
            )
            return "suppressed"

        # CASE 2/3: Production occurred before the OFF event.
        # Shift summary must NEVER be suppressed.
        if is_shift_summary:
            # Always let shift summary through — production occurred.
            logger.info(
                f"[ALLOW-SHIFT-SUMMARY] Shift {shift_now} summary allowed "
                f"(production occurred before OFF)"
            )
            # Fall through to send below.

        elif is_hourly_summary:
            # Hourly summary: apply one-reminder rule.
            # Allow the FIRST one after OFF, suppress the rest.
            if line_off_next_reminder_allowed and not line_off_one_reminder_fired:
                line_off_one_reminder_fired = True
                line_off_next_reminder_allowed = False
                logger.info(
                    f"[ALLOW-ONE] First reminder after OFF — "
                    f"Shift {shift_now} hourly summary allowed"
                )
                # Fall through to send below.
            else:
                logger.info(
                    f"[SUPPRESS-CASE2] Post-one-reminder suppression — "
                    f"Shift {shift_now} | {text[:60]}"
                )
                return "suppressed"

        elif is_planning_reminder:
            # Planning reminders: always suppressed when line is inactive.
            # Exception: if this is the ONE allowed reminder slot and nothing has fired yet,
            # let a plan reminder through (e.g. OFF happened right before :02 plan time).
            if line_off_next_reminder_allowed and not line_off_one_reminder_fired:
                line_off_one_reminder_fired = True
                line_off_next_reminder_allowed = False
                logger.info(
                    f"[ALLOW-ONE] First reminder after OFF (plan) — "
                    f"Shift {shift_now} allowed"
                )
                # Fall through to send below.
            else:
                logger.info(
                    f"[SUPPRESS] Planning reminder suppressed (line {line_state}): "
                    f"Shift {shift_now}"
                )
                return "suppressed"

        else:
            # Unknown reminder type while line inactive — suppress.
            logger.info(f"[SUPPRESS] Unknown type while line inactive, suppressing")
            return "suppressed"

    # ── AI audit block: queue (but never drop) ─────────────────────────────
    if ai_reminder_block:
        pending_reminders.append(
            {
                "text": text,
                "parse_mode": parse_mode,
                "created_at": now,
                "shift": shift_now,
                "date": date_now,
                "mute_type": "ai",
            }
        )
        logger.info(
            f"Reminder queued (AI muted): Shift {shift_now} at {now.strftime('%H:%M:%S')}"
        )
        return "queued"

    # ── Send immediately ────────────────────────────────────────────────────
    logger.info(
        f"Sending reminder to group: Shift {shift_now} at {now.strftime('%H:%M:%S')}"
    )
    try:
        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=text,
            parse_mode=parse_mode,
        )
        return "sent"
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return "failed"


async def flush_pending_reminders(bot, reason: str | None = None) -> None:
    """
    Flush queued reminders.
    - reason="ai":   send ALL AI-muted reminders regardless of time/shift
    - reason="line": send AI-muted reminders only; drop all line-muted items (no backlog)
    """
    global pending_reminders

    if not pending_reminders:
        return

    to_send = []
    remaining = []

    if reason == "ai":
        for item in pending_reminders:
            if item.get("mute_type") == "ai":
                to_send.append(item)
            else:
                remaining.append(item)
    else:
        # reason="line" or default:
        # Send AI-muted items, drop ALL line-muted items (no backlog for planning reminders)
        for item in pending_reminders:
            if item.get("mute_type") == "ai":
                to_send.append(item)
            # line-muted items are intentionally dropped here — no else/remaining

    pending_reminders = remaining  # only non-AI items remain (empty for reason="line")

    if not to_send:
        return

    def _reminder_priority(item):
        text = item.get("text", "")
        if "Daily Production Plan" in text:
            return 0
        if "Plan Reminder" in text and "Hourly" not in text:
            return 1  # Shift plan
        if "Summary Report Reminder" in text:
            return 2  # Shift summary
        if "Hourly Plan" in text:
            return 3
        if "Hourly Summary" in text:
            return 4
        return 5

    to_send.sort(key=lambda x: (
        _reminder_priority(x),
        x.get("created_at", datetime.min)
    ))

    for item in to_send:
        try:
            await bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=item["text"],
                parse_mode=item.get("parse_mode"),
            )
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"flush_pending_reminders: failed to send item: {e}")


# ---------------- RECONNECTION / MISSED REMINDER RECOVERY ----------------
_last_successful_send: datetime | None = None
_recovery_task_running = False


async def recover_missed_reminders_on_reconnect(app) -> None:
    """
    Called when internet reconnects. Sends ONLY reminders still inside their
    valid time windows. If current time is outside a reminder's window, skip
    permanently (no late sends).

    Line state rules:
    - CASE 1: Line OFF/sanitation the ENTIRE shift (no production at all)
              → suppress EVERYTHING including summaries
    - CASE 2: Line turned OFF during shift (production occurred before OFF)
              → planning reminders suppressed, summaries still fire
    - Line running normally → all reminders fire normally
    """
    now = now_ethiopia()
    today_iso = now.date().isoformat()
    current_shift_num = get_shift_for_time(now)
    current_minutes = now.hour * 60 + now.minute
    if current_shift_num == 3 and now.hour < 6:
        current_minutes += 24 * 60
    shift_start_minutes = {1: 6 * 60, 2: 13 * 60, 3: 20 * 60}
    start = shift_start_minutes[current_shift_num]
    if current_shift_num == 3 and now.hour < 6:
        start = 20 * 60
    minutes_into_shift = current_minutes - start

    line_is_active = (line_state == LINE_STATE_RUNNING)

    # Check if this shift had ANY production at all
    # Checks both in-memory (survives within session) and DB (survives restart)
    shift_has_production = (
            shift_had_production.get(current_shift_num, False)
            or _shift_had_any_production(current_shift_num, today_iso)
    )

    logger.info(
        f"[RECOVERY] Reconnected at {now.strftime('%H:%M')} "
        f"Shift {current_shift_num} | line_state={line_state} | "
        f"shift_has_production={shift_has_production}"
    )

    # ── CASE 1: Line OFF entire shift with zero production ───────────────────
    # Suppress absolutely everything — no reminders at all
    if not line_is_active and not shift_has_production:
        logger.info(
            f"[RECOVERY] CASE 1: Line OFF entire shift, no production — "
            f"suppressing all recovery reminders for Shift {current_shift_num}"
        )
        return

    sent_count = 0

    # ── 1. Daily Plan ────────────────────────────────────────────────────────
    # Planning reminder — only send if line is currently active
    if line_is_active:
        if (not bot_state_get(f"daily_plan_{today_iso}")
                and not bot_state_get(f"daily_plan_catchup_{today_iso}")):
            header = f"📅 {format_date_time_12h(now)}\n\n"
            text = (
                    header
                    + "📆 *Daily Production Plan Reminder* _(missed — reconnected)_\n\n"
                    + "Please share today's overall production plan:\n"
                    + "- Products and SKUs by shift\n"
                    + "- Target packs per shift\n"
                    + "- Any known constraints (utilities, materials, manpower)."
            )
            try:
                await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
                bot_state_set(f"daily_plan_{today_iso}", "1")
                bot_state_set(f"daily_plan_catchup_{today_iso}", "1")
                global daily_plan_last_date
                daily_plan_last_date = now.date()
                bot_state_set("daily_plan_last_date", today_iso)
                sent_count += 1
                await asyncio.sleep(1)
                logger.info("[RECOVERY] Daily plan sent")
            except Exception as e:
                logger.error(f"[RECOVERY] Daily plan failed: {e}")
    else:
        logger.info("[RECOVERY] Daily plan skipped — line OFF/sanitation (CASE 2)")

    # ── 2. Shift Plan ────────────────────────────────────────────────────────
    # Planning reminder — only send if line is currently active
    if line_is_active:
        recovery_key = f"shift_plan_recovery_{today_iso}_{current_shift_num}"
        fired_key = f"shift_plan_fired_{today_iso}_{current_shift_num}"
        catch_key = f"shift_plan_catchup_{today_iso}_{current_shift_num}"
        if (not bot_state_get(recovery_key)
                and not bot_state_get(fired_key)
                and not bot_state_get(catch_key)):
            header = f"📅 {format_date_time_12h(now)}\n\n"
            text = (
                    header
                    + f"📋 *Shift {current_shift_num} Plan Reminder* _(missed)_\n\n"
                    + "- Product type\n"
                    + "- Shift plan (packs)\n"
                    + "- Expected manpower / constraints"
            )
            try:
                await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
                bot_state_set(recovery_key, "1")
                bot_state_set(fired_key, "1")
                global shift_plan_sent_today
                shift_plan_sent_today[current_shift_num] = now.date()
                sent_count += 1
                await asyncio.sleep(1)
                logger.info(f"[RECOVERY] Shift {current_shift_num} plan sent")
            except Exception as e:
                logger.error(f"[RECOVERY] Shift plan failed: {e}")
    else:
        logger.info(
            f"[RECOVERY] Shift {current_shift_num} plan skipped "
            f"— line OFF/sanitation (CASE 2)"
        )

    # ── 3. Hourly Plan ───────────────────────────────────────────────────────
    # Planning reminder — only send if line is currently active
    current_hour_num = get_current_hour_number(current_shift_num, now)
    if line_is_active:
        sched_key = f"hourly_plan_scheduled_{today_iso}_{current_shift_num}_{current_hour_num}"
        catch_key = f"hourly_plan_{today_iso}_{current_shift_num}_{current_hour_num}"
        # Only check catch_key — sched_key may have been written while line was OFF
        # (scheduler ran but message was suppressed), so it's not a reliable sent indicator
        if (not bot_state_get(catch_key)
                and is_in_hourly_plan_window(current_shift_num, current_hour_num, now)):
            header = f"📅 {format_date_time_12h(now)}\n\n"
            text = (
                    header
                    + f"⏰ *Hourly Plan – Shift {current_shift_num}, Hour {current_hour_num}*"
                    + " _(missed — reconnected)_\n\n"
                    + "Please share the plan for this hour:\n"
                    + "- Production target\n"
                    + "- Any scheduled maintenance or adjustments\n"
                    + "- Expected challenges"
            )
            try:
                await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
                bot_state_set(sched_key, "1")
                bot_state_set(catch_key, "1")
                sent_count += 1
                await asyncio.sleep(1)
                logger.info(
                    f"[RECOVERY] Hourly plan Shift {current_shift_num} "
                    f"Hr {current_hour_num} sent"
                )
            except Exception as e:
                logger.error(f"[RECOVERY] Hourly plan failed: {e}")
    else:
        logger.info(
            f"[RECOVERY] Hourly plan Shift {current_shift_num} Hr {current_hour_num} "
            f"skipped — line OFF/sanitation (CASE 2)"
        )

    # ── 4. Hourly Summary ────────────────────────────────────────────────────
    # Summary reminder — only send if production occurred this shift
    # (CASE 1 with no production already returned early above)
    if is_in_hourly_summary_window(now, current_shift_num, current_hour_num):
        if shift_has_production:
            sched_key = f"hourly_summary_scheduled_{today_iso}_{current_shift_num}_{current_hour_num}"
            catch_key = f"hourly_summary_{today_iso}_{current_shift_num}_{current_hour_num}"
            if not bot_state_get(sched_key) and not bot_state_get(catch_key):
                header = f"📅 {format_date_time_12h(now)}\n\n"
                text = (
                        header
                        + f"📝 *Hourly Summary – Shift {current_shift_num}, Hour {current_hour_num}*"
                        + " _(missed — reconnected)_\n\n"
                        + "Please provide hourly production data:\n"
                        + "- Actual output for this hour\n"
                        + "- Downtime events (if any)\n"
                        + "- Rejects (if any)\n"
                        + "- Operator notes\n\n"
                        + "💡 AI will generate an hourly summary after you submit the data."
                )
                try:
                    await app.bot.send_message(
                        chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown"
                    )
                    bot_state_set(sched_key, "1")
                    bot_state_set(catch_key, "1")
                    sent_count += 1
                    await asyncio.sleep(1)
                    logger.info(
                        f"[RECOVERY] Hourly summary Shift {current_shift_num} "
                        f"Hr {current_hour_num} sent"
                    )
                except Exception as e:
                    logger.error(f"[RECOVERY] Hourly summary failed: {e}")
        else:
            logger.info(
                f"[RECOVERY] Hourly summary Shift {current_shift_num} Hr {current_hour_num} "
                f"skipped — no production this shift"
            )

    # ── 5. Shift Summary ─────────────────────────────────────────────────────
    # Summary reminder — only send if production occurred this shift
    # (CASE 1 with no production already returned early above)
    if is_in_shift_summary_window(current_shift_num, now):
        if shift_has_production:
            fired_key = f"shift_report_fired_{today_iso}_{current_shift_num}"
            recovery_key = f"shift_report_recovery_{today_iso}_{current_shift_num}"
            if not bot_state_get(fired_key) and not bot_state_get(recovery_key):
                header = f"📅 {format_date_time_12h(now)}\n\n"
                text = (
                        header
                        + f"📊 *Shift {current_shift_num} Summary Report Reminder* _(missed)_\n\n"
                        + "- Actual output\n"
                        + "- Downtime (reason + minutes)\n"
                        + "- Rejects (preform, bottle, cap, label)\n"
                        + "- Operator remarks"
                )
                try:
                    await app.bot.send_message(
                        chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown"
                    )
                    bot_state_set(recovery_key, "1")
                    bot_state_set(fired_key, "1")
                    sent_count += 1
                    await asyncio.sleep(1)
                    logger.info(f"[RECOVERY] Shift {current_shift_num} report sent")
                except Exception as e:
                    logger.error(f"[RECOVERY] Shift report failed: {e}")
        else:
            # No production — mark as fired so scheduler doesn't retry either
            logger.info(
                f"[RECOVERY] Shift {current_shift_num} summary skipped "
                f"— no production this shift"
            )
            bot_state_set(f"shift_report_fired_{today_iso}_{current_shift_num}", "1")

    if sent_count > 0:
        logger.info(f"[RECOVERY] Sent {sent_count} missed reminder(s)")
    else:
        logger.info("[RECOVERY] No missed reminders to send")


async def connection_watchdog(app) -> None:
    """
    Background task: pings Telegram every 60 seconds.
    When it detects a reconnect after failure, calls recover_missed_reminders_on_reconnect().
    """
    global _last_successful_send, _recovery_task_running
    _recovery_task_running = True
    was_offline = False

    logger.info("[WATCHDOG] Connection watchdog started")

    while True:
        await asyncio.sleep(60)  # check every 60 seconds
        try:
            await app.bot.get_me()  # lightweight ping
            if was_offline:
                logger.info("[WATCHDOG] Connection restored! Running missed reminder recovery...")
                was_offline = False
                try:
                    await recover_missed_reminders_on_reconnect(app)
                except Exception as e:
                    logger.error(f"[WATCHDOG] Recovery failed: {e}")
            _last_successful_send = now_ethiopia()
        except Exception as e:
            if not was_offline:
                logger.warning(f"[WATCHDOG] Connection lost: {e}")
            was_offline = True


# ---------------- COMMANDS ----------------
async def start_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ai_reminder_block

    user_id = update.effective_user.id
    active_users.add(user_id)
    user_ai_sessions[user_id] = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
    user_audit_state[user_id] = {"questions": 0, "completed": False, "ended": False}

    # While AI audit is active, silence all shift / hourly reminders (they will be queued)
    ai_reminder_block = True

    await update.message.reply_text(
        "✅ Audit triggered. Send shift reports. Use /end_audit to stop.\n"
        "🔇 While AI audit is active, all production reminders will be queued."
    )


async def end_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ai_reminder_block

    user_id = update.effective_user.id
    active_users.discard(user_id)
    user_ai_sessions.pop(user_id, None)
    user_audit_state.pop(user_id, None)

    # Re‑enable reminders and flush anything that was queued for AI
    ai_reminder_block = False
    await update.message.reply_text("🛑 Audit ended. AI questioning stopped.\n📣 Sending any queued reminders.")
    await flush_pending_reminders(context.bot, reason="ai")


async def _do_shift_summary(update: Update, context: ContextTypes.DEFAULT_TYPE, shift: int):
    """Shared logic for shift summary. Post to Telegram group and save to PostgreSQL."""
    global daily_ai_shift_summaries

    if shift not in [1, 2, 3]:
        await update.message.reply_text("Shift must be 1, 2, or 3.")
        return

    if not ai_shift_evidence[shift]:
        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=f"📊 SHIFT {shift} OFFICIAL SUMMARY\n\n⚠️ Shift summary is not provided.",
        )
        await update.message.reply_text(f"⚠️ Shift {shift} summary is not provided.")
        return

    # Parse and save to PostgreSQL before clearing evidence
    production_data = None
    downtime = []
    rejects = {}
    vos_info = None
    for text in reversed(ai_shift_evidence[shift]):
        try:
            production_data = parse_report(text)
            categorized_dt = parse_downtime_categorized(text)
            downtime = flatten_categorized_downtime(categorized_dt)
            rejects = parse_rejects(text)
            vos_info = parse_vos(text)
            break
        except Exception:
            continue
    if production_data:
        try:
            # Use requested shift (not parsed) to avoid wrong date/shift from mixed evidence
            save_to_database(production_data, downtime, rejects, vos_info, shift_override=shift)
            logger.info(f"Shift {shift} data saved to database")
        except Exception as e:
            logger.error(f"Failed to save shift {shift} to database: {e}")

    ai_text = await ai_generate_summary(shift)
    daily_ai_shift_summaries[shift] = ai_text

    # Send without parse_mode - AI content often contains _*[] that break Markdown
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📊 SHIFT {shift} OFFICIAL SUMMARY\n\n{ai_text}",
    )

    shift_closed[shift] = True
    ai_shift_evidence[shift] = []


async def shift_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Usage: /shift_summary 1 | 2 | 3 (kept for compatibility)."""
    if not context.args:
        await update.message.reply_text(
            "Usage: /shift_summary 1 | 2 | 3\nOr use: /shift_summary_1, /shift_summary_2, /shift_summary_3")
        return
    try:
        shift = int(context.args[0])
    except (ValueError, IndexError):
        await update.message.reply_text("Usage: /shift_summary 1 | 2 | 3")
        return
    await _do_shift_summary(update, context, shift)


async def shift_summary_1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _do_shift_summary(update, context, 1)

async def shift_summary_from_hourly_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /shift_summary_hourly 1 [date]
    Generate a shift summary by aggregating saved hourly data from the DB.
    Date is optional (DD/MM/YY), defaults to most recent data for the shift.
    """
    if not context.args:
        await update.message.reply_text(
            "Usage: /shift_summary_hourly 1 [date]\n"
            "Example: /shift_summary_hourly 2 (most recent)\n"
            "Example: /shift_summary_hourly 1 09/03/26 (specific date)"
        )
        return

    try:
        shift = int(context.args[0])
    except ValueError:
        await update.message.reply_text("First argument must be shift number (1, 2, or 3).")
        return

    if shift not in (1, 2, 3):
        await update.message.reply_text("Shift must be 1, 2, or 3.")
        return

    # Parse optional date - if no date provided, find most recent data
    target_date = None
    date_label = "most recent"
    
    if len(context.args) >= 2:
        raw = context.args[1].strip()
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                target_date = datetime.strptime(raw, fmt).date()
                date_label = target_date.strftime("%d/%m/%Y")
                break
            except ValueError:
                continue
        if target_date is None:
            await update.message.reply_text("Invalid date. Use DD/MM/YY, e.g. 09/03/26")
            return
    else:
        # Find most recent date with hourly data for this shift
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT date
                FROM hourly_production
                WHERE shift = %s
                ORDER BY date DESC
                LIMIT 1
            """, (shift,))
            result = cur.fetchone()
            cur.close()
            
            if not result:
                await update.message.reply_text(
                    f"⚠️ No hourly data found for Shift {shift} in the database.\n"
                    "Submit hourly reports first using /hourly_summary_ai."
                )
                return
            
            target_date = result[0]
            date_label = target_date.strftime("%d/%m/%Y")
            
        except Exception as e:
            logger.error(f"Database error finding recent date: {e}")
            await update.message.reply_text(f"❌ Database error: {e}")
            return
        finally:
            conn.close()

    # Load aggregated hourly data for this shift
    text_blob = load_shift_evidence_from_hourly_db(shift, target_date)
    if not text_blob:
        await update.message.reply_text(
            f"⚠️ No hourly data found for Shift {shift} on {date_label}.\n"
            "Submit hourly reports first using /hourly_summary_ai."
        )
        return

    await update.message.reply_text(
        f"⏳ Generating Shift {shift} summary from hourly data ({date_label})..."
    )

    # Temporarily load into ai_shift_evidence so ai_generate_summary works
    original_evidence = list(ai_shift_evidence[shift])
    ai_shift_evidence[shift] = [text_blob]
    try:
        ai_text = await ai_generate_summary(shift)
        daily_ai_shift_summaries[shift] = ai_text

        # Also save the aggregated data to the main production table
        try:
            prod = parse_report(text_blob)
            cat_dt = parse_downtime_categorized(text_blob)
            dt_flat = flatten_categorized_downtime(cat_dt)
            rej = parse_rejects(text_blob)
            vos = parse_vos(text_blob)
            save_to_database(prod, dt_flat, rej, vos_info=vos, shift_override=shift)
            logger.info(f"Aggregated hourly→shift data saved to production table: shift={shift}")
        except Exception as e:
            logger.warning(f"Failed to save aggregated shift data: {e}")

        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=f"📊 SHIFT {shift} SUMMARY (from hourly data — {date_label})\n\n{ai_text}",
        )
        shift_closed[shift] = True
    except Exception as e:
        logger.error(f"Error generating shift summary from hourly: {e}")
        await update.message.reply_text(f"❌ Error: {e}")
    finally:
        ai_shift_evidence[shift] = original_evidence


async def shift_summary_hourly_1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate Shift 1 summary from most recent hourly data in database"""
    await _generate_shift_summary_from_recent_hourly(update, context, 1)


async def shift_summary_hourly_2_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate Shift 2 summary from most recent hourly data in database"""
    await _generate_shift_summary_from_recent_hourly(update, context, 2)


async def shift_summary_hourly_3_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate Shift 3 summary from most recent hourly data in database"""
    await _generate_shift_summary_from_recent_hourly(update, context, 3)


async def _generate_shift_summary_from_recent_hourly(update: Update, context: ContextTypes.DEFAULT_TYPE, shift: int):
    """Generate shift summary from most recent hourly data for the specified shift"""
    # Find the most recent date with hourly data for this shift
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT date, COUNT(*) as hour_count
            FROM hourly_production
            WHERE shift = %s
            ORDER BY date DESC
            LIMIT 1
        """, (shift,))
        result = cur.fetchone()
        cur.close()
        
        if not result:
            await update.message.reply_text(
                f"⚠️ No hourly data found for Shift {shift} in the database.\n"
                "Submit hourly reports first using /hourly_summary_ai."
            )
            return
        
        recent_date, hour_count = result
        date_label = recent_date.strftime("%d/%m/%Y")
        
        # Load aggregated hourly data for this shift and date
        text_blob = load_shift_evidence_from_hourly_db(shift, recent_date)
        if not text_blob:
            await update.message.reply_text(
                f"⚠️ Error loading hourly data for Shift {shift} on {date_label}."
            )
            return

        await update.message.reply_text(
            f"⏳ Generating Shift {shift} summary from hourly data ({date_label}) - {hour_count} hour(s) found..."
        )

        # Temporarily load into ai_shift_evidence so ai_generate_summary works
        original_evidence = list(ai_shift_evidence[shift])
        ai_shift_evidence[shift] = [text_blob]
        try:
            ai_text = await ai_generate_summary(shift)
            daily_ai_shift_summaries[shift] = ai_text

            # Also save the aggregated data to the main production table
            try:
                prod = parse_report(text_blob)
                cat_dt = parse_downtime_categorized(text_blob)
                dt_flat = flatten_categorized_downtime(cat_dt)
                rej = parse_rejects(text_blob)
                vos = parse_vos(text_blob)
                save_to_database(prod, dt_flat, rej, vos_info=vos, shift_override=shift)
                logger.info(f"Aggregated hourly→shift data saved to production table: shift={shift}")
            except Exception as e:
                logger.warning(f"Failed to save aggregated shift data: {e}")

            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📊 SHIFT {shift} SUMMARY (from hourly data — {date_label})\n\n{ai_text}",
            )
            shift_closed[shift] = True
        except Exception as e:
            logger.error(f"Error generating shift summary from hourly: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
        finally:
            ai_shift_evidence[shift] = original_evidence
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text(f"❌ Database error: {e}")
    finally:
        conn.close()


async def all_shift_summary_from_hourly_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /all_shift_summary_hourly [date]
    Generate a multi-shift summary by aggregating hourly data for ALL shifts from the DB.
    Date is optional (DD/MM/YY), defaults to today.
    """
    target_date = None
    if context.args:
        raw = context.args[0].strip()
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                target_date = datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                continue
        if target_date is None:
            await update.message.reply_text("Invalid date. Use DD/MM/YY, e.g. 09/03/26")
            return

    if target_date is None:
        target_date = now_ethiopia().date()

    date_label = target_date.strftime("%d/%m/%Y")

    # Load all shifts from hourly DB
    hourly_evidence = load_all_shifts_from_hourly_db(target_date)
    shifts_found = [s for s in (1, 2, 3) if hourly_evidence.get(s)]

    if len(shifts_found) < 2:
        # Fall back: try main production table
        db_evidence = load_shift_evidence_from_db(target_date)
        db_shifts = [s for s in (1, 2, 3) if db_evidence.get(s)]

        if len(db_shifts) >= 2:
            await update.message.reply_text(
                f"⚠️ Only {len(shifts_found)} shift(s) with hourly data for {date_label}.\n"
                f"Falling back to shift-level data ({len(db_shifts)} shifts found)..."
            )
            original_evidence = {k: list(v) for k, v in ai_shift_evidence.items()}
            for s in (1, 2, 3):
                ai_shift_evidence[s] = db_evidence.get(s, [])
            try:
                await generate_multi_shift_summary_and_post(context, db_shifts)
            finally:
                for s in (1, 2, 3):
                    ai_shift_evidence[s] = original_evidence[s]
            return
        else:
            await update.message.reply_text(
                f"⚠️ Not enough data for {date_label}.\n"
                f"Hourly shifts: {len(shifts_found)}, Shift-level: {len(db_shifts)}.\n"
                "Need at least 2 shifts. Submit more reports first."
            )
            return

    await update.message.reply_text(
        f"⏳ Generating multi-shift summary from hourly data — "
        f"{date_label} ({len(shifts_found)} shifts: {shifts_found})..."
    )

    # Also save each shift's aggregated data to production table
    for s in shifts_found:
        try:
            text_blob = hourly_evidence[s][0]
            prod = parse_report(text_blob)
            cat_dt = parse_downtime_categorized(text_blob)
            dt_flat = flatten_categorized_downtime(cat_dt)
            rej = parse_rejects(text_blob)
            vos = parse_vos(text_blob)
            save_to_database(prod, dt_flat, rej, vos_info=vos, shift_override=s)
        except Exception as e:
            logger.warning(f"Failed to save aggregated shift {s}: {e}")

    # Swap into ai_shift_evidence temporarily
    original_evidence = {k: list(v) for k, v in ai_shift_evidence.items()}
    for s in (1, 2, 3):
        ai_shift_evidence[s] = hourly_evidence.get(s, [])
    try:
        await generate_multi_shift_summary_and_post(context, shifts_found)
    finally:
        for s in (1, 2, 3):
            ai_shift_evidence[s] = original_evidence[s]

async def shift_summary_2_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _do_shift_summary(update, context, 2)


async def shift_summary_3_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _do_shift_summary(update, context, 3)


async def shift_input_1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Two-step: ask user to paste Shift 1 report next."""
    context.user_data["shift_summary_pending"] = 1
    await update.message.reply_text(
        "✅ Shift set to 1.\n\n"
        "Now send your Shift 1 report in the next message (same format you normally paste).\n"
        "The bot will save it to DB and immediately post the AI summary to the group."
    )


async def shift_input_2_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Two-step: ask user to paste Shift 2 report next."""
    context.user_data["shift_summary_pending"] = 2
    await update.message.reply_text(
        "✅ Shift set to 2.\n\n"
        "Now send your Shift 2 report in the next message (same format you normally paste).\n"
        "The bot will save it to DB and immediately post the AI summary to the group."
    )


async def shift_input_3_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Two-step: ask user to paste Shift 3 report next."""
    context.user_data["shift_summary_pending"] = 3
    await update.message.reply_text(
        "✅ Shift set to 3.\n\n"
        "Now send your Shift 3 report in the next message (same format you normally paste).\n"
        "The bot will save it to DB and immediately post the AI summary to the group."
    )


# async def ai_generate_summary(shift: int):
#     evidence = ai_shift_evidence[shift]
#     if not evidence:
#         return "No evidence found."
#
#     production_data = None
#     downtime = []
#     rejects = {}
#     vos_info = None
#
#     for text in reversed(evidence):
#         try:
#             production_data = parse_report(text)
#             downtime = parse_downtime(text)
#             rejects = parse_rejects(text)
#             vos_info = parse_vos(text)  # Parse vos separately
#             break
#         except:
#             continue
#
#     if not production_data:
#         return "DATA INCOMPLETE – production report missing."
#
#     # ---------------- DATA AGGREGATION ----------------
#     total_downtime = sum(d["duration"] for d in downtime)
#     actual_output = production_data["actual"]
#     plan_output = production_data["plan"]
#
#     # Get available time (machine active time) - default to shift duration if not provided
#     available_time_minutes = production_data.get("available_time")
#     if available_time_minutes is None:
#         available_time_minutes = int(get_default_production_hours("shift", shift) * 60)
#
#     # Convert to production hours for KPI calculation
#     production_hours = available_time_minutes / 60
#
#     # ---------------- KPI CALCULATION ----------------
#     kpis = compute_kpis(
#         plan=plan_output,
#         actual=actual_output,
#         downtime_minutes=total_downtime,
#         production_hours=production_hours,
#         rejects=rejects
#     )
#
#     # ---------------- RISK CLASSIFICATION ----------------
#     risk_score = 0
#
#     if kpis["performance"] < 60:
#         risk_score += 3
#     elif kpis["performance"] < 75:
#         risk_score += 2
#
#     downtime_ratio = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes > 0 else 0
#     if downtime_ratio > 40:
#         risk_score += 3
#     elif downtime_ratio > 25:
#         risk_score += 2
#
#     # Risk assessment based on individual reject types (not summed)
#     total_rejects = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
#     if actual_output > 0:
#         reject_ratio = (total_rejects / actual_output) * 100
#         if reject_ratio > 5:
#             risk_score += 2
#         elif reject_ratio > 2:
#             risk_score += 1
#
#     # Mechanical fault detection
#     downtime_text = " ".join(d["description"] for d in downtime).lower()
#     if "misalignment" in downtime_text or "wear" in downtime_text:
#         risk_score += 1
#     if "short circuit" in downtime_text or "breaker" in downtime_text:
#         risk_score += 1
#     if "glue" in downtime_text or "adhesive" in downtime_text:
#         risk_score += 1
#
#     if risk_score >= 7:
#         risk_level = "CRITICAL"
#     elif risk_score >= 5:
#         risk_level = "HIGH"
#     elif risk_score >= 3:
#         risk_level = "MODERATE"
#     else:
#         risk_level = "LOW"
#
#     audit_status = "CLOSED" if shift_closed[shift] else "FOLLOW-UP REQUIRED"
#
#     # ---------------- STRUCTURED DATA FOR AI ----------------
#     structured_data = f"""
#                             SHIFT: {shift}
#                             DATE: {production_data["date"]}
#                             PRODUCT: {production_data["product_type"]}
#                             PLAN: {production_data["plan"]}
#                             ACTUAL: {production_data["actual"]}
#                             AVAILABLE_TIME: {available_time_minutes} minutes
#                             PRODUCTION_HOURS: {production_hours:.1f}
#                             PERFORMANCE: {kpis["performance"]}%
#                             AVAILABILITY: {kpis["availability"]}%
#                             QUALITY: {kpis["quality"]}%
#                             OEE: {kpis["oee"]}%
#                             TOTAL_DOWNTIME: {total_downtime} minutes
#                             DOWNTIME_RATIO: {downtime_ratio}%
#                             REJECTS_BREAKDOWN:
#                             - Preform: {rejects.get("preform", 0)}
#                             - Bottle: {rejects.get("bottle", 0)}
#                             - Cap: {rejects.get("cap", 0)}
#                             - Label: {rejects.get("label", 0)}
#                             - Shrink: {rejects.get("shrink", 0)} kg
#                             DEFECTIVE_QUANTITY: {kpis["defective_qty"]}
#                             RISK_LEVEL: {risk_level}
#                             AUDIT_STATUS: {audit_status}
#                             DOWNTIME_BREAKDOWN:
#                             {chr(10).join([f"- {d['description']} ({d['duration']} min)" for d in downtime])}
#                     """
#
#     # ---------------- AI EXECUTIVE NARRATIVE ----------------
#     loop = asyncio.get_running_loop()
#
#     def call_ai():
#         return ai_client.chat.completions.create(
#             model=AI_MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": """
#         You are a plant-level executive production analyst writing a professional shift summary report.
#
#        Write a well-structured executive summary that covers:
#         - Operational performance against plan
#         - Downtime impact and equipment reliability concerns
#         - Quality performance based on reject breakdowns
#         - Clear conclusions about shift stability
#
#         FORMATTING RULES (strict):
#         - Output 3–4 separate paragraphs. Do NOT merge into one long block.
#         - Each paragraph: 2–3 sentences only. One idea per paragraph.
#         - Insert exactly one blank line between each paragraph.
#         - Avoid wall-of-text. Preserve clarity and technical tone.
#
#        WRITING STYLE:
#         - Use proper grammar, capitalization, and punctuation.
#         - Begin every sentence with a capital letter.
#         - Use numeric format for all numbers (e.g., 42%, 64.07%, 240 minutes).
#         - Do NOT convert numbers into words.
#         - Be analytical, concise, executive-level.
#         - Base conclusions strictly on the structured data provided.
#
#         """
#
#                 },
#                 {
#                     "role": "user",
#                     "content": structured_data
#                 }
#             ],
#             temperature=0.2  # ensures numeric formatting and consistency
#         )
#
#     response = await loop.run_in_executor(None, call_ai)
#     executive_paragraph = response.choices[0].message.content.strip()
#
#     # ---------------- DETAILED BREAKDOWN SECTIONS ----------------
#     # Production Performance section
#     production_performance = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {production_data['product_type']}\n"
#         f"  • Plan: {plan_output:,} packs\n"
#         f"  • Actual: {actual_output:,} packs\n"
#         f"  • Available Time: {available_time_minutes} minutes\n"
#         f"  • Efficiency: {kpis['performance']}%"
#     )
#
#     # Add VOS if available
#     if vos_info:
#         production_performance += f"\n  • VOS: {vos_info}"
#
#     # Downtime Analysis section
#     # Show downtime items as separate data points
#     downtime_lines = "\n".join(
#         [f"  • {d['description']} ({d['duration']} min)" for d in downtime]) if downtime else "  • None"
#     downtime_analysis = (
#         f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
#         f"  • Total Downtime: {total_downtime} minutes\n"
#         f"  • Downtime Ratio: {downtime_ratio}% of available time\n"
#         f"{downtime_lines}"
#     )
#
#     # Quality Metrics section
#     quality_metrics = (
#         f"\n\n✓ QUALITY METRICS\n\n"
#         f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
#         f"  • Bottle Rejects: {rejects.get('bottle', 0):,} pcs\n"
#         f"  • Cap Rejects: {rejects.get('cap', 0):,} pcs\n"
#         f"  • Label Rejects: {rejects.get('label', 0):,} pcs\n"
#         f"  • Shrink Loss: {rejects.get('shrink', 0)} kg"
#     )
#
#     # ================= EXECUTIVE TABLE BLOCK =================
#     production_performance_kpi = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {production_data['product_type']} Ltr\n"
#         f"  • Plan: {plan_output:,} pcs\n"
#         f"  • Actual: {actual_output:,} pcs\n"
#         f"  • Achievement: {kpis['performance']:.1f}%"
#     )
#
#     reject_table = f"""
# 🔍 QUALITY – REJECT ANALYSIS
# -------------------------
#
# {'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
# {'-' * 47}
# Preform              {kpis['reject_percentages']['preform']:.1f} %           {rejects.get('preform', 0)} pcs
# Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {rejects.get('bottle', 0)} pcs
# Cap                  {kpis['reject_percentages']['cap']:.1f} %          {rejects.get('cap', 0)} pcs
# Label                {kpis['reject_percentages']['label']:.1f} %          {rejects.get('label', 0)} pcs
# Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {rejects.get('shrink', 0)} kg
# """
#
#     oee_performance = (
#         f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
#         f"  • Plan: {plan_output:,} pcs\n"
#         f"  • Actual: {actual_output:,} pcs\n"
#         f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
#         f"  • Production Time: {production_hours:.1f} hr\n"
#         f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
#         f"  • Availability: {kpis['availability']:.1f}%\n"
#         f"  • Performance: {kpis['performance']:.1f}%\n"
#         f"  • Quality: {kpis['quality']:.1f}%\n"
#         f"  • OEE: {kpis['oee']:.2f}%"
#     )
#
#     # ---------------- FINAL REPORT (universal structure) ----------------
#     final_report = (
#         f"✅ STATUS: COMPLETE\n\n"
#         f"⚠️ RISK LEVEL: {risk_level}\n\n"
#         f"📋 EXECUTIVE SUMMARY\n\n"
#         f"{executive_paragraph}\n\n"
#         f"────────────────────────────\n\n"
#         f"{production_performance}"
#         f"{downtime_analysis}"
#         f"{quality_metrics}"
#         f"\n\n────────────────────────────\n\n"
#         f"{reject_table}"
#         f"\n\n────────────────────────────\n\n"
#         f"{production_performance_kpi}"
#         f"\n\n────────────────────────────\n\n"
#         f"{oee_performance}"
#         f"\n\n────────────────────────────\n\n"
#         f"📌 AUDIT STATUS: {audit_status}"
#     )
#
#     return final_report.strip()


# async def ai_generate_hourly_summary_from_text(report_text: str):
#     """
#     Generate an AI executive-style summary for ONE HOUR of production.
#     The input format is the same as for a shift summary:
#     - Contains date, shift, shift plan, actual, downtime, rejects, etc.
#     """
#     try:
#         production_data = parse_report(report_text)
#     except Exception:
#         return "DATA INCOMPLETE – production report missing."
#
#     downtime = parse_downtime(report_text)
#     rejects = parse_rejects(report_text)
#     vos_info = parse_vos(report_text)  # Parse vos separately
#
#     total_downtime = sum(d["duration"] for d in downtime)
#     actual_output = production_data["actual"]
#     plan_output = production_data["plan"]
#
#     # Get available time (machine active time) - default to 1 hour for hourly if not provided
#     available_time_minutes = production_data.get("available_time")
#     if available_time_minutes is None:
#         available_time_minutes = int(get_default_production_hours("hourly") * 60)
#
#     # Convert to production hours for KPI calculation
#     production_hours = available_time_minutes / 60
#
#     # ---------------- KPI CALCULATION ----------------
#     kpis = compute_kpis(
#         plan=plan_output,
#         actual=actual_output,
#         downtime_minutes=total_downtime,
#         production_hours=production_hours,
#         rejects=rejects
#     )
#
#     # ---------------- RISK CLASSIFICATION ----------------
#     risk_score = 0
#
#     if kpis["performance"] < 60:
#         risk_score += 3
#     elif kpis["performance"] < 75:
#         risk_score += 2
#
#     downtime_ratio = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes > 0 else 0
#     if downtime_ratio > 40:
#         risk_score += 3
#     elif downtime_ratio > 25:
#         risk_score += 2
#
#     # Risk assessment based on individual reject types (not summed)
#     total_rejects = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
#     if actual_output > 0:
#         reject_ratio = (total_rejects / actual_output) * 100
#         if reject_ratio > 5:
#             risk_score += 2
#         elif reject_ratio > 2:
#             risk_score += 1
#
#     downtime_text = " ".join(d["description"] for d in downtime).lower()
#     if "misalignment" in downtime_text or "wear" in downtime_text:
#         risk_score += 1
#     if "short circuit" in downtime_text or "breaker" in downtime_text:
#         risk_score += 1
#     if "glue" in downtime_text or "adhesive" in downtime_text:
#         risk_score += 1
#
#     if risk_score >= 7:
#         risk_level = "CRITICAL"
#     elif risk_score >= 5:
#         risk_level = "HIGH"
#     elif risk_score >= 3:
#         risk_level = "MODERATE"
#     else:
#         risk_level = "LOW"
#
#     audit_status = "FOLLOW-UP REQUIRED"
#
#     # ---------------- STRUCTURED DATA FOR AI ----------------
#     structured_data = f"""
#                         HOUR SHIFT: {production_data["shift"]}
#                         DATE: {production_data["date"]}
#                         PRODUCT: {production_data["product_type"]}
#                         PLAN (hour): {production_data["plan"]}
#                         ACTUAL (hour): {production_data["actual"]}
#                         AVAILABLE_TIME: {available_time_minutes} minutes
#                         PRODUCTION_HOURS: {production_hours:.1f}
#                         PERFORMANCE: {kpis["performance"]}%
#                         AVAILABILITY: {kpis["availability"]}%
#                         QUALITY: {kpis["quality"]}%
#                         OEE: {kpis["oee"]}%
#                         TOTAL_DOWNTIME: {total_downtime} minutes
#                         DOWNTIME_RATIO: {downtime_ratio}%
#                         REJECTS_BREAKDOWN:
#                         - Preform: {rejects.get("preform", 0)}
#                         - Bottle: {rejects.get("bottle", 0)}
#                         - Cap: {rejects.get("cap", 0)}
#                         - Label: {rejects.get("label", 0)}
#                         - Shrink: {rejects.get("shrink", 0)} kg
#                         DEFECTIVE_QUANTITY: {kpis["defective_qty"]}
#                         RISK_LEVEL: {risk_level}
#                         AUDIT_STATUS: {audit_status}
#                         DOWNTIME_BREAKDOWN:
#                         {chr(10).join([f"- {d['description']} ({d['duration']} min)" for d in downtime])}
#                 """
#
#     loop = asyncio.get_running_loop()
#
#     def call_ai():
#         return ai_client.chat.completions.create(
#             model=AI_MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": """
# You are a plant-level executive production analyst writing a professional hourly summary report.
#
# Write a well-structured executive summary that evaluates ONE HOUR of production:
# - Operational performance against plan for this hour
# - Downtime impact and equipment reliability concerns
# - Quality performance based on reject breakdowns
# - Clear conclusions about overall hour stability
#
# FORMATTING RULES (strict):
# - Output 3–4 separate paragraphs. Do NOT merge into one long block.
# - Each paragraph: 2–3 sentences only. One idea per paragraph.
# - Insert exactly one blank line between each paragraph.
# - Avoid wall-of-text. Preserve clarity and technical tone.
#
# WRITING STYLE:
# - Use proper grammar, capitalization, and punctuation.
# - Begin every sentence with a capital letter.
# - Use numeric format for all numbers (e.g., 42%, 64.07%, 40 minutes).
# - Do NOT convert numbers into words.
# - Be analytical, concise, executive-level.
# - Base conclusions strictly on the structured data provided.
# """
#                 },
#                 {
#                     "role": "user",
#                     "content": structured_data
#                 },
#             ],
#             temperature=0.2,
#         )
#
#     response = await loop.run_in_executor(None, call_ai)
#     executive_paragraph = response.choices[0].message.content.strip()
#
#     # ---------------- DETAILED BREAKDOWN SECTIONS ----------------
#     # Production Performance section
#     production_performance = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {production_data['product_type']}\n"
#         f"  • Plan: {plan_output:,} packs\n"
#         f"  • Actual: {actual_output:,} packs\n"
#         f"  • Available Time: {available_time_minutes} minutes\n"
#         f"  • Efficiency: {kpis['performance']}%"
#     )
#
#     # Add VOS if available
#     if vos_info:
#         production_performance += f"\n  • VOS: {vos_info}"
#
#     # Downtime Analysis section
#     # Show downtime items as separate data points
#     downtime_lines = "\n".join(
#         [f"  • {d['description']} ({d['duration']} min)" for d in downtime]) if downtime else "  • None"
#     downtime_analysis = (
#         f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
#         f"  • Total Downtime: {total_downtime} minutes\n"
#         f"  • Downtime Ratio: {downtime_ratio}% of available time\n"
#         f"{downtime_lines}"
#     )
#
#     # Quality Metrics section
#     quality_metrics = (
#         f"\n\n✓ QUALITY METRICS\n\n"
#         f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
#         f"  • Bottle Rejects: {rejects.get('bottle', 0):,} pcs\n"
#         f"  • Cap Rejects: {rejects.get('cap', 0):,} pcs\n"
#         f"  • Label Rejects: {rejects.get('label', 0):,} pcs\n"
#         f"  • Shrink Loss: {rejects.get('shrink', 0)} kg"
#     )
#
#     # ================= EXECUTIVE TABLE BLOCK =================
#     production_performance_kpi = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {production_data['product_type']} Ltr\n"
#         f"  • Plan: {plan_output:,} pcs\n"
#         f"  • Actual: {actual_output:,} pcs\n"
#         f"  • Achievement: {kpis['performance']:.1f}%"
#     )
#
#     reject_table = f"""
# 🔍 QUALITY – REJECT ANALYSIS
# -------------------------
#
# {'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
# {'-' * 47}
# Preform              {kpis['reject_percentages']['preform']:.1f} %           {rejects.get('preform', 0)} pcs
# Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {rejects.get('bottle', 0)} pcs
# Cap                  {kpis['reject_percentages']['cap']:.1f} %          {rejects.get('cap', 0)} pcs
# Label                {kpis['reject_percentages']['label']:.1f} %          {rejects.get('label', 0)} pcs
# Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {rejects.get('shrink', 0)} kg
# """
#
#     oee_performance = (
#         f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
#         f"  • Plan: {plan_output:,} pcs\n"
#         f"  • Actual: {actual_output:,} pcs\n"
#         f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
#         f"  • Production Time: {production_hours:.1f} hr\n"
#         f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
#         f"  • Availability: {kpis['availability']:.1f}%\n"
#         f"  • Performance: {kpis['performance']:.1f}%\n"
#         f"  • Quality: {kpis['quality']:.1f}%\n"
#         f"  • OEE: {kpis['oee']:.2f}%"
#     )
#
#     # ---------------- FINAL REPORT (universal structure) ----------------
#     final_report = (
#         f"✅ STATUS: COMPLETE\n\n"
#         f"⚠️ RISK LEVEL: {risk_level}\n\n"
#         f"📋 EXECUTIVE SUMMARY\n\n"
#         f"{executive_paragraph}\n\n"
#         f"────────────────────────────\n\n"
#         f"{production_performance}"
#         f"{downtime_analysis}"
#         f"{quality_metrics}"
#         f"\n\n────────────────────────────\n\n"
#         f"{reject_table}"
#         f"\n\n────────────────────────────\n\n"
#         f"{production_performance_kpi}"
#         f"\n\n────────────────────────────\n\n"
#         f"{oee_performance}"
#         f"\n\n────────────────────────────\n\n"
#         f"📌 AUDIT STATUS: {audit_status}"
#     )
#
#     return final_report.strip()


# async def ai_generate_multi_shift_summary(included_shifts: list[int]):
#     """
#     Generate multi-shift summary using deterministic KPI calculations.
#     Aggregates raw data from included shifts and applies universal report structure.
#     Ensures all shifts with the same date are included for 24hr production.
#     """
#     if not included_shifts:
#         return None
#
#     # ---------------- DATA AGGREGATION ----------------
#     target_date = None
#     for shift in included_shifts:
#         if ai_shift_evidence[shift]:
#             for text in reversed(ai_shift_evidence[shift]):
#                 try:
#                     production_data = parse_report(text)
#                     if production_data and production_data.get("date"):
#                         target_date = str(production_data["date"])
#                         break
#                 except:
#                     continue
#         if target_date:
#             break
#
#     if not target_date:
#         logger.warning("ai_generate_multi_shift_summary: could not find target_date from evidence")
#         return None
#
#     logger.info(f"ai_generate_multi_shift_summary: target_date={target_date}, shifts={included_shifts}")
#
#     total_plan = 0
#     total_actual = 0
#     total_downtime = 0
#     total_production_hours = 0
#     total_rejects = {"preform": 0, "bottle": 0, "cap": 0, "label": 0, "shrink": 0}
#     product_types = []
#     all_downtime_items = []
#     all_vos_info = []  # populated inside loop as "Shift N: value"
#     shifts_with_data = []  # track which shifts actually had data
#
#     for shift in (1, 2, 3):
#         if not ai_shift_evidence[shift]:
#             continue
#
#         shift_production_data = None
#         shift_downtime = []
#         shift_rejects = {}
#         shift_vos_info = None
#
#         for text in reversed(ai_shift_evidence[shift]):
#             try:
#                 production_data = parse_report(text)
#                 if production_data and production_data.get("date"):
#                     parsed_date_str = str(production_data["date"])
#                     if parsed_date_str == target_date:
#                         shift_production_data = production_data
#                         shift_downtime = parse_downtime(text)
#                         shift_rejects = parse_rejects(text)
#                         shift_vos_info = parse_vos(text)
#                         break
#             except:
#                 continue
#
#         if not shift_production_data:
#             logger.info(f"ai_generate_multi_shift_summary: no matching data for shift {shift}")
#             continue
#
#         logger.info(
#             f"ai_generate_multi_shift_summary: aggregating shift {shift} — plan={shift_production_data['plan']}, actual={shift_production_data['actual']}")
#
#         shifts_with_data.append(shift)
#
#         total_plan += shift_production_data["plan"]
#         total_actual += shift_production_data["actual"]
#         total_downtime += sum(d["duration"] for d in shift_downtime)
#
#         for d in shift_downtime:
#             all_downtime_items.append({
#                 "description": d["description"],
#                 "duration": d["duration"],
#                 "shift": shift
#             })
#
#         # Collect VOS per shift — always add entry (even if None)
#         vos_val = shift_vos_info.strip() if shift_vos_info else "none"
#         all_vos_info.append(f"Shift {shift}: {vos_val}")
#
#         available_time_minutes = shift_production_data.get("available_time")
#         if available_time_minutes is None:
#             available_time_minutes = int(get_default_production_hours("shift", shift) * 60)
#         total_production_hours += available_time_minutes / 60
#
#         for category in total_rejects:
#             total_rejects[category] = round(total_rejects[category] + shift_rejects.get(category, 0), 2)
#
#         if shift_production_data["product_type"]:
#             product_types.append(shift_production_data["product_type"])
#
#     if total_plan == 0:
#         logger.warning("ai_generate_multi_shift_summary: total_plan=0, no data aggregated")
#         return None
#
#     # ---------------- KPI CALCULATION ----------------
#     kpis = compute_kpis(
#         plan=total_plan,
#         actual=total_actual,
#         downtime_minutes=total_downtime,
#         production_hours=total_production_hours,
#         rejects=total_rejects
#     )
#
#     # ---------------- RISK CLASSIFICATION ----------------
#     risk_score = 0
#
#     if kpis["performance"] < 60:
#         risk_score += 3
#     elif kpis["performance"] < 75:
#         risk_score += 2
#
#     total_available_minutes = total_production_hours * 60
#     downtime_ratio = round((total_downtime / total_available_minutes) * 100, 2) if total_available_minutes > 0 else 0
#     if downtime_ratio > 40:
#         risk_score += 3
#     elif downtime_ratio > 25:
#         risk_score += 2
#
#     total_reject_count = total_rejects.get("bottle", 0) + total_rejects.get("cap", 0) + total_rejects.get("label", 0)
#     if total_actual > 0:
#         reject_ratio = (total_reject_count / total_actual) * 100
#         if reject_ratio > 5:
#             risk_score += 2
#         elif reject_ratio > 2:
#             risk_score += 1
#
#     if risk_score >= 7:
#         risk_level = "CRITICAL"
#     elif risk_score >= 5:
#         risk_level = "HIGH"
#     elif risk_score >= 3:
#         risk_level = "MODERATE"
#     else:
#         risk_level = "LOW"
#
#     audit_status = "CLOSED"
#
#     # ---------------- STRUCTURED DATA FOR AI ----------------
#     product_type_str = ", ".join(set(product_types)) if product_types else "Mixed"
#
#     structured_data = f"""
#                         MULTI-SHIFT SUMMARY: ALL SHIFTS FOR {target_date}
#                         DATE: {target_date}
#                         PRODUCT(S): {product_type_str}
#                         TOTAL PLAN: {total_plan:,}
#                         TOTAL ACTUAL: {total_actual:,}
#                         TOTAL PRODUCTION HOURS: {total_production_hours:.1f} hr (22hr planned production)
#                         PERFORMANCE: {kpis["performance"]}%
#                         AVAILABILITY: {kpis["availability"]}%
#                         QUALITY: {kpis["quality"]}%
#                         OEE: {kpis["oee"]}%
#                         TOTAL DOWNTIME: {total_downtime} minutes
#                         DOWNTIME_RATIO: {downtime_ratio}%
#                         REJECTS_BREAKDOWN:
#                         - Preform: {total_rejects.get("preform", 0):,}
#                         - Bottle: {total_rejects.get("bottle", 0):,}
#                         - Cap: {total_rejects.get("cap", 0):,}
#                         - Label: {total_rejects.get("label", 0):,}
#                         - Shrink: {total_rejects.get("shrink", 0):,} kg
#                         DEFECTIVE_QUANTITY: {kpis["defective_qty"]:,}
#                         RISK_LEVEL: {risk_level}
#                         AUDIT_STATUS: {audit_status}
#                     """
#
#     # ---------------- AI EXECUTIVE NARRATIVE ----------------
#     loop = asyncio.get_running_loop()
#
#     def call_ai():
#         return ai_client.chat.completions.create(
#             model=AI_MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"""
# You are a plant-level executive production analyst writing a professional multi-shift summary report.
#
# Write a well-structured executive summary that analyzes the complete 24-hour production performance for {target_date}:
# - Overall operational performance against aggregated plan across all shifts
# - Cumulative downtime impact and equipment reliability trends across all shifts
# - Aggregate quality performance and reject patterns across all shifts
# - Clear conclusions about full-day production stability and 24-hour operational effectiveness
#
# FORMATTING RULES (strict):
# - Output 3–4 separate paragraphs. Do NOT merge into one long block.
# - Each paragraph: 2–3 sentences only. One idea per paragraph.
# - Insert exactly one blank line between each paragraph.
# - Avoid wall-of-text. Preserve clarity and technical tone.
#
# WRITING STYLE:
# - Use proper grammar, capitalization, and punctuation.
# - Begin every sentence with a capital letter.
# - Use numeric format for all numbers (e.g., 42%, 64.07%, 350 minutes).
# - Do NOT convert numbers into words.
# - Be analytical, concise, executive-level.
# - Base conclusions strictly on the structured data provided for all shifts with the same date.
# """
#                 },
#                 {
#                     "role": "user",
#                     "content": structured_data
#                 }
#             ],
#             temperature=0.2,
#         )
#
#     response = await loop.run_in_executor(None, call_ai)
#     executive_paragraph = response.choices[0].message.content.strip()
#
#     # ---------------- DETAILED BREAKDOWN SECTIONS ----------------
#
#     # ── Production Performance ───────────────────────────────────────────────
#     production_performance = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {product_type_str}\n"
#         f"  • Plan: {total_plan:,} packs\n"
#         f"  • Actual: {total_actual:,} packs\n"
#         f"  • Available Time: {total_available_minutes:.0f} minutes\n"
#         f"  • Efficiency: {kpis['performance']}%\n"
#         f"  • VOS:\n"
#     )
#     # One line per shift that had data — always shows, "none" if no VOS reported
#     for vos_entry in all_vos_info:
#         production_performance += f"      {vos_entry}\n"
#
#     # ── Downtime Analysis ────────────────────────────────────────────────────
#     shift_downtime_groups = {}
#     for item in all_downtime_items:
#         shift_num = item["shift"]
#         if shift_num not in shift_downtime_groups:
#             shift_downtime_groups[shift_num] = []
#         shift_downtime_groups[shift_num].append(f"  • {item['description']} ({item['duration']} min)")
#
#     downtime_lines = ""
#     for shift_num in sorted(shift_downtime_groups.keys()):
#         downtime_lines += f"\nShift {shift_num}:\n"
#         downtime_lines += "\n".join(shift_downtime_groups[shift_num])
#
#     if not all_downtime_items:
#         downtime_lines = "  • None"
#
#     downtime_analysis = (
#         f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
#         f"  • Total Downtime: {total_downtime} minutes\n"
#         f"  • Downtime Ratio: {downtime_ratio}% of available time\n"
#         f"{downtime_lines}"
#     )
#
#     # ── Quality Metrics ──────────────────────────────────────────────────────
#     quality_metrics = (
#         f"\n\n✓ QUALITY METRICS\n\n"
#         f"  • Preform Rejects: {total_rejects.get('preform', 0):,} pcs\n"
#         f"  • Bottle Rejects: {total_rejects.get('bottle', 0):,} pcs\n"
#         f"  • Cap Rejects: {total_rejects.get('cap', 0):,} pcs\n"
#         f"  • Label Rejects: {total_rejects.get('label', 0):,} pcs\n"
#         f"  • Shrink Loss: {round(total_rejects.get('shrink', 0), 2)} kg"
#     )
#
#     # ── KPI Table ────────────────────────────────────────────────────────────
#     production_performance_kpi = (
#         f"📊 PRODUCTION PERFORMANCE\n\n"
#         f"  • Product: {product_type_str} Ltr\n"
#         f"  • Plan: {total_plan:,} pcs\n"
#         f"  • Actual: {total_actual:,} pcs\n"
#         f"  • Achievement: {kpis['performance']:.1f}%"
#     )
#
#     reject_table = f"""
# 🔍 QUALITY – REJECT ANALYSIS
# -------------------------
#
# {'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
# {'-' * 47}
# Preform              {kpis['reject_percentages']['preform']:.1f} %           {total_rejects.get('preform', 0)} pcs
# Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {total_rejects.get('bottle', 0)} pcs
# Cap                  {kpis['reject_percentages']['cap']:.1f} %          {total_rejects.get('cap', 0)} pcs
# Label                {kpis['reject_percentages']['label']:.1f} %          {total_rejects.get('label', 0)} pcs
# Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {round(total_rejects.get('shrink', 0), 2)} kg
# """
#
#     oee_performance = (
#         f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
#         f"  • Plan: {total_plan:,} pcs\n"
#         f"  • Actual: {total_actual:,} pcs\n"
#         f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
#         f"  • Production Time: {total_production_hours:.1f} hr\n"
#         f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
#         f"  • Availability: {kpis['availability']:.1f}%\n"
#         f"  • Performance: {kpis['performance']:.1f}%\n"
#         f"  • Quality: {kpis['quality']:.1f}%\n"
#         f"  • OEE: {kpis['oee']:.2f}%"
#     )
#
#     # ---------------- FINAL REPORT ----------------
#     final_report = (
#         f"✅ STATUS: COMPLETE\n\n"
#         f"⚠️ RISK LEVEL: {risk_level}\n\n"
#         f"📋 EXECUTIVE SUMMARY\n\n"
#         f"{executive_paragraph}\n\n"
#         f"────────────────────────────\n\n"
#         f"{production_performance}"
#         f"{downtime_analysis}"
#         f"{quality_metrics}"
#         f"\n\n────────────────────────────\n\n"
#         f"{reject_table}"
#         f"\n\n────────────────────────────\n\n"
#         f"{production_performance_kpi}"
#         f"\n\n────────────────────────────\n\n"
#         f"{oee_performance}"
#         f"\n\n────────────────────────────\n\n"
#         f"📌 AUDIT STATUS: {audit_status}"
#     )
#
#     return final_report.strip()


async def generate_multi_shift_summary_and_post(
        context: ContextTypes.DEFAULT_TYPE,
        included_shifts: list[int],
) -> None:
    """
    Helper to call multi-shift AI and post into group.
    """
    # Build label directly — never re-scan ai_shift_evidence for this
    if len(included_shifts) == 1:
        label = f"Shift {included_shifts[0]}"
    elif len(included_shifts) == 2:
        label = f"Shifts {included_shifts[0]} and {included_shifts[1]}"
    else:
        label = f"Shifts {', '.join(str(s) for s in included_shifts[:-1])} and {included_shifts[-1]}"

    daily_text = await ai_generate_multi_shift_summary(included_shifts)
    if not daily_text:
        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text="⚠️ No complete multi-shift summary available. Please ensure all shifts have data for the same date.",
            parse_mode=None,
        )
        return

    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📘 MULTI-SHIFT PRODUCTION SUMMARY – {label}\n\n{daily_text}",
        parse_mode=None,
    )


DOWNTIME_CATEGORIES = {
    "MECHANICAL": ["mechanical", "machine", "technical"],
    "ELECTRICAL": ["electrical", "electric"],
    "UTILITY":    ["utility", "utilities"],
}

def parse_downtime_categorized(text: str) -> dict:
    """
    Parses downtime events from input text grouped under:
    MECHANICAL, ELECTRICAL, UTILITY

    Input format expected:
        MECHANICAL
        • Some event description (245 min)
        ELECTRICAL
        • None
        UTILITY
        • Low pressure shortage problem (20 min)
    """
    result  = {"MECHANICAL": [], "ELECTRICAL": [], "UTILITY": []}
    current = None  # no default — wait for a real header

    for raw_line in text.split('\n'):
        line  = raw_line.strip()
        lower = line.lower()

        if not line:
            continue

        # ── Step 1: Detect category header ────────────────────────────────
        matched_cat = None
        for cat, keywords in DOWNTIME_CATEGORIES.items():
            if any(kw in lower for kw in keywords):
                matched_cat = cat
                break

        if matched_cat:
            # Strip everything except the keyword to confirm it's a header
            cleaned = lower
            cleaned = re.sub(r'\(?\d+\s*(?:min|minutes?|\')\)?', '', cleaned)
            cleaned = re.sub(r'[\[\]()\-—•:\d]+', '', cleaned)
            for kw in DOWNTIME_CATEGORIES[matched_cat]:
                cleaned = cleaned.replace(kw, '')
            cleaned = cleaned.strip()

            if len(cleaned) <= 5:
                current = matched_cat
                continue  # it's a header, not an event

        # ── Step 2: Skip if no active category yet ────────────────────────
        if current is None:
            continue

        # ── Step 3: Skip "None" placeholder lines ─────────────────────────
        if re.match(r'^[•\-\*]?\s*none\s*$', lower):
            continue

        # ── Step 4: Parse bullet event lines ──────────────────────────────
        m = re.search(r'\(?(\d+)\s*(?:min|minutes?|\')\)?', lower)
        if m:
            duration = int(m.group(1))

            # Clean description
            desc = re.sub(r'^[•\-\*]\s*', '', line)
            desc = re.sub(r'^\d+\.\s*', '', desc)
            desc = re.sub(
                r'\s*\(?\d+\s*(?:min|minutes?|\')\)?\s*$', '', desc,
                flags=re.IGNORECASE
            ).strip()

            if len(desc) > 2:
                result[current].append({
                    "description": desc,
                    "duration":    duration,
                    "category":    current,
                })

    # ── Compute totals per category ────────────────────────────────────────
    result["_totals"] = {
        cat: sum(e["duration"] for e in result[cat])
        for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY")
    }
    return result

def format_downtime_category_block(categorized: dict) -> str:
    icons = {
        "MECHANICAL": "⚙️",
        "ELECTRICAL": "🔌",
        "UTILITY":    "🏭",
    }
    lines = []
    for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY"):
        items  = categorized.get(cat, [])
        totals = categorized["_totals"]
        icon   = icons[cat]
        total  = totals.get(cat, 0)

        lines.append(f"\n\n  {icon} {cat} — {total} min")
        if items:
            for item in items:
                lines.append(f"    • {item['description']} ({item['duration']} min)")
        else:
            lines.append(f"    • None")

    return "\n".join(lines)

def build_downtime_analysis_block(categorized: dict, available_time: int) -> str:
    """
    Builds the full DOWNTIME ANALYSIS summary block.
    """
    totals     = categorized.get("_totals", {})
    mech_total = totals.get("MECHANICAL", 0)
    elec_total = totals.get("ELECTRICAL", 0)
    util_total = totals.get("UTILITY", 0)
    total_dt   = mech_total + elec_total + util_total

    # Downtime ratio
    ratio = (total_dt / available_time * 100) if available_time > 0 else 0.0

    # Dominant category
    cat_map = {
        "MECHANICAL": mech_total,
        "ELECTRICAL": elec_total,
        "UTILITY":    util_total,
    }
    dominant_cat   = max(cat_map, key=cat_map.get)
    dominant_total = cat_map[dominant_cat]

    block = (
        f"\n⏱️ DOWNTIME ANALYSIS\n\n"
        f"  • Total Downtime:     {total_dt} minutes\n"
        f"  • Downtime Ratio:     {ratio:.2f}% of available time\n"
        f"  • Dominant Category:  {dominant_cat} ({dominant_total} min)\n"
        f"  • Mechanical:         {mech_total} min\n"
        f"  • Electrical:         {elec_total} min\n"
        f"  • Utility:            {util_total} min\n"
        f"{format_downtime_category_block(categorized)}"
    )
    return block

def flatten_categorized_downtime(categorized: dict) -> list:
    """
    Flattens the categorized downtime dict into a simple list of event dicts.
    Each item: {"description": str, "duration": int, "category": str}
    """
    flat = []
    for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY"):
        flat.extend(categorized.get(cat, []))
    return flat

async def ai_generate_summary(shift: int):
    evidence = ai_shift_evidence[shift]
    if not evidence:
        return "No evidence found."

    production_data = None
    downtime        = []
    rejects         = {}
    vos_info        = None
    categorized_dt  = {"MECHANICAL": [], "ELECTRICAL": [], "UTILITY": [], "_totals": {}}

    for text in reversed(evidence):
        try:
            production_data = parse_report(text)
            categorized_dt  = parse_downtime_categorized(text)   # ← NEW
            downtime        = flatten_categorized_downtime(categorized_dt)  # ← NEW
            rejects         = parse_rejects(text)
            vos_info        = parse_vos(text)
            break
        except Exception:
            continue

    if not production_data:
        return "DATA INCOMPLETE – production report missing."

    # ── Aggregation (unchanged) ───────────────────────────────────────────────
    total_downtime         = sum(d["duration"] for d in downtime)
    actual_output          = production_data["actual"]
    plan_output            = production_data["plan"]
    available_time_minutes = production_data.get("available_time") or int(
        get_default_production_hours("shift", shift) * 60
    )
    production_hours       = available_time_minutes / 60
    dt_totals              = categorized_dt["_totals"]
    dominant_cat           = max(dt_totals, key=dt_totals.get) if any(dt_totals.values()) else "N/A"

    # ── KPI (unchanged) ──────────────────────────────────────────────────────
    kpis          = compute_kpis(plan_output, actual_output, total_downtime, production_hours, rejects)
    downtime_ratio = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes else 0

    # ── Risk (unchanged) ─────────────────────────────────────────────────────
    risk_score = 0
    if kpis["performance"] < 60:   risk_score += 3
    elif kpis["performance"] < 75: risk_score += 2
    if downtime_ratio > 40:        risk_score += 3
    elif downtime_ratio > 25:      risk_score += 2
    total_rejects_count = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
    if actual_output > 0:
        rr = (total_rejects_count / actual_output) * 100
        if rr > 5:   risk_score += 2
        elif rr > 2: risk_score += 1
    downtime_text = " ".join(d["description"] for d in downtime).lower()
    if any(w in downtime_text for w in ("misalignment", "wear")):       risk_score += 1
    if any(w in downtime_text for w in ("short circuit", "breaker")):   risk_score += 1
    if any(w in downtime_text for w in ("glue", "adhesive")):           risk_score += 1
    risk_level   = ("CRITICAL" if risk_score >= 7 else "HIGH" if risk_score >= 5
                    else "MODERATE" if risk_score >= 3 else "LOW")
    audit_status = "CLOSED" if shift_closed[shift] else "FOLLOW-UP REQUIRED"

    # ── Structured data for AI (UPDATED downtime section) ────────────────────
    structured_data = f"""
SHIFT: {shift}
DATE: {production_data["date"]}
PRODUCT: {production_data["product_type"]}
PLAN: {production_data["plan"]}
ACTUAL: {production_data["actual"]}
AVAILABLE_TIME: {available_time_minutes} minutes
PRODUCTION_HOURS: {production_hours:.1f}
PERFORMANCE: {kpis["performance"]}%
AVAILABILITY: {kpis["availability"]}%
QUALITY: {kpis["quality"]}%
OEE: {kpis["oee"]}%
TOTAL_DOWNTIME: {total_downtime} minutes
DOWNTIME_RATIO: {downtime_ratio}%

DOWNTIME BY CATEGORY:
  MECHANICAL: {dt_totals.get("MECHANICAL", 0)} min
  ELECTRICAL: {dt_totals.get("ELECTRICAL", 0)} min
  UTILITY:    {dt_totals.get("UTILITY", 0)} min
  DOMINANT:   {dominant_cat}

DOWNTIME DETAIL:
{format_downtime_category_block(categorized_dt)}

REJECTS_BREAKDOWN:
- Preform: {rejects.get("preform", 0)}
- Bottle:  {rejects.get("bottle", 0)}
- Cap:     {rejects.get("cap", 0)}
- Label:   {rejects.get("label", 0)}
- Shrink:  {rejects.get("shrink", 0)} kg
DEFECTIVE_QUANTITY: {kpis["defective_qty"]}
RISK_LEVEL: {risk_level}
AUDIT_STATUS: {audit_status}
"""

    # ── AI narrative (system prompt UPDATED) ─────────────────────────────────
    loop = asyncio.get_running_loop()
    def call_ai():
        return ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a plant-level executive production analyst writing a professional shift summary report.

Write a well-structured executive summary covering:
- Operational performance against plan
- Downtime impact: state which category (MECHANICAL / ELECTRICAL / UTILITY) caused the most loss
  and what that means for equipment or infrastructure reliability
- Quality performance based on reject breakdowns
- Clear conclusions about shift stability

FORMATTING RULES (strict):
- Output 3–4 separate paragraphs. Do NOT merge into one block.
- Each paragraph: 2–3 sentences. One idea per paragraph.
- One blank line between paragraphs.

WRITING STYLE:
- Proper grammar, capitalization, punctuation.
- Every sentence starts with a capital letter.
- Numeric format for all numbers (42%, 130 min, 4,420 packs).
- Do NOT convert numbers to words.
- Analytical, concise, executive-level.
- Base conclusions strictly on the structured data provided.
"""
                },
                {"role": "user", "content": structured_data},
            ],
            temperature=0.2,
        )
    response          = await loop.run_in_executor(None, call_ai)
    executive_paragraph = response.choices[0].message.content.strip()

    # ── Report sections ───────────────────────────────────────────────────────
    production_performance = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {production_data['product_type']}\n"
        f"  • Plan: {plan_output:,} packs\n"
        f"  • Actual: {actual_output:,} packs\n"
        f"  • Available Time: {available_time_minutes} minutes\n"
        f"  • Efficiency: {kpis['performance']}%"
    )
    if vos_info:
        production_performance += f"\n  • VOS: {vos_info}"

    # ── UPDATED downtime_analysis with categories ─────────────────────────────
    downtime_hours_display = round(total_downtime / 60, 1)
    available_hours_display = round(available_time_minutes / 60, 1)
    downtime_analysis = (
        f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
        f"  • Total Downtime:     {total_downtime} minutes\n"
        f"  • Downtime Ratio:     {downtime_ratio}%({downtime_hours_display}hr) of {available_hours_display}hr(available time)\n"
        f"  • Dominant Category:  {dominant_cat} ({dt_totals.get(dominant_cat, 0)} min)\n"
        f"  • Mechanical:         {dt_totals.get('MECHANICAL', 0)} min\n"
        f"  • Electrical:         {dt_totals.get('ELECTRICAL', 0)} min\n"
        f"  • Utility:            {dt_totals.get('UTILITY', 0)} min"
        f"{format_downtime_category_block(categorized_dt)}"
    )

    quality_metrics = (
        f"\n\n✓ QUALITY METRICS\n\n"
        f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
        f"  • Bottle Rejects:  {rejects.get('bottle', 0):,} pcs\n"
        f"  • Cap Rejects:     {rejects.get('cap', 0):,} pcs\n"
        f"  • Label Rejects:   {rejects.get('label', 0):,} pcs\n"
        f"  • Shrink Loss:     {rejects.get('shrink', 0)} kg"
    )

    production_performance_kpi = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {production_data['product_type']} Ltr\n"
        f"  • Plan: {plan_output:,} pcs\n"
        f"  • Actual: {actual_output:,} pcs\n"
        f"  • Achievement: {kpis['performance']:.1f}%"
    )

    reject_table = f"""
🗑 QUALITY – REJECT ANALYSIS
-------------------------

{'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
{'-' * 47}
Preform              {kpis['reject_percentages']['preform']:.1f} %           {rejects.get('preform', 0)} pcs
Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {rejects.get('bottle', 0)} pcs
Cap                  {kpis['reject_percentages']['cap']:.1f} %          {rejects.get('cap', 0)} pcs
Label                {kpis['reject_percentages']['label']:.1f} %          {rejects.get('label', 0)} pcs
Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {rejects.get('shrink', 0)} kg
"""

    oee_performance = (
        f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
        f"  • Plan: {plan_output:,} pcs\n"
        f"  • Actual: {actual_output:,} pcs\n"
        f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
        f"  • Production Time: {production_hours:.1f} hr\n"
        f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
        f"  • Availability: {kpis['availability']:.1f}%\n"
        f"  • Performance: {kpis['performance']:.1f}%\n"
        f"  • Quality: {kpis['quality']:.1f}%\n"
        f"  • OEE: {kpis['oee']:.2f}%"
    )

    final_report = (
        f"✅ STATUS: COMPLETE\n\n"
        f"⚠️ RISK LEVEL: {risk_level}\n\n"
        f"📋 EXECUTIVE SUMMARY\n\n"
        f"{executive_paragraph}\n\n"
        f"────────────────────────────\n\n"
        f"{production_performance}"
        f"{downtime_analysis}"
        f"{quality_metrics}"
        f"\n\n────────────────────────────\n\n"
        f"{reject_table}"
        f"\n\n────────────────────────────\n\n"
        f"{production_performance_kpi}"
        f"\n\n────────────────────────────\n\n"
        f"{oee_performance}"
        f"\n\n────────────────────────────\n\n"
        f"📌 AUDIT STATUS: {audit_status}"
    )
    return final_report.strip()

async def ai_generate_hourly_summary_from_text(report_text: str):
    try:
        production_data = parse_report(report_text)
    except Exception:
        return "DATA INCOMPLETE – production report missing."

    categorized_dt  = parse_downtime_categorized(report_text)   # ← NEW
    downtime        = flatten_categorized_downtime(categorized_dt)  # ← NEW
    rejects         = parse_rejects(report_text)
    vos_info        = parse_vos(report_text)
    dt_totals       = categorized_dt["_totals"]
    dominant_cat    = max(dt_totals, key=dt_totals.get) if any(dt_totals.values()) else "N/A"

    total_downtime         = sum(d["duration"] for d in downtime)
    actual_output          = production_data["actual"]
    plan_output            = production_data["plan"]
    available_time_minutes = production_data.get("available_time") or int(
        get_default_production_hours("hourly") * 60
    )
    production_hours = available_time_minutes / 60

    kpis           = compute_kpis(plan_output, actual_output, total_downtime, production_hours, rejects)
    downtime_ratio  = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes else 0

    # ── Risk (unchanged logic) ────────────────────────────────────────────────
    risk_score = 0
    if kpis["performance"] < 60:   risk_score += 3
    elif kpis["performance"] < 75: risk_score += 2
    if downtime_ratio > 40:        risk_score += 3
    elif downtime_ratio > 25:      risk_score += 2
    total_rejects_count = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
    if actual_output > 0:
        rr = (total_rejects_count / actual_output) * 100
        if rr > 5:   risk_score += 2
        elif rr > 2: risk_score += 1
    downtime_text = " ".join(d["description"] for d in downtime).lower()
    if any(w in downtime_text for w in ("misalignment", "wear")):     risk_score += 1
    if any(w in downtime_text for w in ("short circuit", "breaker")): risk_score += 1
    if any(w in downtime_text for w in ("glue", "adhesive")):         risk_score += 1
    risk_level   = ("CRITICAL" if risk_score >= 7 else "HIGH" if risk_score >= 5
                    else "MODERATE" if risk_score >= 3 else "LOW")
    audit_status = "FOLLOW-UP REQUIRED"

    # ── Structured data (UPDATED) ─────────────────────────────────────────────
    structured_data = f"""
HOUR SHIFT: {production_data["shift"]}
DATE: {production_data["date"]}
PRODUCT: {production_data["product_type"]}
PLAN (hour): {production_data["plan"]}
ACTUAL (hour): {production_data["actual"]}
AVAILABLE_TIME: {available_time_minutes} minutes
PRODUCTION_HOURS: {production_hours:.1f}
PERFORMANCE: {kpis["performance"]}%
AVAILABILITY: {kpis["availability"]}%
QUALITY: {kpis["quality"]}%
OEE: {kpis["oee"]}%
TOTAL_DOWNTIME: {total_downtime} minutes
DOWNTIME_RATIO: {downtime_ratio}%

DOWNTIME BY CATEGORY:
  MECHANICAL: {dt_totals.get("MECHANICAL", 0)} min
  ELECTRICAL: {dt_totals.get("ELECTRICAL", 0)} min
  UTILITY:    {dt_totals.get("UTILITY", 0)} min
  DOMINANT:   {dominant_cat}

DOWNTIME DETAIL:
{format_downtime_category_block(categorized_dt)}

REJECTS_BREAKDOWN:
- Preform: {rejects.get("preform", 0)}
- Bottle:  {rejects.get("bottle", 0)}
- Cap:     {rejects.get("cap", 0)}
- Label:   {rejects.get("label", 0)}
- Shrink:  {rejects.get("shrink", 0)} kg
DEFECTIVE_QUANTITY: {kpis["defective_qty"]}
RISK_LEVEL: {risk_level}
AUDIT_STATUS: {audit_status}
"""

    loop = asyncio.get_running_loop()
    def call_ai():
        return ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a plant-level executive production analyst writing a professional hourly summary report.

Write a well-structured executive summary evaluating ONE HOUR of production:
- Operational performance against plan for this hour
- Downtime impact: state which category (MECHANICAL / ELECTRICAL / UTILITY) caused the most loss
- Quality performance based on reject breakdowns
- Clear conclusions about hour stability

FORMATTING RULES (strict):
- Output 3–4 separate paragraphs. Do NOT merge into one block.
- Each paragraph: 2–3 sentences. One idea per paragraph.
- One blank line between paragraphs.

WRITING STYLE:
- Proper grammar, capitalization, punctuation.
- Every sentence starts with a capital letter.
- Numeric format for all numbers (42%, 60 min, 4,420 packs).
- Do NOT convert numbers to words.
- Analytical, concise, executive-level.
- Base conclusions strictly on the structured data provided.
"""
                },
                {"role": "user", "content": structured_data},
            ],
            temperature=0.2,
        )
    response            = await loop.run_in_executor(None, call_ai)
    executive_paragraph = response.choices[0].message.content.strip()

    # ── Report sections ───────────────────────────────────────────────────────
    production_performance = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {production_data['product_type']}\n"
        f"  • Plan: {plan_output:,} packs\n"
        f"  • Actual: {actual_output:,} packs\n"
        f"  • Available Time: {available_time_minutes} minutes\n"
        f"  • Efficiency: {kpis['performance']}%"
    )
    if vos_info:
        production_performance += f"\n  • VOS: {vos_info}"

    downtime_hours_display = round(total_downtime / 60, 1)
    available_hours_display = round(available_time_minutes / 60, 1)
    downtime_analysis = (
        f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
        f"  • Total Downtime:     {total_downtime} minutes\n"
        f"  • Downtime Ratio:     {downtime_ratio}%({downtime_hours_display}hr) of {available_hours_display}hr(available time)\n"
        f"  • Dominant Category:  {dominant_cat} ({dt_totals.get(dominant_cat, 0)} min)\n"
        f"  • Mechanical:         {dt_totals.get('MECHANICAL', 0)} min\n"
        f"  • Electrical:         {dt_totals.get('ELECTRICAL', 0)} min\n"
        f"  • Utility:            {dt_totals.get('UTILITY', 0)} min"
        f"{format_downtime_category_block(categorized_dt)}"
    )

    quality_metrics = (
        f"\n\n✓ QUALITY METRICS\n\n"
        f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
        f"  • Bottle Rejects:  {rejects.get('bottle', 0):,} pcs\n"
        f"  • Cap Rejects:     {rejects.get('cap', 0):,} pcs\n"
        f"  • Label Rejects:   {rejects.get('label', 0):,} pcs\n"
        f"  • Shrink Loss:     {rejects.get('shrink', 0)} kg"
    )

    production_performance_kpi = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {production_data['product_type']} Ltr\n"
        f"  • Plan: {plan_output:,} pcs\n"
        f"  • Actual: {actual_output:,} pcs\n"
        f"  • Achievement: {kpis['performance']:.1f}%"
    )

    reject_table = f"""
🗑 QUALITY – REJECT ANALYSIS
-------------------------

{'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
{'-' * 47}
Preform              {kpis['reject_percentages']['preform']:.1f} %           {rejects.get('preform', 0)} pcs
Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {rejects.get('bottle', 0)} pcs
Cap                  {kpis['reject_percentages']['cap']:.1f} %          {rejects.get('cap', 0)} pcs
Label                {kpis['reject_percentages']['label']:.1f} %          {rejects.get('label', 0)} pcs
Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {rejects.get('shrink', 0)} kg
"""

    oee_performance = (
        f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
        f"  • Plan: {plan_output:,} pcs\n"
        f"  • Actual: {actual_output:,} pcs\n"
        f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
        f"  • Production Time: {production_hours:.1f} hr\n"
        f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
        f"  • Availability: {kpis['availability']:.1f}%\n"
        f"  • Performance: {kpis['performance']:.1f}%\n"
        f"  • Quality: {kpis['quality']:.1f}%\n"
        f"  • OEE: {kpis['oee']:.2f}%"
    )

    final_report = (
        f"✅ STATUS: COMPLETE\n\n"
        f"⚠️ RISK LEVEL: {risk_level}\n\n"
        f"📋 EXECUTIVE SUMMARY\n\n"
        f"{executive_paragraph}\n\n"
        f"────────────────────────────\n\n"
        f"{production_performance}"
        f"{downtime_analysis}"
        f"{quality_metrics}"
        f"\n\n────────────────────────────\n\n"
        f"{reject_table}"
        f"\n\n────────────────────────────\n\n"
        f"{production_performance_kpi}"
        f"\n\n────────────────────────────\n\n"
        f"{oee_performance}"
        f"\n\n────────────────────────────\n\n"
        f"📌 AUDIT STATUS: {audit_status}"
    )
    return final_report.strip()

async def ai_generate_multi_shift_summary(included_shifts: list[int]):
    if not included_shifts:
        return None

    target_date = None
    for shift in included_shifts:
        if ai_shift_evidence[shift]:
            for text in reversed(ai_shift_evidence[shift]):
                try:
                    production_data = parse_report(text)
                    if production_data and production_data.get("date"):
                        target_date = str(production_data["date"])
                        break
                except:
                    continue
        if target_date:
            break

    if not target_date:
        return None

    logger.info(f"ai_generate_multi_shift_summary: target_date={target_date}, shifts={included_shifts}")

    total_plan            = 0
    total_actual          = 0
    total_downtime        = 0
    total_production_hours = 0
    total_rejects         = {"preform": 0, "bottle": 0, "cap": 0, "label": 0, "shrink": 0}
    # ── Aggregated category totals across all shifts ──────────────────────────
    agg_cat_totals        = {"MECHANICAL": 0, "ELECTRICAL": 0, "UTILITY": 0}
    # Per-shift categorized downtime for display
    shift_categorized     = {}
    product_types         = []
    all_vos_info          = []
    shifts_with_data      = []

    for shift in (1, 2, 3):
        if not ai_shift_evidence[shift]:
            continue

        shift_production_data = None
        shift_categorized_dt  = None
        shift_rejects         = {}
        shift_vos_info        = None

        for text in reversed(ai_shift_evidence[shift]):
            try:
                production_data = parse_report(text)
                if production_data and str(production_data.get("date")) == target_date:
                    shift_production_data = production_data
                    shift_categorized_dt  = parse_downtime_categorized(text)  # ← NEW
                    shift_rejects         = parse_rejects(text)
                    shift_vos_info        = parse_vos(text)
                    break
            except:
                continue

        if not shift_production_data:
            continue

        shifts_with_data.append(shift)
        shift_flat_dt  = flatten_categorized_downtime(shift_categorized_dt)
        shift_dt_total = sum(d["duration"] for d in shift_flat_dt)
        shift_categorized[shift] = shift_categorized_dt

        total_plan    += shift_production_data["plan"]
        total_actual  += shift_production_data["actual"]
        total_downtime += shift_dt_total

        # Accumulate category totals
        for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY"):
            agg_cat_totals[cat] += shift_categorized_dt["_totals"].get(cat, 0)

        available_time_minutes = shift_production_data.get("available_time") or int(
            get_default_production_hours("shift", shift) * 60
        )
        total_production_hours += available_time_minutes / 60

        for category in total_rejects:
            total_rejects[category] = round(
                total_rejects[category] + shift_rejects.get(category, 0), 2
            )

        if shift_production_data["product_type"]:
            product_types.append(shift_production_data["product_type"])

        vos_val = shift_vos_info.strip() if shift_vos_info else "none"
        all_vos_info.append(f"Shift {shift}: {vos_val}")

    if total_plan == 0:
        return None

    dominant_cat           = max(agg_cat_totals, key=agg_cat_totals.get) if any(agg_cat_totals.values()) else "N/A"
    kpis                   = compute_kpis(total_plan, total_actual, total_downtime, total_production_hours, total_rejects)
    total_available_minutes = total_production_hours * 60
    downtime_ratio          = round((total_downtime / total_available_minutes) * 100, 2) if total_available_minutes else 0

    # Risk (unchanged logic)
    risk_score = 0
    if kpis["performance"] < 60:   risk_score += 3
    elif kpis["performance"] < 75: risk_score += 2
    if downtime_ratio > 40:        risk_score += 3
    elif downtime_ratio > 25:      risk_score += 2
    total_reject_count = total_rejects.get("bottle", 0) + total_rejects.get("cap", 0) + total_rejects.get("label", 0)
    if total_actual > 0:
        rr = (total_reject_count / total_actual) * 100
        if rr > 5:   risk_score += 2
        elif rr > 2: risk_score += 1
    risk_level   = ("CRITICAL" if risk_score >= 7 else "HIGH" if risk_score >= 5
                    else "MODERATE" if risk_score >= 3 else "LOW")
    audit_status = "CLOSED"
    product_type_str = ", ".join(set(product_types)) if product_types else "Mixed"

    # ── Structured data (UPDATED) ─────────────────────────────────────────────
    structured_data = f"""
MULTI-SHIFT SUMMARY: ALL SHIFTS FOR {target_date}
DATE: {target_date}
PRODUCT(S): {product_type_str}
TOTAL PLAN: {total_plan:,}
TOTAL ACTUAL: {total_actual:,}
TOTAL PRODUCTION HOURS: {total_production_hours:.1f} hr
PERFORMANCE: {kpis["performance"]}%
AVAILABILITY: {kpis["availability"]}%
QUALITY: {kpis["quality"]}%
OEE: {kpis["oee"]}%
TOTAL DOWNTIME: {total_downtime} minutes
DOWNTIME_RATIO: {downtime_ratio}%

DOWNTIME BY CATEGORY (ALL SHIFTS COMBINED):
  MECHANICAL: {agg_cat_totals["MECHANICAL"]} min
  ELECTRICAL: {agg_cat_totals["ELECTRICAL"]} min
  UTILITY:    {agg_cat_totals["UTILITY"]} min
  DOMINANT:   {dominant_cat}

REJECTS_BREAKDOWN:
- Preform: {total_rejects.get("preform", 0):,}
- Bottle:  {total_rejects.get("bottle", 0):,}
- Cap:     {total_rejects.get("cap", 0):,}
- Label:   {total_rejects.get("label", 0):,}
- Shrink:  {total_rejects.get("shrink", 0):,} kg
DEFECTIVE_QUANTITY: {kpis["defective_qty"]:,}
RISK_LEVEL: {risk_level}
AUDIT_STATUS: {audit_status}
"""

    loop = asyncio.get_running_loop()
    def call_ai():
        return ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a plant-level executive production analyst writing a professional multi-shift summary.

Write a well-structured executive summary for the full 24-hour production on {target_date}:
- Overall operational performance against aggregated plan
- Downtime: state which category (MECHANICAL / ELECTRICAL / UTILITY) dominated across all shifts
  and what it implies for the plant
- Aggregate quality performance and reject patterns
- Clear conclusions about full-day stability

FORMATTING RULES (strict):
- Output 3–4 separate paragraphs. Do NOT merge into one block.
- Each paragraph: 2–3 sentences. One idea per paragraph.
- One blank line between paragraphs.

WRITING STYLE:
- Proper grammar, capitalization, punctuation.
- Every sentence starts with a capital letter.
- Numeric format for all numbers (42%, 350 min, 9,100 packs).
- Do NOT convert numbers to words.
- Analytical, concise, executive-level.
- Base conclusions strictly on the structured data provided.
"""
                },
                {"role": "user", "content": structured_data},
            ],
            temperature=0.2,
        )
    response            = await loop.run_in_executor(None, call_ai)
    executive_paragraph = response.choices[0].message.content.strip()

    # ── Production performance ────────────────────────────────────────────────
    production_performance = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {product_type_str}\n"
        f"  • Plan: {total_plan:,} packs\n"
        f"  • Actual: {total_actual:,} packs\n"
        f"  • Available Time: {total_available_minutes:.0f} minutes\n"
        f"  • Efficiency: {kpis['performance']}%\n"
        f"  • VOS:\n"
    )
    for vos_entry in all_vos_info:
        production_performance += f"      {vos_entry}\n"

    # ── UPDATED multi-shift downtime analysis with categories ─────────────────
    # Build per-shift category breakdown
    per_shift_detail = ""
    for shift in sorted(shift_categorized.keys()):
        scat = shift_categorized[shift]
        per_shift_detail += f"\n\n  Shift {shift}:\n"
        per_shift_detail += f"{format_downtime_category_block(scat)}"

    downtime_hours = round(total_downtime / 60, 1)
    available_hours = round(total_available_minutes / 60, 1)

    downtime_analysis = (
        f"\n\n⏱️ DOWNTIME ANALYSIS\n\n"
        f"  • Total Downtime:     {total_downtime} minutes\n"
        f"  • Downtime Ratio:     {downtime_ratio}%({downtime_hours}hr) of {available_hours}hr(available time)\n"
        f"  • Dominant Category:  {dominant_cat} ({agg_cat_totals.get(dominant_cat, 0)} min)\n"
        f"  • Mechanical (all):   {agg_cat_totals['MECHANICAL']} min\n"
        f"  • Electrical (all):   {agg_cat_totals['ELECTRICAL']} min\n"
        f"  • Utility (all):      {agg_cat_totals['UTILITY']} min"
        f"{per_shift_detail}"
    )



    quality_metrics = (
        f"\n\n✓ QUALITY METRICS\n\n"
        f"  • Preform Rejects: {total_rejects.get('preform', 0):,} pcs\n"
        f"  • Bottle Rejects:  {total_rejects.get('bottle', 0):,} pcs\n"
        f"  • Cap Rejects:     {total_rejects.get('cap', 0):,} pcs\n"
        f"  • Label Rejects:   {total_rejects.get('label', 0):,} pcs\n"
        f"  • Shrink Loss:     {round(total_rejects.get('shrink', 0), 2)} kg"
    )

    production_performance_kpi = (
        f"📊 PRODUCTION PERFORMANCE\n\n"
        f"  • Product: {product_type_str} Ltr\n"
        f"  • Plan: {total_plan:,} pcs\n"
        f"  • Actual: {total_actual:,} pcs\n"
        f"  • Achievement: {kpis['performance']:.1f}%"
    )

    reject_table = f"""
🗑 QUALITY – REJECT ANALYSIS
-------------------------

{'Item':<20} {'Reject %':<15} {'Reject Qty':<12}
{'-' * 47}
Preform              {kpis['reject_percentages']['preform']:.1f} %           {total_rejects.get('preform', 0)} pcs
Bottle               {kpis['reject_percentages']['bottle']:.1f} %          {total_rejects.get('bottle', 0)} pcs
Cap                  {kpis['reject_percentages']['cap']:.1f} %          {total_rejects.get('cap', 0)} pcs
Label                {kpis['reject_percentages']['label']:.1f} %          {total_rejects.get('label', 0)} pcs
Shrink               {kpis['reject_percentages']['shrink']:.1f} %           {round(total_rejects.get('shrink', 0), 2)} kg
"""

    oee_performance = (
        f"📈 OVERALL EQUIPMENT EFFECTIVENESS\n\n"
        f"  • Plan: {total_plan:,} pcs\n"
        f"  • Actual: {total_actual:,} pcs\n"
        f"  • Defective Quantity: {kpis['defective_qty']:,} pcs\n"
        f"  • Production Time: {total_production_hours:.1f} hr\n"
        f"  • Downtime: {kpis['downtime_hours']:.1f} hr\n"
        f"  • Availability: {kpis['availability']:.1f}%\n"
        f"  • Performance: {kpis['performance']:.1f}%\n"
        f"  • Quality: {kpis['quality']:.1f}%\n"
        f"  • OEE: {kpis['oee']:.2f}%"
    )

    final_report = (
        f"✅ STATUS: COMPLETE\n\n"
        f"⚠️ RISK LEVEL: {risk_level}\n\n"
        f"📋 EXECUTIVE SUMMARY\n\n"
        f"{executive_paragraph}\n\n"
        f"────────────────────────────\n\n"
        f"{production_performance}"
        f"{downtime_analysis}"
        f"{quality_metrics}"
        f"\n\n────────────────────────────\n\n"
        f"{reject_table}"
        f"\n\n────────────────────────────\n\n"
        f"{production_performance_kpi}"
        f"\n\n────────────────────────────\n\n"
        f"{oee_performance}"
        f"\n\n────────────────────────────\n\n"
        f"📌 AUDIT STATUS: {audit_status}"
    )
    return final_report.strip()

# ---------------- PRODUCTION VALIDATION ENGINE ----------------

def calculate_expected_production(
    plan: int,
    shift: int,
    available_time_minutes: int = None,
    downtime_minutes: int = 0,
    report_type: str = "hourly",
) -> dict:
    """
    Calculate what production SHOULD have been given the reported downtime.
    downtime_minutes here already includes VOS — caller adds it before passing.
    """
    if report_type == "hourly":
        total_minutes = available_time_minutes or 60
    elif report_type == "shift":
        total_minutes = available_time_minutes or get_shift_duration_minutes(shift)
    else:
        total_minutes = available_time_minutes or 60

    rate_per_minute = plan / total_minutes if total_minutes > 0 else 0
    net_production_minutes = max(0, total_minutes - downtime_minutes)
    expected_output = round(rate_per_minute * net_production_minutes)

    tolerance_pct = 0.05
    minimum_acceptable = round(expected_output * (1 - tolerance_pct))

    max_tolerance_pct = 0.10
    maximum_possible = round(rate_per_minute * net_production_minutes * (1 + max_tolerance_pct))

    return {
        "total_minutes": total_minutes,
        "downtime_minutes": downtime_minutes,
        "net_production_minutes": net_production_minutes,
        "rate_per_minute": round(rate_per_minute, 2),
        "expected_output": expected_output,
        "minimum_acceptable": minimum_acceptable,
        "maximum_possible": maximum_possible,
        "max_tolerance_pct": max_tolerance_pct,
        "tolerance_pct": tolerance_pct,
    }


def validate_production(
    plan: int,
    actual: int,
    downtime_minutes: int,
    shift: int,
    available_time_minutes: int = None,
    report_type: str = "hourly",
    vos_minutes: int = 0,
) -> dict:
    """
    Validate whether actual production matches what's mathematically possible.
    VOS is passed in separately so it's visible in the result, but it is added
    to downtime_minutes before any math — same treatment as mechanical/electrical.
    """
    total_downtime = downtime_minutes + vos_minutes

    expected = calculate_expected_production(
        plan=plan,
        shift=shift,
        available_time_minutes=available_time_minutes,
        downtime_minutes=total_downtime,
        report_type=report_type,
    )

    issues = []
    gap = 0
    gap_minutes = 0
    severity = "NONE"

    # ── Check 1: Actual lower than expected after downtime ──────────────
    if actual < expected["minimum_acceptable"]:
        gap = expected["expected_output"] - actual
        gap_minutes = round(gap / expected["rate_per_minute"]) if expected["rate_per_minute"] > 0 else 0
        gap_pct = (gap / expected["expected_output"] * 100) if expected["expected_output"] > 0 else 0

        if gap_pct > 20:
            severity = "CRITICAL"
        elif gap_pct > 10:
            severity = "SIGNIFICANT"
        elif gap_pct > 5:
            severity = "MINOR"

        issues.append({
            "type": "PRODUCTION_GAP",
            "message": (
                f"Expected ~{expected['expected_output']:,} packs based on "
                f"{expected['net_production_minutes']} min net production time "
                f"(plan rate: {expected['rate_per_minute']:.1f}/min), "
                f"but actual is {actual:,}. "
                f"Gap of {gap:,} packs (~{gap_minutes} min unaccounted)."
            ),
            "gap": gap,
            "gap_minutes": gap_minutes,
            "gap_pct": round(gap_pct, 1),
        })

    # ── Check 2: Downtime suspiciously low for the output gap ───────────
    plan_achievement = (actual / plan * 100) if plan > 0 else 0
    downtime_ratio = (total_downtime / expected["total_minutes"] * 100) if expected["total_minutes"] > 0 else 0

    if plan_achievement < 70 and downtime_ratio < 10:
        missing_minutes = round((plan - actual) / expected["rate_per_minute"]) if expected["rate_per_minute"] > 0 else 0
        # BUG FIX #1: missing_minutes - total_downtime can go negative when
        # total_downtime > missing_minutes. Guard with max(0, ...).
        unaccounted = max(0, missing_minutes - total_downtime)
        issues.append({
            "type": "DOWNTIME_UNDERREPORTED",
            "message": (
                f"Achievement is only {plan_achievement:.1f}% but only "
                f"{total_downtime} min total downtime reported "
                f"({downtime_ratio:.1f}% of available time). "
                f"The production shortfall suggests ~{missing_minutes} min of lost time. "
                f"Where are the missing ~{unaccounted} minutes?"
            ),
            "missing_minutes": unaccounted,
        })

    # ── Check 3: Downtime exceeds or equals available time ────────────────────────
    if total_downtime >= expected["total_minutes"]:
        if total_downtime > expected["total_minutes"]:
            issue_type = "DOWNTIME_EXCEEDS_AVAILABLE"
            message = (
                f"Total downtime ({total_downtime} min, incl. {vos_minutes} min VOS) "
                f"exceeds available time ({expected['total_minutes']} min). "
                f"Please verify downtime entries."
            )
        else:  # total_downtime == expected["total_minutes"]
            issue_type = "ZERO_PRODUCTION_TIME"
            message = (
                f"Total downtime ({total_downtime} min, incl. {vos_minutes} min VOS) "
                f"consumes the entire available time ({expected['total_minutes']} min). "
                f"There is ZERO time available for production. "
                f"{'Any production reported is impossible.' if actual > 0 else 'Zero production is expected.'}"
            )
        
        issues.append({
            "type": issue_type,
            "message": message,
        })

    # ── Check 4: Actual exceeds plan significantly ───────────────────────
    if actual > plan * 1.15:
        issues.append({
            "type": "OVER_PRODUCTION",
            "message": (
                f"Actual ({actual:,}) exceeds plan ({plan:,}) by "
                f"{((actual / plan - 1) * 100):.1f}%. "
                f"Was the plan updated during the period?"
            ),
        })

    # ── Check 5: Zero actual with zero downtime ──────────────────────────
    if actual == 0 and total_downtime == 0:
        issues.append({
            "type": "ZERO_OUTPUT_NO_DOWNTIME",
            "message": "Zero output reported with zero downtime. This requires justification.",
        })

    # ── Check 5b: Unrealistic production in very short time window ───────────
    if (0 < expected["net_production_minutes"] <= 15 and actual > 0):
        # For very short production windows (≤15 min), apply stricter validation
        realistic_max_rate = expected["rate_per_minute"] * 1.5  # Allow 50% above normal rate for short bursts
        actual_rate = actual / expected["net_production_minutes"]
        
        if actual_rate > realistic_max_rate:
            excess_rate = actual_rate - realistic_max_rate
            excess_rate_pct = (excess_rate / realistic_max_rate) * 100
            
            check_severity = "CRITICAL" if excess_rate_pct > 100 else "SIGNIFICANT" if excess_rate_pct > 50 else "MINOR"
            severity_rank = {"NONE": 0, "MINOR": 1, "SIGNIFICANT": 2, "CRITICAL": 3}
            if severity_rank.get(check_severity, 0) > severity_rank.get(severity, 0):
                severity = check_severity
            
            issues.append({
                "type": "UNREALISTIC_SHORT_TIME_PRODUCTION",
                "message": (
                    f"UNREALISTIC PRODUCTION RATE: "
                    f"With only {expected['net_production_minutes']} min available for production, "
                    f"you reported {actual:,} packs, which equals {actual_rate:.1f} packs/min. "
                    f"The planned rate is {expected['rate_per_minute']:.1f} packs/min. "
                    f"Even allowing for short bursts, the maximum realistic rate is {realistic_max_rate:.1f} packs/min. "
                    f"Your reported rate exceeds this by {excess_rate_pct:.0f}%. "
                    f"This level of output in such a short time window is physically implausible. "
                    f"Please provide evidence of machine speed settings or verify the production time/downtime entries."
                ),
                "actual_rate": actual_rate,
                "realistic_max_rate": realistic_max_rate,
                "excess_rate_pct": excess_rate_pct,
                "production_time_minutes": expected["net_production_minutes"],
            })

    # ── Check 5c: Unrealistic production in medium-short time window ─────────
    elif (15 < expected["net_production_minutes"] <= 30 and actual > 0):
        # For medium-short production windows (16-30 min), apply moderate validation
        realistic_max_rate = expected["rate_per_minute"] * 1.3  # Allow 30% above normal rate
        actual_rate = actual / expected["net_production_minutes"]
        
        if actual_rate > realistic_max_rate:
            excess_rate = actual_rate - realistic_max_rate
            excess_rate_pct = (excess_rate / realistic_max_rate) * 100
            
            check_severity = "CRITICAL" if excess_rate_pct > 80 else "SIGNIFICANT" if excess_rate_pct > 40 else "MINOR"
            severity_rank = {"NONE": 0, "MINOR": 1, "SIGNIFICANT": 2, "CRITICAL": 3}
            if severity_rank.get(check_severity, 0) > severity_rank.get(severity, 0):
                severity = check_severity
            
            issues.append({
                "type": "UNREALISTIC_MEDIUM_TIME_PRODUCTION",
                "message": (
                    f"QUESTIONABLE PRODUCTION RATE: "
                    f"With only {expected['net_production_minutes']} min available for production, "
                    f"you reported {actual:,} packs, which equals {actual_rate:.1f} packs/min. "
                    f"The planned rate is {expected['rate_per_minute']:.1f} packs/min. "
                    f"The maximum realistic rate for this time window is {realistic_max_rate:.1f} packs/min. "
                    f"Your reported rate exceeds this by {excess_rate_pct:.0f}%. "
                    f"This requires verification of machine performance or production time calculations."
                ),
                "actual_rate": actual_rate,
                "realistic_max_rate": realistic_max_rate,
                "excess_rate_pct": excess_rate_pct,
                "production_time_minutes": expected["net_production_minutes"],
            })

    # ── Check 6: Exaggerated output — exceeds physical maximum ───────────
    if expected["net_production_minutes"] > 0 and actual > expected["maximum_possible"]:
        excess = actual - expected["maximum_possible"]
        excess_pct = round((excess / expected["maximum_possible"]) * 100, 1) if expected["maximum_possible"] > 0 else 0
        extra_minutes_needed = round(excess / expected["rate_per_minute"]) if expected["rate_per_minute"] > 0 else 0

        if excess_pct > 50:
            exag_severity = "CRITICAL"
        elif excess_pct > 25:
            exag_severity = "SIGNIFICANT"
        else:
            exag_severity = "MINOR"

        severity_rank = {"NONE": 0, "MINOR": 1, "SIGNIFICANT": 2, "CRITICAL": 3}
        if severity_rank.get(exag_severity, 0) > severity_rank.get(severity, 0):
            severity = exag_severity

        issues.append({
            "type": "EXAGGERATED_OUTPUT",
            "message": (
                f"EXAGGERATED PRODUCTION DETECTED: "
                f"With {total_downtime} min total downtime "
                f"({vos_minutes} min VOS + {downtime_minutes} min internal), "
                f"only {expected['net_production_minutes']} min of production time remains. "
                f"At plan rate of {expected['rate_per_minute']:.1f} packs/min, "
                f"maximum possible output is ~{expected['maximum_possible']:,} packs "
                f"(including {expected['max_tolerance_pct']*100:.0f}% tolerance). "
                f"But actual reported is {actual:,} packs — "
                f"that's {excess:,} packs MORE than physically possible "
                f"({excess_pct}% over maximum). "
                f"This would require an EXTRA {extra_minutes_needed} minutes of production time "
                f"that does not exist."
            ),
            "excess": excess,
            "excess_pct": excess_pct,
            "extra_minutes_needed": extra_minutes_needed,
        })

    # ── Check 7: Cross-validate downtime vs actual ───────────────────────
    if total_downtime > 0 and expected["net_production_minutes"] < expected["total_minutes"]:
        plan_pct = (actual / plan * 100) if plan > 0 else 0
        time_pct = (expected["net_production_minutes"] / expected["total_minutes"] * 100) if expected["total_minutes"] > 0 else 0

        if plan_pct > 90 and time_pct < 70:
            issues.append({
                "type": "INCONSISTENT_DOWNTIME_VS_OUTPUT",
                "message": (
                    f"INCONSISTENCY: {plan_pct:.1f}% of plan achieved "
                    f"but only {time_pct:.1f}% of time was available "
                    f"({expected['net_production_minutes']} of {expected['total_minutes']} min). "
                    f"Achieving {plan_pct:.1f}% output in {time_pct:.1f}% of time would require "
                    f"the line to run {(plan_pct/time_pct*100):.0f}% faster than planned rate. "
                    f"Either the downtime is overstated or the output count is inflated."
                ),
            })

    return {
        "is_valid": len(issues) == 0,
        "severity": severity if issues else "NONE",
        "expected": expected,
        "actual": actual,
        "plan": plan,
        "gap": gap,
        "gap_minutes": gap_minutes,
        "issues": issues,
        "vos_minutes": vos_minutes,
        "internal_downtime_minutes": downtime_minutes,
        "total_downtime_minutes": total_downtime,
    }


# ---------------- SKIP DETECTION ----------------

SKIP_PHRASES = {
    "skip", "s", "pass", "ignore", "refuse", "no", "nope",
}

def is_skip_response(text: str) -> bool:
    """
    Returns True if the operator's message is clearly a skip/refuse.
    Matches exact short tokens and common skip phrases case-insensitively.
    """
    normalized = text.strip().lower()
    if normalized in SKIP_PHRASES:
        return True
    for phrase in SKIP_PHRASES:
        if len(phrase) > 3 and normalized.startswith(phrase):
            return True
    return False


# ---------------- AI PRODUCTION QUESTIONING ----------------

async def generate_production_validation_questions(
    validation_result: dict,
    report_type: str = "hourly",
    shift: int = None,
    hour: int = None,
    downtime_events: list = None,
    categorized_dt: dict = None,
) -> str | None:
    """
    Generates audit questions when production numbers don't match downtime.
    Returns AI-generated questions, or None if production is valid.
    """
    if validation_result["is_valid"]:
        return None

    issues_text = "\n".join([
        f"- [{issue['type']}] {issue['message']}"
        for issue in validation_result["issues"]
    ])

    expected        = validation_result["expected"]
    vos_minutes     = validation_result.get("vos_minutes", 0)
    internal_dt     = validation_result.get("internal_downtime_minutes", expected["downtime_minutes"])
    total_dt        = validation_result.get("total_downtime_minutes", expected["downtime_minutes"])

    dt_detail = (
        f"\nDowntime breakdown:\n"
        f"  Internal (Mech + Elec + Utility) : {internal_dt} min\n"
        f"  VOS (external power outage)      : {vos_minutes} min\n"
        f"  ──────────────────────────────────\n"
        f"  TOTAL downtime                   : {total_dt} min\n"
    )

    if downtime_events:
        dt_detail += "\nOperator-reported events:\n"
        for d in downtime_events:
            cat = d.get("category", "UNKNOWN")
            dt_detail += f"  - {d['description']} ({d['duration']} min) [{cat}]\n"

    if categorized_dt:
        totals = categorized_dt.get("_totals", {})
        dt_detail += (
            f"\n  MECHANICAL : {totals.get('MECHANICAL', 0)} min\n"
            f"  ELECTRICAL : {totals.get('ELECTRICAL', 0)} min\n"
            f"  UTILITY    : {totals.get('UTILITY', 0)} min\n"
        )

    period_label = f"Hour {hour}" if report_type == "hourly" and hour else f"Shift {shift}"

    max_info = ""
    if expected.get("maximum_possible"):
        max_info = (
            f"\n- Maximum possible output "
            f"(with {expected['max_tolerance_pct'] * 100:.0f}% tolerance): "
            f"{expected['maximum_possible']:,} packs"
        )

    vos_policy = (
        f"\nVOS NOTE: {vos_minutes} min of VOS (external power outage) is already "
        f"counted in the {total_dt} min total downtime. VOS is valid and system-recorded — "
        f"do NOT question it. Only question the gap that remains AFTER all downtime is subtracted.\n"
    ) if vos_minutes > 0 else ""

    prompt = f"""
PRODUCTION VALIDATION ALERT — {period_label}

FACTS:
- Plan                  : {validation_result['plan']:,} packs
- Actual                : {validation_result['actual']:,} packs
- Available time        : {expected['total_minutes']} min
- Internal downtime     : {internal_dt} min  (mechanical + electrical + utility)
- VOS (external)        : {vos_minutes} min  (power outage — valid, system-recorded)
- Total downtime        : {total_dt} min
- Net production time   : {expected['net_production_minutes']} min
- Rate (from plan)      : {expected['rate_per_minute']:.1f} packs/min
- Expected output       : {expected['expected_output']:,} packs{max_info}
- Severity              : {validation_result['severity']}
{dt_detail}
{vos_policy}
DETECTED ISSUES:
{issues_text}

TASK:
Generate 2-3 SHORT, POINTED, NUMBERED audit questions targeting the UNEXPLAINED
gap — the gap that remains AFTER all {total_dt} min of downtime (incl. {vos_minutes} min VOS)
has already been accounted for.

Guidelines per issue type:
1. EXAGGERATED_OUTPUT  → question how actual exceeded physical maximum
2. PRODUCTION_GAP      → question where the missing packs/time went
3. INCONSISTENT        → "Which number is wrong — the downtime total or the output count?"
4. Always ask for physical evidence (machine counters, batch logs, supervisor sign-off).

RULES:
- Use exact numbers from above
- VOS is valid — never question it
- 1-2 sentences per question, firm audit tone
- Show the arithmetic that exposes the gap
- Do NOT summarize or suggest solutions
"""

    try:
        loop = asyncio.get_running_loop()

        def call_ai():
            return ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a production audit AI for a bottling plant. "
                            "VOS = external power outage — legitimate downtime, never question it. "
                            "Question ONLY the gap that remains after ALL downtime is subtracted. "
                            "Be direct, mathematical, and firm."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

        response = await loop.run_in_executor(None, call_ai)
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"AI production validation question error: {e}")
        # ── Rule-based fallback ──────────────────────────────────────────
        fallback_questions = []
        for i, issue in enumerate(validation_result["issues"], 1):
            if issue["type"] == "PRODUCTION_GAP":
                fallback_questions.append(
                    f"{i}. After subtracting all {total_dt} min downtime "
                    f"(incl. {vos_minutes} min VOS), net time = "
                    f"{expected['net_production_minutes']} min → expected "
                    f"~{expected['expected_output']:,} packs at "
                    f"{expected['rate_per_minute']:.1f}/min. "
                    f"Actual: {validation_result['actual']:,}. "
                    f"Where are the missing {issue['gap']:,} packs (~{issue['gap_minutes']} min)?"
                )
            elif issue["type"] == "DOWNTIME_UNDERREPORTED":
                fallback_questions.append(
                    f"{i}. Achievement is "
                    f"{(validation_result['actual'] / validation_result['plan'] * 100):.1f}% "
                    f"but total downtime is only {total_dt} min "
                    f"({vos_minutes} min of which is VOS). "
                    f"~{issue['missing_minutes']} min appear unaccounted. What caused this gap?"
                )
            else:
                fallback_questions.append(f"{i}. {issue['message']}")
        return "\n".join(fallback_questions) if fallback_questions else None


# ---------------- OPERATOR ANSWER EVALUATION ----------------

async def evaluate_operator_answer(
    session_key: str,
    operator_answer: str,
    validation_result: dict,
    conversation_history: list,
) -> dict:
    """
    AI evaluates whether the operator's answer convincingly explains
    the production gap. Returns verdict + follow-up or approval.

    Returns:
        {
            "verdict": "ACCEPTED" | "FOLLOW_UP" | "REJECTED",
            "ai_response": str,  # AI's evaluation text
            "reasoning": str,    # Why accepted/rejected
        }
    """
    expected = validation_result["expected"]
    issues = validation_result["issues"]

    issues_text = "\n".join([
        f"- [{issue['type']}] {issue['message']}"
        for issue in issues
    ])

    # Build conversation context
    conv_context = ""
    for entry in conversation_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if role == "ai_question":
            conv_context += f"\nAI QUESTION: {content}\n"
        elif role == "operator_answer":
            conv_context += f"\nOPERATOR ANSWER: {content}\n"
        elif role == "ai_evaluation":
            conv_context += f"\nAI EVALUATION: {content}\n"

    round_num = len([e for e in conversation_history if e.get("role") == "operator_answer"])

    prompt = f"""
PRODUCTION VALIDATION — EVALUATE OPERATOR RESPONSE

ORIGINAL ISSUE:
- Plan: {validation_result['plan']:,} packs
- Actual: {validation_result['actual']:,} packs
- Available time: {expected['total_minutes']} min
- Reported downtime: {expected['downtime_minutes']} min
- Net production time: {expected['net_production_minutes']} min
- Expected output: ~{expected['expected_output']:,} packs
- Gap: {validation_result['gap']:,} packs (~{validation_result['gap_minutes']} min)
- Severity: {validation_result['severity']}

DETECTED ISSUES:
{issues_text}

CONVERSATION SO FAR:
{conv_context}

LATEST OPERATOR ANSWER:
{operator_answer}

CURRENT ROUND: {round_num} of {MAX_VALIDATION_ROUNDS}

YOUR TASK:
Evaluate whether the operator's answer CONVINCINGLY explains the production gap.

ACCEPTANCE CRITERIA (ALL must be met):
1. The explanation must account for the SPECIFIC gap in packs/minutes
2. The reasons given must be technically plausible for a bottling plant
3. The total unaccounted time must be explained (not just "we had issues")
4. If additional downtime is mentioned, it should roughly match the gap

RESPOND WITH EXACTLY ONE OF:

Option A — If the answer IS convincing:
VERDICT: ACCEPTED
REASONING: [1-2 sentences explaining why the answer is acceptable]

Option B — If the answer is PARTIALLY convincing but needs clarification:
VERDICT: FOLLOW_UP
REASONING: [1 sentence on what's still unclear]
QUESTION: [ONE specific follow-up question about the remaining gap]

Option C — If the answer is NOT convincing and round >= {MAX_VALIDATION_ROUNDS}:
VERDICT: REJECTED
REASONING: [1-2 sentences explaining why the gap remains unjustified]

RULES:
- Be fair but rigorous
- Accept reasonable explanations (speed loss, micro-stoppages, startup time)
- Do NOT accept vague answers like "we had problems" without specifics
- If they mention additional downtime not in the original report, that's useful info
- After round {MAX_VALIDATION_ROUNDS}, you MUST choose ACCEPTED or REJECTED
- Numbers matter: does their explanation account for the gap mathematically?
"""

    try:
        loop = asyncio.get_running_loop()

        def call_ai():
            return ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a production audit validator. You evaluate operator "
                            "explanations for production gaps. You are fair but rigorous. "
                            "You accept reasonable technical explanations but reject vague ones. "
                            "You always respond with VERDICT: ACCEPTED, FOLLOW_UP, or REJECTED."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

        response = await loop.run_in_executor(None, call_ai)
        ai_text = response.choices[0].message.content.strip()

        # Parse verdict
        verdict = "FOLLOW_UP"  # default
        reasoning = ai_text
        follow_up_question = None

        if "VERDICT: ACCEPTED" in ai_text.upper() or "VERDICT:ACCEPTED" in ai_text.upper():
            verdict = "ACCEPTED"
        elif "VERDICT: REJECTED" in ai_text.upper() or "VERDICT:REJECTED" in ai_text.upper():
            verdict = "REJECTED"
        elif "VERDICT: FOLLOW_UP" in ai_text.upper() or "VERDICT:FOLLOW_UP" in ai_text.upper():
            verdict = "FOLLOW_UP"

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=\nQUESTION:|\nVERDICT:|\Z)", ai_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract follow-up question
        question_match = re.search(r"QUESTION:\s*(.+?)(?=\nVERDICT:|\nREASONING:|\Z)", ai_text, re.DOTALL | re.IGNORECASE)
        if question_match:
            follow_up_question = question_match.group(1).strip()

        # Force decision after max rounds
        if round_num >= MAX_VALIDATION_ROUNDS and verdict == "FOLLOW_UP":
            verdict = "REJECTED"
            reasoning += " Maximum validation rounds reached without satisfactory explanation."

        result = {
            "verdict": verdict,
            "ai_response": ai_text,
            "reasoning": reasoning,
        }
        if follow_up_question:
            result["follow_up_question"] = follow_up_question

        return result

    except Exception as e:
        logger.error(f"AI evaluation error: {e}")
        # On AI failure, accept to avoid blocking production
        return {
            "verdict": "ACCEPTED",
            "ai_response": "AI evaluation unavailable — accepting by default.",
            "reasoning": "AI service error — validation skipped.",
        }

# ---------------- HOURLY VALIDATION ----------------

def parse_vos_minutes(vos_value) -> int:
    """
    Parses VOS field which can be:
      - int/float  : 20       → 20
      - numeric str: "20"     → 20
      - text       : "power off 20 min" → 20
      - None / 0 / empty      → 0
    """
    if vos_value is None:
        return 0

    if isinstance(vos_value, (int, float)):
        return int(vos_value)

    try:
        return int(str(vos_value).strip())
    except ValueError:
        pass

    match = re.search(r'\b(\d+)\b', str(vos_value))
    if match:
        return int(match.group(1))

    return 0


# ---------------- SHIFT VALIDATION ----------------

async def validate_and_question_shift(
    context: ContextTypes.DEFAULT_TYPE,
    report_text: str,
    shift: int,
) -> dict | None:
    """
    Called after operator submits shift data.
    If gap found: posts questions, stores session, BLOCKS summary.
    Returns validation result (with session_key if blocking).
    """
    try:
        production_data = parse_report(report_text)
    except Exception:
        return None

    categorized_dt = parse_downtime_categorized(report_text)
    downtime = flatten_categorized_downtime(categorized_dt)
    total_downtime = sum(d["duration"] for d in downtime)
    
    # Parse VOS and extract minutes
    vos_info = parse_vos(report_text)
    vos_minutes = parse_vos_minutes(vos_info)
    total_downtime_with_vos = total_downtime + vos_minutes

    available_time = production_data.get("available_time") or get_shift_duration_minutes(shift)

    validation = validate_production(
        plan=production_data["plan"],
        actual=production_data["actual"],
        downtime_minutes=total_downtime,
        shift=shift,
        available_time_minutes=available_time,
        report_type="shift",
        vos_minutes=vos_minutes,
    )

    if validation["is_valid"]:
        return validation

    # Generate initial questions
    questions = await generate_production_validation_questions(
        validation_result=validation,
        report_type="shift",
        shift=shift,
        downtime_events=downtime,
        categorized_dt=categorized_dt,
    )

    if not questions:
        return validation

    session_key = f"shift_{shift}"

    # Store validation session — this BLOCKS the summary
    validation_sessions[session_key] = {
        "state": VALIDATION_STATE_PENDING,
        "validation_result": validation,
        "report_text": report_text,
        "shift": shift,
        "hour": None,
        "report_type": "shift",
        "conversation": [
            {"role": "ai_question", "content": questions}
        ],
        "rounds": 0,
        "verdict": None,
        "verdict_reasoning": None,
    }

    severity_icon = {
        "CRITICAL": "🔴",
        "SIGNIFICANT": "🟠",
        "MINOR": "🟡",
        "NONE": "🟢",
    }
    icon = severity_icon.get(validation["severity"], "⚠️")
    expected = validation["expected"]

    header = (
        f"{icon} PRODUCTION VALIDATION — Shift {shift} Summary\n\n"
        f"📊 Plan: {validation['plan']:,} | Actual: {validation['actual']:,}\n"
        f"⏱ Total Downtime: {total_downtime_with_vos} min"
        f"{' (VOS: ' + str(vos_minutes) + ' min)' if vos_minutes > 0 else ''} | "
        f"Available: {expected['total_minutes']} min | "
        f"Net: {expected['net_production_minutes']} min\n"
        f"📐 Expected: ~{expected['expected_output']:,} | "
        f"Gap: {validation['gap']:,} packs (~{validation['gap_minutes']} min)\n"
        f"⚠️ Severity: {validation['severity']}\n\n"
        f"❓ AI Questions:\n{questions}\n\n"
        f"⏳ Summary is HELD until these questions are answered.\n"
        f"Please reply with your explanation."
    )

    try:
        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=header,
        )
    except Exception as e:
        logger.error(f"Failed to send validation questions: {e}")

    # Add session_key to validation result so caller knows summary is blocked
    validation["_session_key"] = session_key
    validation["_blocked"] = True
    return validation


async def validate_and_question_hourly(
    context: ContextTypes.DEFAULT_TYPE,
    report_text: str,
    shift: int,
    hour: int,
) -> dict | None:
    """
    Called after operator submits hourly data.
    If gap found: posts questions, stores session, BLOCKS summary.
    """
    try:
        production_data = parse_report(report_text)
    except Exception:
        return None

    categorized_dt = parse_downtime_categorized(report_text)
    downtime = flatten_categorized_downtime(categorized_dt)
    total_downtime = sum(d["duration"] for d in downtime)
    
    # Parse VOS and extract minutes
    vos_info = parse_vos(report_text)
    vos_minutes = parse_vos_minutes(vos_info)
    total_downtime_with_vos = total_downtime + vos_minutes

    available_time = production_data.get("available_time") or 60

    validation = validate_production(
        plan=production_data["plan"],
        actual=production_data["actual"],
        downtime_minutes=total_downtime,
        shift=production_data["shift"],
        available_time_minutes=available_time,
        report_type="hourly",
        vos_minutes=vos_minutes,
    )

    if validation["is_valid"]:
        return validation

    questions = await generate_production_validation_questions(
        validation_result=validation,
        report_type="hourly",
        shift=shift,
        hour=hour,
        downtime_events=downtime,
        categorized_dt=categorized_dt,
    )

    if not questions:
        return validation

    session_key = f"hourly_{shift}_{hour}"

    validation_sessions[session_key] = {
        "state": VALIDATION_STATE_PENDING,
        "validation_result": validation,
        "report_text": report_text,
        "shift": shift,
        "hour": hour,
        "report_type": "hourly",
        "conversation": [
            {"role": "ai_question", "content": questions}
        ],
        "rounds": 0,
        "verdict": None,
        "verdict_reasoning": None,
    }

    severity_icon = {
        "CRITICAL": "🔴",
        "SIGNIFICANT": "🟠",
        "MINOR": "🟡",
        "NONE": "🟢",
    }
    icon = severity_icon.get(validation["severity"], "⚠️")
    expected = validation["expected"]

    header = (
        f"{icon} PRODUCTION VALIDATION — Shift {shift}, Hour {hour}\n\n"
        f"📊 Plan: {validation['plan']:,} | Actual: {validation['actual']:,}\n"
        f"⏱ Downtime: {total_downtime_with_vos} min"
        f"{' (VOS: ' + str(vos_minutes) + ' min)' if vos_minutes > 0 else ''} | Net time: {expected['net_production_minutes']} min\n"
        f"📐 Expected: ~{expected['expected_output']:,} | Gap: {validation['gap']:,} packs\n"
        f"⚠️ Severity: {validation['severity']}\n\n"
        f"❓ AI Questions:\n{questions}\n\n"
        f"⏳ Summary is HELD until these questions are answered.\n"
        f"Please reply with your explanation."
    )

    try:
        await context.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=header,
        )
    except Exception as e:
        logger.error(f"Failed to send validation questions: {e}")

    validation["_session_key"] = session_key
    validation["_blocked"] = True
    return validation


async def _release_summary_after_validation(
    context: ContextTypes.DEFAULT_TYPE,
    session: dict,
) -> None:
    """
    Called after validation is APPROVED or REJECTED.
    Generates the summary with validation verdict embedded.
    """
    report_type = session["report_type"]
    report_text = session.get("_report_text", "")
    verdict = session.get("verdict", "APPROVED")
    verdict_reasoning = session.get("verdict_reasoning", "")
    validation_result = session["validation_result"]

    # Build validation notice for the summary
    validation_notice = ""
    if verdict == "REJECTED":
        gap = validation_result["gap"]
        gap_min = validation_result["gap_minutes"]
        expected = validation_result["expected"]

        # Build conversation summary for the report
        conv_summary = ""
        for entry in session.get("conversation", []):
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "ai_question":
                conv_summary += f"\n  AI: {content[:200]}"
            elif role == "operator_answer":
                conv_summary += f"\n  Operator: {content[:200]}"

        validation_notice = (
            f"\n\n🚨 PRODUCTION VALIDATION — UNACCOUNTED LOSS\n"
            f"────────────────────────────\n"
            f"  • Expected Output: ~{expected['expected_output']:,} packs "
            f"(rate {expected['rate_per_minute']:.1f}/min × {expected['net_production_minutes']} min)\n"
            f"  • Actual Output: {validation_result['actual']:,} packs\n"
            f"  • Unaccounted Gap: {gap:,} packs (~{gap_min} min)\n"
            f"  • Severity: {validation_result['severity']}\n"
            f"  • Verdict: ❌ REJECTED — Operator explanation NOT convincing\n"
            f"  • Reason: {verdict_reasoning}\n"
            f"\n  📋 Validation Exchange:{conv_summary}\n"
            f"────────────────────────────"
        )
    elif verdict == "APPROVED":
        validation_notice = (
            f"\n\n✅ PRODUCTION VALIDATION — APPROVED\n"
            f"────────────────────────────\n"
            f"  • Gap of {validation_result['gap']:,} packs was identified\n"
            f"  • Operator explanation: ACCEPTED\n"
            f"  • Reason: {verdict_reasoning}\n"
            f"────────────────────────────"
        )

    if report_type == "shift":
        shift = session["shift"]
        pending_shift = session.get("_pending_shift", shift)

        try:
            ai_text = await ai_generate_summary(pending_shift)

            # Inject validation notice into the summary
            if validation_notice:
                insert_marker = "────────────────────────────\n\n📊 PRODUCTION PERFORMANCE"
                if insert_marker in ai_text:
                    ai_text = ai_text.replace(
                        insert_marker,
                        f"{validation_notice}\n\n{insert_marker}",
                        1
                    )
                else:
                    ai_text += validation_notice

            daily_ai_shift_summaries[pending_shift] = ai_text
            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📊 SHIFT {pending_shift} OFFICIAL SUMMARY\n\n{ai_text}",
            )
            shift_closed[pending_shift] = True
        except Exception as e:
            logger.error(f"Error generating shift summary after validation: {e}")

    elif report_type == "hourly":
        hour_label = session.get("_hour_label", f"Hour {session.get('hour', '?')}")

        try:
            ai_summary = await ai_generate_hourly_summary_from_text(report_text)

            if validation_notice:
                insert_marker = "────────────────────────────\n\n📊 PRODUCTION PERFORMANCE"
                if insert_marker in ai_summary:
                    ai_summary = ai_summary.replace(
                        insert_marker,
                        f"{validation_notice}\n\n{insert_marker}",
                        1
                    )
                else:
                    ai_summary += validation_notice

            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📝 HOURLY AI SUMMARY ({hour_label})\n\n{ai_summary}",
            )
        except Exception as e:
            logger.error(f"Error generating hourly summary after validation: {e}")

    # Clean up session
    session_key = f"{report_type}_{session['shift']}"
    if report_type == "hourly":
        session_key = f"hourly_{session['shift']}_{session.get('hour', 0)}"
    validation_sessions.pop(session_key, None)


# ---------------- MESSAGE HANDLER ----------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return
    if update.effective_user.is_bot:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 1: Check if there's an active validation session waiting
    # for an operator answer. This takes priority over everything.
    # ═══════════════════════════════════════════════════════════════════
    active_validation_key = context.user_data.get("active_validation_session")
    if active_validation_key and not text.startswith("/"):
        session = validation_sessions.get(active_validation_key)
        if session and session["state"] in (VALIDATION_STATE_PENDING, VALIDATION_STATE_FOLLOWUP):
            # Operator is answering validation questions
            session["conversation"].append({
                "role": "operator_answer",
                "content": text,
            })
            session["rounds"] += 1

            # AI evaluates the answer
            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text="🔍 Evaluating your explanation..."
            )

            eval_result = await evaluate_operator_answer(
                session_key=active_validation_key,
                operator_answer=text,
                validation_result=session["validation_result"],
                conversation_history=session["conversation"],
            )

            verdict = eval_result["verdict"]
            reasoning = eval_result["reasoning"]

            session["conversation"].append({
                "role": "ai_evaluation",
                "content": f"VERDICT: {verdict}\n{reasoning}",
            })

            if verdict == "ACCEPTED":
                # ✅ Operator's answer is convincing — approve and release summary
                session["state"] = VALIDATION_STATE_APPROVED
                session["verdict"] = "APPROVED"
                session["verdict_reasoning"] = reasoning

                await context.bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=(
                        f"✅ VALIDATION APPROVED\n\n"
                        f"{reasoning}\n\n"
                        f"📊 Generating summary now..."
                    ),
                )

                # Clear the active session
                context.user_data.pop("active_validation_session", None)

                # Now generate and post the summary
                await _release_summary_after_validation(context, session)
                return

            elif verdict == "FOLLOW_UP":
                # ❓ Need more info — ask follow-up
                session["state"] = VALIDATION_STATE_FOLLOWUP
                follow_up = eval_result.get("follow_up_question", "Please provide more details.")

                session["conversation"].append({
                    "role": "ai_question",
                    "content": follow_up,
                })

                remaining = MAX_VALIDATION_ROUNDS - session["rounds"]
                await context.bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=(
                        f"🔄 FOLLOW-UP REQUIRED\n\n"
                        f"{reasoning}\n\n"
                        f"❓ {follow_up}\n\n"
                        f"⏳ {remaining} attempt(s) remaining before final verdict."
                    ),
                )
                return

            elif verdict == "REJECTED":
                # ❌ Not convincing — declare unaccounted loss
                session["state"] = VALIDATION_STATE_REJECTED
                session["verdict"] = "REJECTED"
                session["verdict_reasoning"] = reasoning

                gap = session["validation_result"]["gap"]
                gap_min = session["validation_result"]["gap_minutes"]

                await context.bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=(
                        f"❌ VALIDATION REJECTED — UNACCOUNTED PRODUCTION LOSS\n\n"
                        f"{reasoning}\n\n"
                        f"⚠️ {gap:,} packs (~{gap_min} min) remain UNACCOUNTED.\n"
                        f"This will be flagged in the official summary.\n\n"
                        f"📊 Generating summary with loss notice..."
                    ),
                )

                context.user_data.pop("active_validation_session", None)

                # Generate summary WITH the rejection notice
                await _release_summary_after_validation(context, session)
                return

        else:
            # Session expired or invalid — clean up
            context.user_data.pop("active_validation_session", None)

    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 2: Shift summary two-step
    # ═══════════════════════════════════════════════════════════════════
    pending_shift = context.user_data.get("shift_summary_pending")
    if pending_shift is not None and text and not text.startswith("/"):
        context.user_data.pop("shift_summary_pending", None)
        try:
            ai_shift_evidence[pending_shift].append(text)
            shift_closed[pending_shift] = False

            # Save to DB
            try:
                production_data = parse_report(text)
                categorized_dt = parse_downtime_categorized(text)
                downtime = flatten_categorized_downtime(categorized_dt)
                rejects = parse_rejects(text)
                vos_info = parse_vos(text)
                save_to_database(production_data, downtime, rejects,
                                 vos_info=vos_info, shift_override=pending_shift)
            except Exception as e:
                logger.warning(f"Manual shift input DB save skipped: {e}")

            # Validate production — may BLOCK summary
            validation = await validate_and_question_shift(context, text, pending_shift)
            await asyncio.sleep(1)

            if validation and validation.get("_blocked"):
                # Summary is BLOCKED — store context for when validation completes
                session_key = validation["_session_key"]
                context.user_data["active_validation_session"] = session_key
                # Store the report text in session for later summary generation
                validation_sessions[session_key]["_pending_shift"] = pending_shift
                validation_sessions[session_key]["_report_text"] = text
                return
            else:
                # No gap or valid production — generate summary immediately
                ai_text = await ai_generate_summary(pending_shift)
                daily_ai_shift_summaries[pending_shift] = ai_text
                await context.bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=f"📊 SHIFT {pending_shift} OFFICIAL SUMMARY\n\n{ai_text}",
                )
                shift_closed[pending_shift] = True
        except Exception as e:
            logger.error(f"Error generating shift summary (manual): {e}")
            await update.message.reply_text(f"❌ Error generating shift summary: {e}")
        return

    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 3: Hourly summary two-step
    # ═══════════════════════════════════════════════════════════════════
    pending_hour = context.user_data.get("hourly_summary_pending")
    if pending_hour is not None and text and not text.startswith("/"):
        context.user_data.pop("hourly_summary_pending", None)
        hour_label = format_hour_range_12h(pending_hour)
        try:
            current_shift_num = get_shift_for_time(now_ethiopia())

            # ──────────────────────────────────────────────────────
            # ADD THIS BLOCK: Save hourly data to database
            # ──────────────────────────────────────────────────────
            try:
                production_data = parse_report(text)
                categorized_dt = parse_downtime_categorized(text)
                downtime = flatten_categorized_downtime(categorized_dt)
                rejects = parse_rejects(text)
                vos_info = parse_vos(text)
                save_hourly_to_database(
                    data=production_data,
                    downtime=downtime,
                    rejects=rejects,
                    hour_number=pending_hour,
                    vos_info=vos_info,
                    shift_override=current_shift_num,
                )
                logger.info(f"Hourly data saved: shift={current_shift_num}, hour={pending_hour}")
            except Exception as e:
                logger.warning(f"Hourly DB save skipped: {e}")
            # ──────────────────────────────────────────────────────

            # Validate production — may BLOCK summary
            validation = await validate_and_question_hourly(
                context, text, current_shift_num, pending_hour
            )
            await asyncio.sleep(1)

            if validation and validation.get("_blocked"):
                session_key = validation["_session_key"]
                context.user_data["active_validation_session"] = session_key
                validation_sessions[session_key]["_pending_hour"] = pending_hour
                validation_sessions[session_key]["_report_text"] = text
                validation_sessions[session_key]["_hour_label"] = hour_label
                return
            else:
                ai_summary = await ai_generate_hourly_summary_from_text(text)
                await context.bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=f"📝 HOURLY AI SUMMARY ({hour_label})\n\n{ai_summary}",
                )
        except Exception as e:
            logger.error(f"Error generating hourly summary: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
        return
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 4: AI Audit mode (existing behavior)
    # ═══════════════════════════════════════════════════════════════════
    if user_id not in active_users:
        return

    if not text.startswith("/"):
        target_shift = None
        try:
            parsed = parse_report(text)
            target_shift = parsed.get("shift")
        except Exception:
            target_shift = None
        if target_shift not in (1, 2, 3):
            target_shift = current_shift
        if not shift_closed[target_shift]:
            ai_shift_evidence[target_shift].append(text)
        if target_shift in (1, 2, 3):
            try:
                production_data = parse_report(text)
                categorized_dt = parse_downtime_categorized(text)
                downtime = flatten_categorized_downtime(categorized_dt)
                rejects = parse_rejects(text)
                vos_info = parse_vos(text)
                save_to_database(production_data, downtime, rejects,
                                 vos_info=vos_info, shift_override=target_shift)
                logger.info(f"Shift {target_shift} report saved to database (AI audit)")
            except Exception as e:
                logger.warning(f"AI audit DB save skipped: {e}")

    try:
        next_ai_question = await generate_ai_questions_for_message(user_id, text)

        if not next_ai_question:
            return

        if next_ai_question == "STOP":
            await context.bot.send_message(
                GROUP_CHAT_ID,
                "✅ Audit completed.\nAll observed issues have been addressed or scheduled.\nNo further AI questions."
            )
            active_users.discard(user_id)
            user_ai_sessions.pop(user_id, None)
            user_audit_state.pop(user_id, None)
            global ai_reminder_block
            ai_reminder_block = False
            await flush_pending_reminders(context.bot, reason="ai")
            return

        msg = f"❓ AI Question:\n{next_ai_question}"
        await context.bot.send_message(GROUP_CHAT_ID, msg)
        user_audit_state[user_id]["questions"] += 1

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text("Error processing message. Please try again.")

# ---------------- SCHEDULER ----------------
async def scheduled_audit(app, chat_id, message_text, delay_seconds):
    await asyncio.sleep(delay_seconds)
    # Anonymous trigger
    user_id = 0  # Dummy user for scheduler
    user_ai_sessions[user_id] = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
    question = await generate_ai_questions_for_message(user_id, message_text)
    if question:
        await app.bot.send_message(chat_id,
                                   f"📅 Scheduled Audit:\n❓ AI Question:\n{question}\n\n🛠 Operator answer:\n{message_text}")


async def remind_shift_plan(context: ContextTypes.DEFAULT_TYPE):
    global shift_plan_sent_today
    shift = context.job.data["shift"]
    now = now_ethiopia()
    today = now.date()
    today_iso = today.isoformat()

    key = f"shift_plan_fired_{today_iso}_{shift}"
    if bot_state_get(key):
        logger.info(f"Shift {shift} plan already fired today, skipping")
        return

    if line_state != LINE_STATE_RUNNING:
        logger.info(
            f"Shift {shift} plan suppressed (line={line_state}) "
            f"— key NOT written, catchup will resend on line_on"
        )
        return

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📋 *Shift {shift} Plan Reminder*\n\n"
            + "- Product type\n"
            + "- Shift plan (packs)\n"
            + "- Expected manpower / constraints"
    )
    result = await send_or_queue_reminder(context, text, parse_mode="Markdown")
    if result in ("sent", "queued"):
        bot_state_set(key, "1")
        shift_plan_sent_today[shift] = today
        logger.info(f"Shift {shift} plan reminder fired by scheduler at :02")
    else:
        logger.warning(f"Shift {shift} plan reminder NOT marked sent (delivery failed) — will retry on reconnect")


async def remind_shift_report(context: ContextTypes.DEFAULT_TYPE):
    shift = context.job.data["shift"]
    now = now_ethiopia()
    if not is_in_shift_summary_window(shift, now):
        logger.info(f"Shift {shift} summary outside window (min={now.minute}), skipping")
        return

    today_iso = now.date().isoformat()
    key = f"shift_report_fired_{today_iso}_{shift}"
    if bot_state_get(key):
        logger.info(f"Shift {shift} report already fired today, skipping")
        return

    if not _shift_had_any_production(shift, today_iso):
        logger.info(f"Shift {shift} had no production, skipping summary reminder")
        bot_state_set(key, "1")
        return

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📊 *Shift {shift} Summary Report Reminder*\n\n"
            + "- Actual output\n"
            + "- Downtime (reason + minutes)\n"
            + "- Rejects (preform, bottle, cap, label)\n"
            + "- Operator remarks"
    )
    result = await send_or_queue_reminder(context, text, parse_mode="Markdown")

    # Write key ONLY if actually sent or queued (not failed)
    if result in ("sent", "queued"):
        bot_state_set(key, "1")
        logger.info(f"Shift {shift} report reminder fired by scheduler at :55")
    else:
        logger.warning(f"Shift {shift} report reminder NOT marked sent (delivery failed) — will retry on reconnect")


async def remind_hourly_plan(context: ContextTypes.DEFAULT_TYPE):
    now = now_ethiopia()
    job_data = context.job.data or {}

    shift = job_data.get("shift") or get_shift_for_time(now)
    hour = job_data.get("hour") or get_current_hour_number(shift, now)
    today_iso = now.date().isoformat()

    if not is_in_hourly_plan_window(shift, hour, now):
        logger.info(
            f"Hourly plan Shift {shift} Hour {hour} outside window (min={now.minute}), skipping"
        )
        return

    sched_key = f"hourly_plan_scheduled_{today_iso}_{shift}_{hour}"
    catch_key = f"hourly_plan_{today_iso}_{shift}_{hour}"

    if bot_state_get(catch_key):
        logger.info(f"Hourly plan Shift {shift} Hour {hour} already sent, skipping")
        return

    # Line inactive — do NOT write DB key so catchup on line_on can resend
    if line_state != LINE_STATE_RUNNING:
        logger.info(
            f"Hourly plan Shift {shift} Hour {hour} suppressed (line={line_state}) "
            f"— key NOT written, catchup will resend on line_on"
        )
        return

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"⏰ *Hourly Plan Reminder – Shift {shift}, Hour {hour}*\n\n"
            + "Please share the plan for this hour:\n"
            + "- Production target\n"
            + "- Any scheduled maintenance or adjustments\n"
            + "- Expected challenges"
    )

    # ✅ Capture result
    success = await send_or_queue_reminder(
        context,
        text,
        parse_mode="Markdown"
    )

    # ✅ Write DB ONLY if message actually sent
    if success in ("sent", "queued"):
        bot_state_set(sched_key, "1")
        bot_state_set(catch_key, "1")
        logger.info(f"Hourly plan confirmed sent: Shift {shift} Hour {hour}")
    else:
        logger.warning(
            f"Hourly plan NOT marked sent (delivery failed): Shift {shift} Hour {hour}"
        )


async def remind_hourly_summary(context: ContextTypes.DEFAULT_TYPE):
    now = now_ethiopia()
    job_data = context.job.data or {}
    shift = job_data.get("shift") or get_shift_for_time(now)
    hour = job_data.get("hour") or get_current_hour_number(shift, now)

    if not is_in_hourly_summary_window(now, shift, hour):
        logger.info(f"Hourly summary outside window (min={now.minute}), skipping")
        return

    today_iso = now.date().isoformat()
    sched_key = f"hourly_summary_scheduled_{today_iso}_{shift}_{hour}"
    catch_key = f"hourly_summary_{today_iso}_{shift}_{hour}"

    if bot_state_get(sched_key):
        logger.info(f"Hourly summary Shift {shift} Hour {hour} already sent, skipping")
        return

    # No production this hour — mark and skip
    if not _hour_had_production_or_partial(shift, hour, today_iso):
        logger.info(f"Hourly summary Shift {shift} Hour {hour} — no production, skipping")
        bot_state_set(sched_key, "1")
        bot_state_set(catch_key, "1")
        return

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📝 *Hourly Summary Reminder – Shift {shift}, Hour {hour}*\n\n"
            + "Please provide hourly production data:\n"
            + "- Actual output for this hour\n"
            + "- Downtime events (if any)\n"
            + "- Rejects (if any)\n"
            + "- Operator notes\n\n"
            + "💡 AI will generate an hourly summary after you submit the data."
    )
    result = await send_or_queue_reminder(context, text, parse_mode="Markdown")

    # ✅ Only write DB keys if actually sent or queued (not failed)
    if result in ("sent", "queued"):
        bot_state_set(sched_key, "1")
        bot_state_set(catch_key, "1")
        logger.info(f"Hourly summary fired: Shift {shift} Hour {hour}")
    else:
        logger.warning(f"Hourly summary NOT marked sent (delivery failed): Shift {shift} Hour {hour}")


async def remind_daily_production_plan(context: ContextTypes.DEFAULT_TYPE):
    global daily_plan_last_date
    now = now_ethiopia()
    today = now.date()
    today_iso = today.isoformat()

    key = f"daily_plan_{today_iso}"
    if bot_state_get(key):
        logger.info("Daily plan already sent today (scheduled job), skipping")
        return

    if line_state != LINE_STATE_RUNNING:
        logger.info(
            f"Daily plan suppressed (line={line_state}) "
            f"— key NOT written, catchup will resend on line_on"
        )
        return

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + "📆 *Daily Production Plan Reminder*\n\n"
            + "Please share today's overall production plan:\n"
            + "- Products and SKUs by shift\n"
            + "- Target packs per shift\n"
            + "- Any known constraints (utilities, materials, manpower)."
    )
    result = await send_or_queue_reminder(context, text, parse_mode="Markdown")
    if result in ("sent", "queued"):
        bot_state_set(key, "1")
        daily_plan_last_date = today
        bot_state_set("daily_plan_last_date", today_iso)
        logger.info("Daily plan reminder fired by scheduler")
    else:
        logger.warning("Daily plan reminder NOT marked sent (delivery failed) — will retry on reconnect")


async def setup_shift_schedules(app):
    job_queue = app.job_queue

    # Clear all old jobs first to prevent stale jobs firing at wrong times
    for job in job_queue.jobs():
        job.schedule_removal()
    logger.info("Cleared old jobs from queue")

    logger.info("Setting up shift schedules and reminders...")

    # ── DAILY PLAN ──────────────────────────────────────────────────────────
    # Ethiopian 00:00 → PC 06:00 (primary, Shift 1 start)
    job_queue.run_daily(remind_daily_production_plan,
                        time=ethiopian_clock_time_to_pc_time(time(0, 0)),
                        name="daily_plan_shift1")
    # Fallbacks (once-per-day guard inside the function)
    job_queue.run_daily(remind_daily_production_plan,
                        time=ethiopian_clock_time_to_pc_time(time(7, 0)),
                        name="daily_plan_shift2")
    job_queue.run_daily(remind_daily_production_plan,
                        time=ethiopian_clock_time_to_pc_time(time(14, 0)),
                        name="daily_plan_shift3")

    # ════════════════════════════════════════════════════════════════════════
    # SHIFT 1 │ Ethiopian 00:00–07:00 │ PC 06:00–13:00
    # 12:02 Shift Plan, 12:05 Hr1 Plan, 12:55 Hr1 Summary
    # 1:02 Hr2 Plan,  1:55 Hr2 Summary
    # 2:02 Hr3 Plan,  2:55 Hr3 Summary
    # 3:02 Hr4 Plan,  3:55 Hr4 Summary
    # 4:02 Hr5 Plan,  4:55 Hr5 Summary
    # 5:02 Hr6 Plan,  5:55 Hr6 Summary
    # 6:02 Hr7 Plan,  6:50 Hr7 Summary,  6:55 Shift Summary
    # ════════════════════════════════════════════════════════════════════════
    job_queue.run_daily(remind_shift_plan,
                        time=ethiopian_clock_time_to_pc_time(time(0, 2)),
                        data={"shift": 1}, name="shift1_plan")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(0, 5)),
                        data={"shift": 1, "hour": 1}, name="shift1_hour1_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(0, 55)),
                        data={"shift": 1, "hour": 1}, name="shift1_hour1_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(1, 2)),
                        data={"shift": 1, "hour": 2}, name="shift1_hour2_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(1, 55)),
                        data={"shift": 1, "hour": 2}, name="shift1_hour2_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(2, 2)),
                        data={"shift": 1, "hour": 3}, name="shift1_hour3_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(2, 55)),
                        data={"shift": 1, "hour": 3}, name="shift1_hour3_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(3, 2)),
                        data={"shift": 1, "hour": 4}, name="shift1_hour4_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(3, 55)),
                        data={"shift": 1, "hour": 4}, name="shift1_hour4_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(4, 2)),
                        data={"shift": 1, "hour": 5}, name="shift1_hour5_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(4, 55)),
                        data={"shift": 1, "hour": 5}, name="shift1_hour5_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(5, 2)),
                        data={"shift": 1, "hour": 6}, name="shift1_hour6_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(5, 55)),
                        data={"shift": 1, "hour": 6}, name="shift1_hour6_summary")

    # Last hour: Hourly Summary at :50, Shift Summary at :55
    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(6, 2)),
                        data={"shift": 1, "hour": 7}, name="shift1_hour7_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(6, 50)),
                        data={"shift": 1, "hour": 7}, name="shift1_hour7_summary")
    job_queue.run_daily(remind_shift_report,
                        time=ethiopian_clock_time_to_pc_time(time(6, 55)),
                        data={"shift": 1}, name="shift1_report")

    logger.info("Shift 1 schedule registered")

    # ════════════════════════════════════════════════════════════════════════
    # SHIFT 2 │ Ethiopian 07:00–14:00 │ PC 13:00–20:00
    # 7:02 Shift Plan, 7:05 Hr1 Plan,  7:55 Hr1 Summary
    # 8:02 Hr2 Plan,   8:55 Hr2 Summary
    # 9:02 Hr3 Plan,   9:55 Hr3 Summary
    # 10:02 Hr4 Plan,  10:55 Hr4 Summary
    # 11:02 Hr5 Plan,  11:55 Hr5 Summary
    # 12:02 Hr6 Plan,  12:55 Hr6 Summary
    # 1:02 Hr7 Plan,   1:50 Hr7 Summary,  1:55 Shift Summary
    # ════════════════════════════════════════════════════════════════════════
    job_queue.run_daily(remind_shift_plan,
                        time=ethiopian_clock_time_to_pc_time(time(7, 2)),
                        data={"shift": 2}, name="shift2_plan")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(7, 5)),
                        data={"shift": 2, "hour": 1}, name="shift2_hour1_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(7, 55)),
                        data={"shift": 2, "hour": 1}, name="shift2_hour1_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(8, 2)),
                        data={"shift": 2, "hour": 2}, name="shift2_hour2_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(8, 55)),
                        data={"shift": 2, "hour": 2}, name="shift2_hour2_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(9, 2)),
                        data={"shift": 2, "hour": 3}, name="shift2_hour3_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(9, 55)),
                        data={"shift": 2, "hour": 3}, name="shift2_hour3_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(10, 2)),
                        data={"shift": 2, "hour": 4}, name="shift2_hour4_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(10, 55)),
                        data={"shift": 2, "hour": 4}, name="shift2_hour4_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(11, 2)),
                        data={"shift": 2, "hour": 5}, name="shift2_hour5_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(11, 55)),
                        data={"shift": 2, "hour": 5}, name="shift2_hour5_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(12, 2)),
                        data={"shift": 2, "hour": 6}, name="shift2_hour6_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(12, 55)),
                        data={"shift": 2, "hour": 6}, name="shift2_hour6_summary")

    # Last hour: Hourly Summary at :50, Shift Summary at :55
    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(13, 2)),
                        data={"shift": 2, "hour": 7}, name="shift2_hour7_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(13, 50)),
                        data={"shift": 2, "hour": 7}, name="shift2_hour7_summary")
    job_queue.run_daily(remind_shift_report,
                        time=ethiopian_clock_time_to_pc_time(time(13, 55)),
                        data={"shift": 2}, name="shift2_report")

    logger.info("Shift 2 schedule registered")

    # ════════════════════════════════════════════════════════════════════════
    # SHIFT 3 │ Ethiopian 14:00–24:00 │ PC 20:00–06:00
    # 2:02 Shift Plan, 2:05 Hr1 Plan,  2:55 Hr1 Summary
    # 3:02 Hr2 Plan,   3:55 Hr2 Summary
    # 4:02 Hr3 Plan,   4:55 Hr3 Summary
    # 5:02 Hr4 Plan,   5:55 Hr4 Summary
    # 6:02 Hr5 Plan,   6:55 Hr5 Summary
    # 7:02 Hr6 Plan,   7:55 Hr6 Summary
    # 8:02 Hr7 Plan,   8:55 Hr7 Summary
    # 9:02 Hr8 Plan,   9:55 Hr8 Summary
    # 10:02 Hr9 Plan,  10:55 Hr9 Summary
    # 11:02 Hr10 Plan, 11:50 Hr10 Summary, 11:55 Shift Summary
    # ════════════════════════════════════════════════════════════════════════
    job_queue.run_daily(remind_shift_plan,
                        time=ethiopian_clock_time_to_pc_time(time(14, 2)),
                        data={"shift": 3}, name="shift3_plan")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(14, 5)),
                        data={"shift": 3, "hour": 1}, name="shift3_hour1_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(14, 55)),
                        data={"shift": 3, "hour": 1}, name="shift3_hour1_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(15, 2)),
                        data={"shift": 3, "hour": 2}, name="shift3_hour2_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(15, 55)),
                        data={"shift": 3, "hour": 2}, name="shift3_hour2_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(16, 2)),
                        data={"shift": 3, "hour": 3}, name="shift3_hour3_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(16, 55)),
                        data={"shift": 3, "hour": 3}, name="shift3_hour3_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(17, 2)),
                        data={"shift": 3, "hour": 4}, name="shift3_hour4_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(17, 55)),
                        data={"shift": 3, "hour": 4}, name="shift3_hour4_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(18, 2)),
                        data={"shift": 3, "hour": 5}, name="shift3_hour5_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(18, 55)),
                        data={"shift": 3, "hour": 5}, name="shift3_hour5_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(19, 2)),
                        data={"shift": 3, "hour": 6}, name="shift3_hour6_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(19, 55)),
                        data={"shift": 3, "hour": 6}, name="shift3_hour6_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(20, 2)),
                        data={"shift": 3, "hour": 7}, name="shift3_hour7_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(20, 55)),
                        data={"shift": 3, "hour": 7}, name="shift3_hour7_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(21, 2)),
                        data={"shift": 3, "hour": 8}, name="shift3_hour8_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(21, 55)),
                        data={"shift": 3, "hour": 8}, name="shift3_hour8_summary")

    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(22, 2)),
                        data={"shift": 3, "hour": 9}, name="shift3_hour9_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(22, 55)),
                        data={"shift": 3, "hour": 9}, name="shift3_hour9_summary")

    # Last hour
    # Last hour: Hourly Summary at :50, Shift Summary at :55
    job_queue.run_daily(remind_hourly_plan,
                        time=ethiopian_clock_time_to_pc_time(time(23, 2)),
                        data={"shift": 3, "hour": 10}, name="shift3_hour10_plan")
    job_queue.run_daily(remind_hourly_summary,
                        time=ethiopian_clock_time_to_pc_time(time(23, 50)),
                        data={"shift": 3, "hour": 10}, name="shift3_hour10_summary")
    job_queue.run_daily(remind_shift_report,
                        time=ethiopian_clock_time_to_pc_time(time(23, 55)),
                        data={"shift": 3}, name="shift3_report")

    logger.info("Shift 3 schedule registered")
    logger.info("✅ All reminders scheduled successfully!")


# ---------------- BOT SETUP ----------------
async def setup_bot_commands(app):
    commands = [
        BotCommand("start_audit", "Start production audit manually"),
        BotCommand("end_audit", "End current audit"),
        BotCommand("shift_input_1", "Paste Shift 1 report (then auto-summary)"),
        BotCommand("shift_input_2", "Paste Shift 2 report (then auto-summary)"),
        BotCommand("shift_input_3", "Paste Shift 3 report (then auto-summary)"),
        BotCommand("shift_summary_1", "Shift 1 summary (post to group)"),
        BotCommand("shift_summary_2", "Shift 2 summary (post to group)"),
        BotCommand("shift_summary_3", "Shift 3 summary (post to group)"),
        BotCommand("all_shift_summary", "AI summary across closed shifts"),
        BotCommand("hourly_summary_ai", "Hourly AI summary (optional: hour 0-23)"),
        BotCommand("shift_report", "View shift reports (1+2 or all shifts)"),
        BotCommand("test_reminder", "Test reminder system (sends immediately)"),
        BotCommand("bot_status", "Check bot status and reminder state"),
        BotCommand("line_off", "Set line OFF (queue all reminders)"),
        BotCommand("line_on", "Set line ON (flush queued reminders)"),
        BotCommand("sanitation_start", "Start sanitation (queue all reminders)"),
        BotCommand("sanitation_end", "End sanitation (flush queued reminders)"),
    ]
    await app.bot.set_my_commands(commands)


async def send_daily_plan_if_needed(bot, now: datetime, skip_window_check: bool = False) -> bool:
    """
    Send daily plan catchup if not yet sent today.
    skip_window_check: when True (line_on, sanitation_end, reboot), post regardless of time.
    Otherwise only post within first 45 min of shift 1.
    """
    global daily_plan_last_date
    today = now.date()
    today_iso = today.isoformat()

    # Only skip if the catchup key was set — meaning we actually delivered it.
    # Do NOT skip based on daily_plan_{today_iso} alone — scheduler may have
    # been suppressed while line was OFF and never written that key (with the
    # new fix), but old stale keys from previous session could still exist.
    if bot_state_get(f"daily_plan_catchup_{today_iso}"):
        logger.info("Daily plan already sent today (catchup check), skipping")
        return False

    if not skip_window_check and not is_in_daily_plan_recovery_window(now):
        logger.info("Daily plan catchup: outside window, skipping")
        return False

    header = f"📅 {format_date_time_12h(now)}\n\n"
    daily_plan_text = (
            header
            + "📆 *Daily Production Plan Reminder*\n\n"
            + "Please share today's overall production plan:\n"
            + "- Products and SKUs by shift\n"
            + "- Target packs per shift\n"
            + "- Any known constraints (utilities, materials, manpower)."
    )
    try:
        await bot.send_message(chat_id=GROUP_CHAT_ID, text=daily_plan_text, parse_mode="Markdown")
        # Only mark as sent AFTER successful delivery
        bot_state_set(f"daily_plan_catchup_{today_iso}", "1")
        bot_state_set(f"daily_plan_{today_iso}", "1")
        daily_plan_last_date = today
        bot_state_set("daily_plan_last_date", today_iso)
        logger.info("Daily plan reminder sent (catchup)")
        return True
    except Exception as e:
        logger.error(f"Failed to send daily plan reminder (catchup): {e}")
        return False


async def send_shift_plan_if_needed(bot, current_shift_num: int, now: datetime,
                                    skip_window_check: bool = False) -> bool:
    """Send shift plan for current shift if not yet sent (once per shift)."""
    global shift_plan_sent_today
    today = now.date()
    today_iso = today.isoformat()

    actual_shift = get_shift_for_time(now)
    if actual_shift != current_shift_num:
        return False

    catch_key = f"shift_plan_catchup_{today_iso}_{current_shift_num}"
    fired_key = f"shift_plan_fired_{today_iso}_{current_shift_num}"

    # Only skip based on catch_key (confirmed delivery).
    # fired_key alone is NOT reliable — scheduler may have been suppressed
    # while line was OFF and never written it (with the new fix), but a
    # stale fired_key from a previous run could still block the catchup.
    if bot_state_get(catch_key):
        logger.info(f"Shift {current_shift_num} plan already sent, skipping catchup")
        return False

    if not skip_window_check:
        shift_start_minutes = {1: 6 * 60, 2: 13 * 60, 3: 20 * 60}
        start = shift_start_minutes[current_shift_num]
        current_minutes = now.hour * 60 + now.minute
        if current_shift_num == 3 and now.hour < 6:
            current_minutes += 24 * 60
            start = 20 * 60
        minutes_into_shift = current_minutes - start
        if minutes_into_shift < 0 or minutes_into_shift > 45:
            return False

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📋 *Shift {current_shift_num} Plan Reminder*\n\n"
            + "- Product type\n"
            + "- Shift plan (packs)\n"
            + "- Expected manpower / constraints"
    )
    try:
        await bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
        # Only mark as sent AFTER successful delivery
        bot_state_set(catch_key, "1")
        bot_state_set(fired_key, "1")
        shift_plan_sent_today[current_shift_num] = today
        logger.info(f"Shift {current_shift_num} plan sent (catchup)")
        return True
    except Exception as e:
        logger.error(f"Failed to send shift plan (catchup): {e}")
        return False


def get_current_hour_number(current_shift_num: int, now: datetime) -> int:
    """
    Get the current hour number within the shift based on the Ethiopian clock shift model
    converted to PC/international clock.

    PC/international shift windows:
    - Shift 1: 06:00–13:00 (7 hours)
    - Shift 2: 13:00–20:00 (7 hours)
    - Shift 3: 20:00–06:00 (10 hours, wraps midnight)
    """
    minutes = now.hour * 60 + now.minute

    if current_shift_num == 1:
        start = 6 * 60
        shift_hours = 7
    elif current_shift_num == 2:
        start = 13 * 60
        shift_hours = 7
    else:
        start = 20 * 60
        shift_hours = 10
        if minutes < 6 * 60:
            minutes += 24 * 60

    elapsed = max(0, minutes - start)
    hour_num = int(elapsed // 60) + 1
    if hour_num < 1:
        hour_num = 1
    if hour_num > shift_hours:
        hour_num = shift_hours
    return hour_num


async def send_current_hour_plan(bot, current_shift_num: int, now: datetime,
                                 force_if_late: bool = False) -> bool:
    """
    Send hourly plan for the current hour if not already sent.
    force_if_late=True: send even if outside the normal :02-:30 window
                        (used when line turns ON mid-hour after being OFF).
    Never sends at :55+ (summary window).
    """
    current_hour = get_current_hour_number(current_shift_num, now)
    today_iso = now.date().isoformat()

    # Never send during summary window — too late for a plan
    if now.minute >= 55:
        logger.info(
            f"Hourly plan Shift {current_shift_num} Hour {current_hour} "
            f"skipped — in summary window (min={now.minute})"
        )
        return False

    # Normal window check — skip only if NOT forcing late send
    if not force_if_late and not is_in_hourly_plan_window(current_shift_num, current_hour, now):
        logger.info(
            f"Hourly plan Shift {current_shift_num} Hour {current_hour} "
            f"outside window (min={now.minute}), skipping"
        )
        return False

    sched_key = f"hourly_plan_scheduled_{today_iso}_{current_shift_num}_{current_hour}"
    catch_key = f"hourly_plan_{today_iso}_{current_shift_num}_{current_hour}"

    # Only skip if catch_key set — confirmed actual delivery
    if bot_state_get(catch_key):
        logger.info(
            f"Hourly plan Shift {current_shift_num} Hour {current_hour} already sent, skipping"
        )
        return False

    late_note = " _(late — line resumed)_" if force_if_late and now.minute > 30 else ""
    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"⏰ *Hourly Plan Reminder – Shift {current_shift_num}, Hour {current_hour}*"
            + f"{late_note}\n\n"
            + "Please share the plan for this hour:\n"
            + "- Production target\n"
            + "- Any scheduled maintenance or adjustments\n"
            + "- Expected challenges"
    )
    try:
        await bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
        bot_state_set(catch_key, "1")
        bot_state_set(sched_key, "1")
        logger.info(
            f"Hourly plan sent (catchup{'/late' if force_if_late else ''}): "
            f"Shift {current_shift_num} Hour {current_hour}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send hourly plan (catchup): {e}")
        return False


async def handle_partial_hours_on_line_resume(bot, current_shift_num: int, line_off_time: datetime,
                                              line_on_time: datetime):
    """Handle partial hours when line comes back on after being off."""
    if not line_off_time:
        return

    off_time = line_off_time.time()
    on_time = line_on_time.time()
    current_hour = get_current_hour_number(current_shift_num, line_on_time)

    # Calculate production time (time when line was ON) in the hour when line went off
    # We need to check if there was MORE than 20 minutes of production in that hour
    off_minutes = off_time.hour * 60 + off_time.minute
    on_minutes = on_time.hour * 60 + on_time.minute

    # Get hour start time for the hour when line went off
    hour_start_minutes = (off_time.hour * 60)  # Start of the hour (e.g., 3:00 = 180)
    hour_end_minutes = hour_start_minutes + 60  # End of the hour (e.g., 4:00 = 240)

    # Calculate production time in the hour when line went off
    if off_time.hour == on_time.hour:
        # Same hour: production = time before off + time after on (remaining in hour)
        production_before_off = off_minutes - hour_start_minutes
        production_after_on = hour_end_minutes - on_minutes
        total_production_minutes = production_before_off + production_after_on
    else:
        # Different hour: only count production before line went off in that hour
        # (Line came back in a different hour, so that hour is done)
        total_production_minutes = off_minutes - hour_start_minutes

    # Only send if there was >20 min production AND we're in hourly summary window (:55-:59 or :50-:59 last hour)
    if total_production_minutes > 20 and is_in_hourly_summary_window(line_on_time, current_shift_num, current_hour):
        # Check if hourly summary was already sent for this hour
        today_iso = line_on_time.date().isoformat()
        key = f"hourly_summary_{today_iso}_{current_shift_num}_{current_hour}"
        if not bot_state_get(key):
            header = f"📅 {format_date_time_12h(line_on_time)}\n\n"
            hourly_summary_text = (
                    header
                    + f"📝 *Hourly Summary Reminder – Shift {current_shift_num}, Hour {current_hour}*\n\n"
                    + f"⚠️ Partial hour production ({total_production_minutes} min active production)\n\n"
                    + "Please provide hourly production data:\n"
                    + "- Actual output for this period\n"
                    + "- Downtime events (if any)\n"
                    + "- Rejects (if any)\n"
                    + "- Operator notes\n\n"
                    + "💡 AI will generate an hourly summary after you submit the data."
            )
            try:
                await bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=hourly_summary_text,
                    parse_mode="Markdown"
                )
                bot_state_set(key, "1")
                logger.info(
                    f"Partial hour summary reminder sent for Shift {current_shift_num}, Hour {current_hour} ({total_production_minutes} min production)")
            except Exception as e:
                logger.error(f"Failed to send partial hour summary: {e}")
        else:
            logger.info(
                f"Hourly summary already sent for Shift {current_shift_num} Hour {current_hour} today, skipping")

    # If in shift summary window (:55-:59 at shift end), also send shift summary reminder
    if is_in_shift_summary_window(current_shift_num, line_on_time):
        # Check if shift summary was already sent for this shift today
        today_iso = line_on_time.date().isoformat()
        key = f"shift_report_{today_iso}_{current_shift_num}"
        if not bot_state_get(key):
            header = f"📅 {format_date_time_12h(line_on_time)}\n\n"
            shift_summary_text = (
                    header
                    + f"📊 *Shift {current_shift_num} Summary Report Reminder*\n\n"
                    + "- Actual output\n"
                    + "- Downtime (reason + minutes)\n"
                    + "- Rejects (preform, bottle, cap, label)\n"
                    + "- Operator remarks"
            )
            try:
                await bot.send_message(
                    chat_id=GROUP_CHAT_ID,
                    text=shift_summary_text,
                    parse_mode="Markdown"
                )
                bot_state_set(key, "1")
                logger.info(f"Shift {current_shift_num} summary reminder sent (near shift end after line resume)")
            except Exception as e:
                logger.error(f"Failed to send shift summary reminder: {e}")


def is_near_shift_end(current_shift_num: int, now: datetime) -> bool:
    """True if within last 20 minutes of shift end. Used for partial-hour logic only."""
    t = now.time()
    if current_shift_num == 1:  # PC ends 13:00
        return time(12, 40) <= t < time(13, 0)
    if current_shift_num == 2:  # PC ends 20:00
        return time(19, 40) <= t < time(20, 0)
    # Shift 3 ends at 06:00
    return time(5, 40) <= t < time(6, 0)


# ---------------- STRICT TIMING WINDOWS ----------------
# All reminder execution must stay within these windows. Never execute outside.

HOURLY_PLAN_WINDOW_END_MINUTE = 30  # Never after :30
HOURLY_SUMMARY_WINDOW_START = 55
HOURLY_SUMMARY_LAST_HOUR_START = 50  # Last hour of shift: :50, normal hours: :55
HOURLY_SUMMARY_WINDOW_END = 59
SHIFT_SUMMARY_WINDOW_START = 55
SHIFT_SUMMARY_WINDOW_END = 59


def is_in_hourly_plan_window(shift: int, hour: int, now: datetime) -> bool:
    """Hourly Plan: :02-:30 (normal) or :05-:30 (first hour of shift). Never after :30."""
    m = now.minute
    if m > HOURLY_PLAN_WINDOW_END_MINUTE:
        return False
    if hour == 1:
        return 5 <= m <= HOURLY_PLAN_WINDOW_END_MINUTE
    return 2 <= m <= HOURLY_PLAN_WINDOW_END_MINUTE


def is_in_hourly_summary_window(now: datetime, shift: int | None = None, hour: int | None = None) -> bool:
    """Hourly Summary: :55-:59 (normal hours), :50-:59 (last hour of shift). Never after :59."""
    m = now.minute
    if m > HOURLY_SUMMARY_WINDOW_END:
        return False
    shift_hours = {1: 7, 2: 7, 3: 10}
    is_last = shift is not None and hour is not None and (hour == shift_hours.get(shift, 0))
    start = HOURLY_SUMMARY_LAST_HOUR_START if is_last else HOURLY_SUMMARY_WINDOW_START
    return start <= m <= HOURLY_SUMMARY_WINDOW_END


def is_in_shift_summary_window(shift: int, now: datetime) -> bool:
    """Shift Summary: strict :55-:59 at end of shift only. Never outside that window."""
    t = now.time()
    m = now.minute
    if m < SHIFT_SUMMARY_WINDOW_START or m > SHIFT_SUMMARY_WINDOW_END:
        return False
    if shift == 1:
        return time(12, 55) <= t <= time(12, 59)
    if shift == 2:
        return time(19, 55) <= t <= time(19, 59)
    return time(5, 55) <= t <= time(5, 59)


def is_in_daily_plan_recovery_window(now: datetime) -> bool:
    """Daily Plan recovery: first 45 min of shift 1 only (once per calendar day)."""
    shift = get_shift_for_time(now)
    if shift != 1:
        return False
    minutes = now.hour * 60 + now.minute
    start = 6 * 60
    return 0 <= (minutes - start) <= 45


def is_in_shift_plan_recovery_window(shift: int, now: datetime) -> bool:
    """Shift Plan recovery: first 45 min of the shift only."""
    actual_shift = get_shift_for_time(now)
    if actual_shift != shift:
        return False
    shift_start = {1: 6 * 60, 2: 13 * 60, 3: 20 * 60}
    minutes = now.hour * 60 + now.minute
    if shift == 3 and now.hour < 6:
        minutes += 24 * 60
    start = shift_start[shift]
    if shift == 3 and now.hour < 6:
        start = 20 * 60
    return 0 <= (minutes - start) <= 45


async def catch_up_missed_reminders(app, current_shift_num: int, now: datetime):
    """
    On bot startup: send Daily Plan and Shift Plan if not posted; then any missed hourly reminders.
    Respects line state — planning reminders suppressed if line is OFF/sanitation.
    """
    today_iso = now.date().isoformat()
    line_is_active = (line_state == LINE_STATE_RUNNING)
    shift_has_production = (
            shift_had_production.get(current_shift_num, False)
            or _shift_had_any_production(current_shift_num, today_iso)
    )

    logger.info(
        f"[STARTUP-CATCHUP] Shift {current_shift_num} | "
        f"line_state={line_state} | shift_has_production={shift_has_production}"
    )

    # CASE 1: Line OFF entire shift, no production — suppress everything
    if not line_is_active and not shift_has_production:
        logger.info("[STARTUP-CATCHUP] CASE 1: no production, suppressing all catchup reminders")
        return

    # 1. Daily plan — only if line active
    if line_is_active:
        await send_daily_plan_if_needed(app.bot, now, skip_window_check=True)
        await asyncio.sleep(1)
    else:
        logger.info("[STARTUP-CATCHUP] Daily plan skipped — line OFF (CASE 2)")

    # 2. Shift plan — only if line active
    if line_is_active:
        await send_shift_plan_if_needed(app.bot, current_shift_num, now, skip_window_check=True)
        await asyncio.sleep(1)
    else:
        logger.info(f"[STARTUP-CATCHUP] Shift {current_shift_num} plan skipped — line OFF (CASE 2)")

    # 3. Hourly plan — only if line active
    if line_is_active:
        await send_current_hour_plan(app.bot, current_shift_num, now)
        await asyncio.sleep(1)
    else:
        logger.info(f"[STARTUP-CATCHUP] Hourly plan skipped — line OFF (CASE 2)")

    # 4. Hourly summary — send if production occurred (CASE 1 already returned)
    current_hour_num = get_current_hour_number(current_shift_num, now)
    if is_in_hourly_summary_window(now, current_shift_num, current_hour_num):
        if shift_has_production:
            sched_key = f"hourly_summary_scheduled_{today_iso}_{current_shift_num}_{current_hour_num}"
            catch_key = f"hourly_summary_{today_iso}_{current_shift_num}_{current_hour_num}"
            if not bot_state_get(sched_key) and not bot_state_get(catch_key):
                header = f"📅 {format_date_time_12h(now)}\n\n"
                text = (
                        header
                        + f"📝 *Hourly Summary Reminder – Shift {current_shift_num}, Hour {current_hour_num}*\n\n"
                        + "Please provide hourly production data:\n"
                        + "- Actual output for this hour\n"
                        + "- Downtime events (if any)\n"
                        + "- Rejects (if any)\n"
                        + "- Operator notes\n\n"
                        + "💡 AI will generate an hourly summary after you submit the data."
                )
                try:
                    await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
                    bot_state_set(sched_key, "1")
                    bot_state_set(catch_key, "1")
                    logger.info(
                        f"[STARTUP-CATCHUP] Hourly summary sent: "
                        f"Shift {current_shift_num} Hr {current_hour_num}"
                    )
                except Exception as e:
                    logger.error(f"[STARTUP-CATCHUP] Hourly summary failed: {e}")
                await asyncio.sleep(1)
        else:
            logger.info(
                f"[STARTUP-CATCHUP] Hourly summary skipped — no production "
                f"Shift {current_shift_num} Hr {current_hour_num}"
            )

    # 5. Shift summary — send if production occurred (CASE 1 already returned)
    if is_in_shift_summary_window(current_shift_num, now):
        if shift_has_production:
            fired_key = f"shift_report_fired_{today_iso}_{current_shift_num}"
            recovery_key = f"shift_report_recovery_{today_iso}_{current_shift_num}"
            if not bot_state_get(fired_key) and not bot_state_get(recovery_key):
                header = f"📅 {format_date_time_12h(now)}\n\n"
                text = (
                        header
                        + f"📊 *Shift {current_shift_num} Summary Report Reminder*\n\n"
                        + "- Actual output\n"
                        + "- Downtime (reason + minutes)\n"
                        + "- Rejects (preform, bottle, cap, label)\n"
                        + "- Operator remarks"
                )
                try:
                    await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode="Markdown")
                    bot_state_set(fired_key, "1")
                    bot_state_set(recovery_key, "1")
                    logger.info(
                        f"[STARTUP-CATCHUP] Shift {current_shift_num} summary sent"
                    )
                except Exception as e:
                    logger.error(f"[STARTUP-CATCHUP] Shift summary failed: {e}")
        else:
            logger.info(
                f"[STARTUP-CATCHUP] Shift summary skipped — no production "
                f"Shift {current_shift_num}"
            )
            bot_state_set(f"shift_report_fired_{today_iso}_{current_shift_num}", "1")


async def post_init(app):
    global current_shift, daily_plan_last_date

    load_bot_state_from_db()
    await setup_bot_commands(app)
    await setup_shift_schedules(app)

    now = now_ethiopia()
    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock
    logger.info(f"Bot started: Synced current_shift to {current_shift} (clock time: {now.strftime('%H:%M:%S')})")

    # On startup: only catchup daily plan + shift plan + hourly plan if missed
    # Do NOT call recover_missed_reminders_on_reconnect here — that causes
    # shift plan to fire as "missed" before the scheduler gets a chance at :02
    await catch_up_missed_reminders(app, current_shift, now)

    # Start background connection watchdog — recovery only fires after real internet drop
    asyncio.create_task(connection_watchdog(app))
    logger.info("[WATCHDOG] Connection watchdog task created")

    startup_msg = (
        f"🤖 Bot Started Successfully\n\n"
        f"⏰ Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📅 Current Shift: {current_shift}\n"
        f"🏭 Line State: {line_state}\n"
        f"✅ Reminders: ACTIVE\n"
        f"🔌 Connection Watchdog: ACTIVE\n\n"
        f"All scheduled reminders are configured.\nUse /bot_status to check current state."
    )
    try:
        await app.bot.send_message(chat_id=GROUP_CHAT_ID, text=startup_msg)
        logger.info("Startup message sent to group")
    except Exception as e:
        logger.error(f"Failed to send startup message: {e}")


# ---------------- LINE / SANITATION CONTROL COMMANDS ----------------
async def line_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since, line_off_next_reminder_allowed, line_off_one_reminder_fired
    now = now_ethiopia()
    line_state = LINE_STATE_OFF
    line_off_since = now
    # Allow exactly ONE next scheduled reminder after this OFF event, then suppress.
    line_off_next_reminder_allowed = True
    line_off_one_reminder_fired = False
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", now.isoformat())

    current_shift_num = get_shift_for_time(now)
    shift_had_production[current_shift_num] = True  # production existed before this OFF

    await update.message.reply_text(
        "⚠️ Line set to OFF.\n"
        "✅ The NEXT scheduled reminder will still fire.\n"
        "After that, all hourly reminders will be suppressed until line is ON.\n"
        "📊 Shift summary will still be sent at shift end (production occurred)."
    )


async def line_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since, current_shift
    global line_off_next_reminder_allowed, line_off_one_reminder_fired
    now = now_ethiopia()
    line_state = LINE_STATE_RUNNING
    line_off_since = None
    line_off_next_reminder_allowed = True
    line_off_one_reminder_fired = False
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", "")

    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock
    shift_had_production[current_shift] = True

    await update.message.reply_text(
        "✅ Line set to ON.\n"
        "Processing reminders and checking for missed items..."
    )

    today_iso = now.date().isoformat()

    # 1. Daily plan
    await send_daily_plan_if_needed(context.bot, now, skip_window_check=True)
    await asyncio.sleep(1)

    # 2. Shift plan
    await send_shift_plan_if_needed(context.bot, current_shift, now, skip_window_check=True)
    await asyncio.sleep(1)

    # 3. Hourly plan — force send even if past :30 window
    #    (operator needs it regardless of what minute line turned ON)
    #    Only skipped if we're in summary window (:55+) — too late for a plan
    await send_current_hour_plan(context.bot, current_shift, now, force_if_late=True)
    await asyncio.sleep(1)

    # 4. Flush any AI-muted queued reminders
    await flush_pending_reminders(context.bot, reason="line")


async def sanitation_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since, line_off_next_reminder_allowed, line_off_one_reminder_fired
    now = now_ethiopia()
    line_state = LINE_STATE_SANITATION
    line_off_since = now
    # Allow exactly ONE next scheduled reminder after sanitation starts, then suppress.
    line_off_next_reminder_allowed = True
    line_off_one_reminder_fired = False
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", now.isoformat())

    current_shift_num = get_shift_for_time(now)
    shift_had_production[current_shift_num] = True  # production existed before sanitation

    await update.message.reply_text(
        "🧼 Sanitation started.\n"
        "✅ The NEXT scheduled reminder will still fire.\n"
        "After that, all hourly reminders will be suppressed until sanitation ends.\n"
        "📊 Shift summary will still be sent at shift end (production occurred)."
    )


async def sanitation_end_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since, current_shift
    global line_off_next_reminder_allowed, line_off_one_reminder_fired
    now = now_ethiopia()
    line_state = LINE_STATE_RUNNING
    line_off_since = None
    line_off_next_reminder_allowed = True
    line_off_one_reminder_fired = False
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", "")

    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock
    shift_had_production[current_shift] = True

    await update.message.reply_text(
        "✅ Sanitation finished.\n"
        "Processing reminders and checking for missed items..."
    )

    # 1. Daily plan
    await send_daily_plan_if_needed(context.bot, now, skip_window_check=True)
    await asyncio.sleep(1)

    # 2. Shift plan
    await send_shift_plan_if_needed(context.bot, current_shift, now, skip_window_check=True)
    await asyncio.sleep(1)

    # 3. Hourly plan — force send even if past :30 window
    await send_current_hour_plan(context.bot, current_shift, now, force_if_late=True)
    await asyncio.sleep(1)

    # 4. Flush AI-muted queued reminders
    await flush_pending_reminders(context.bot, reason="line")


def load_shift_evidence_from_db(target_date=None) -> dict:
    """
    Load shift data from DB and reconstruct ai_shift_evidence-compatible text blobs.
    Reconstructs MECHANICAL / ELECTRICAL / UTILITY headers so parse_downtime_categorized works.
    If target_date is None, auto-detects the most recent date with >= 2 shifts.
    Returns {1: [...], 2: [...], 3: [...], "_resolved_date": date_obj}
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if target_date is None:
            cur.execute("""
                SELECT date FROM production
                GROUP BY date
                HAVING COUNT(DISTINCT shift) >= 2
                ORDER BY date DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                logger.info("load_shift_evidence_from_db: no date with >= 2 shifts found in DB")
                cur.close()
                return {}
            target_date = row[0]
            logger.info(f"load_shift_evidence_from_db: auto-detected date = {target_date}")

        from datetime import date as date_type
        if isinstance(target_date, str):
            for fmt in ("%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"):
                try:
                    target_date = datetime.strptime(target_date, fmt).date()
                    break
                except ValueError:
                    continue
        if not isinstance(target_date, date_type):
            logger.error(f"load_shift_evidence_from_db: cannot parse target_date={target_date}")
            cur.close()
            return {}

        logger.info(f"load_shift_evidence_from_db: querying for date={target_date}")

        cur.execute("""
            SELECT p.id, p.shift, p.product_type, p.shift_plan_pack, p.actual_output_pack,
                   p.date, p.vos_info, p.available_time
            FROM production p
            WHERE p.date = %s
            ORDER BY p.shift
        """, (target_date,))
        rows = cur.fetchall()

        logger.info(f"load_shift_evidence_from_db: found {len(rows)} shift row(s) for {target_date}")

        if not rows:
            cur.close()
            return {}

        result = {}

        for row in rows:
            prod_id, shift, product_type, plan, actual, date_val, vos_info, available_time = row

            # ── Fetch downtime WITH category ──────────────────────────────
            cur.execute("""
                SELECT description, duration_min, category
                FROM downtime_events
                WHERE production_id = %s
            """, (prod_id,))
            downtime_rows = cur.fetchall()

            cur.execute("""
                SELECT preform, bottle, cap, label, shrink FROM rejects WHERE production_id = %s
            """, (prod_id,))
            rej_row = cur.fetchone()

            date_str = date_val.strftime("%d/%m/%y") if hasattr(date_val, "strftime") else str(date_val)
            shift_label = {1: "1st", 2: "2nd", 3: "3rd"}.get(shift, "1st")

            # ── Header / production fields ────────────────────────────────
            lines = [
                f"Date {date_str}",
                f"Shift {shift_label}",
                f"Product type {product_type or 'N/A'}",
                f"Shift plan = {plan}",
                f"Actual output = {actual}",
            ]

            if available_time is not None:
                lines.append(f"Available time = {available_time}")

            if vos_info:
                lines.append(f"VOS = {vos_info}")

            # ── Reconstruct downtime WITH category headers ────────────────
            # Group downtime events by category
            cat_events = {"MECHANICAL": [], "ELECTRICAL": [], "UTILITY": []}
            for desc, dur, cat in downtime_rows:
                # Normalize category — fallback to MECHANICAL if NULL or unknown
                cat_upper = (cat or "MECHANICAL").upper().strip()
                if cat_upper not in cat_events:
                    cat_upper = "MECHANICAL"
                cat_events[cat_upper].append((desc, dur))

            # Write each category header + its events (or "None")
            for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY"):
                lines.append(cat)
                events = cat_events[cat]
                if events:
                    for desc, dur in events:
                        lines.append(f"• {desc} {dur} min")
                else:
                    lines.append("• None")

            # ── Rejects ───────────────────────────────────────────────────
            if rej_row:
                preform, bottle, cap, label, shrink = rej_row
                lines.append(f"Preform = {preform or 0}")
                lines.append(f"Bottle = {bottle or 0}")
                lines.append(f"Cap = {cap or 0}")
                lines.append(f"Label = {label or 0}")
                lines.append(f"Shrink = {shrink or 0}")

            result[int(shift)] = ["\n".join(lines)]
            logger.info(
                f"load_shift_evidence_from_db: loaded shift {shift} "
                f"(plan={plan}, actual={actual}, downtime={len(downtime_rows)} events, "
                f"vos={vos_info}, available_time={available_time})"
            )

        result["_resolved_date"] = target_date
        cur.close()
        return result

    except Exception as e:
        logger.error(f"load_shift_evidence_from_db failed: {e}", exc_info=True)
        return {}
    finally:
        if conn:
            conn.close()


def load_shift_evidence_from_hourly_db(shift: int, target_date=None) -> str | None:
    """
    Load ALL hourly records for a given shift+date from hourly_production,
    aggregate them into a single shift-level text blob that parse_report(),
    parse_downtime_categorized(), parse_rejects() can all understand.

    Returns a reconstructed report string or None if no hourly data found.
    """
    _ensure_hourly_production_table()
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if target_date is None:
            target_date = now_ethiopia().date()

        from datetime import date as date_type
        if isinstance(target_date, str):
            for fmt in ("%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"):
                try:
                    target_date = datetime.strptime(target_date, fmt).date()
                    break
                except ValueError:
                    continue
        if not isinstance(target_date, date_type):
            return None

        # Fetch all hourly rows for this shift+date
        cur.execute("""
            SELECT id, hour_number, product_type, hour_plan, actual_output,
                   available_time_minutes, vos_info
            FROM hourly_production
            WHERE date = %s AND shift = %s
            ORDER BY hour_number
        """, (target_date, shift))
        hourly_rows = cur.fetchall()

        if not hourly_rows:
            cur.close()
            return None

        # Aggregate
        total_plan = 0
        total_actual = 0
        total_available = 0
        product_types = set()
        all_vos = []
        # Downtime by category
        agg_downtime = {"MECHANICAL": [], "ELECTRICAL": [], "UTILITY": []}
        # Rejects
        total_rejects = {"preform": 0, "bottle": 0, "cap": 0, "label": 0, "shrink": 0.0}
        hours_included = []

        for row in hourly_rows:
            h_id, hour_num, prod_type, h_plan, h_actual, h_avail, h_vos = row
            total_plan += h_plan or 0
            total_actual += h_actual or 0
            total_available += h_avail or 60
            if prod_type:
                product_types.add(prod_type.strip())
            if h_vos:
                all_vos.append(f"Hr{hour_num}: {h_vos}")
            hours_included.append(hour_num)

            # Fetch downtime for this hour
            cur.execute("""
                SELECT description, duration_min, category
                FROM hourly_downtime_events
                WHERE hourly_production_id = %s
            """, (h_id,))
            for desc, dur, cat in cur.fetchall():
                cat_upper = (cat or "MECHANICAL").upper().strip()
                if cat_upper not in agg_downtime:
                    cat_upper = "MECHANICAL"
                agg_downtime[cat_upper].append((desc, dur))

            # Fetch rejects for this hour
            cur.execute("""
                SELECT preform, bottle, cap, label, shrink
                FROM hourly_rejects
                WHERE hourly_production_id = %s
            """, (h_id,))
            rej = cur.fetchone()
            if rej:
                total_rejects["preform"] += rej[0] or 0
                total_rejects["bottle"] += rej[1] or 0
                total_rejects["cap"] += rej[2] or 0
                total_rejects["label"] += rej[3] or 0
                total_rejects["shrink"] += rej[4] or 0.0

        cur.close()

        # Build text blob
        date_str = target_date.strftime("%d/%m/%y")
        shift_label = {1: "1st", 2: "2nd", 3: "3rd"}.get(shift, "1st")
        product_str = ", ".join(product_types) if product_types else "N/A"

        lines = [
            f"Date {date_str}",
            f"Shift {shift_label}",
            f"Product type {product_str}",
            f"Shift plan = {total_plan}",
            f"Actual output = {total_actual}",
            f"Available time = {total_available}",
        ]

        if all_vos:
            lines.append(f"VOS = {'; '.join(all_vos)}")

        # Downtime with category headers
        for cat in ("MECHANICAL", "ELECTRICAL", "UTILITY"):
            lines.append(cat)
            events = agg_downtime[cat]
            if events:
                for desc, dur in events:
                    lines.append(f"• {desc} {dur} min")
            else:
                lines.append("• None")

        # Rejects
        lines.append(f"Preform = {total_rejects['preform']}")
        lines.append(f"Bottle = {total_rejects['bottle']}")
        lines.append(f"Cap = {total_rejects['cap']}")
        lines.append(f"Label = {total_rejects['label']}")
        lines.append(f"Shrink = {round(total_rejects['shrink'], 2)}")

        text_blob = "\n".join(lines)
        logger.info(
            f"load_shift_evidence_from_hourly_db: shift={shift} date={target_date} "
            f"hours={hours_included} plan={total_plan} actual={total_actual}"
        )
        return text_blob

    except Exception as e:
        logger.error(f"load_shift_evidence_from_hourly_db failed: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()


def load_all_shifts_from_hourly_db(target_date=None) -> dict:
    """
    Load hourly data for ALL shifts on a date, aggregate each shift,
    return {1: [text], 2: [text], 3: [text], "_resolved_date": date}
    compatible with ai_shift_evidence format.
    """
    _ensure_hourly_production_table()

    if target_date is None:
        target_date = now_ethiopia().date()

    from datetime import date as date_type
    if isinstance(target_date, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"):
            try:
                target_date = datetime.strptime(target_date, fmt).date()
                break
            except ValueError:
                continue
    if not isinstance(target_date, date_type):
        return {}

    result = {}
    for shift in (1, 2, 3):
        text = load_shift_evidence_from_hourly_db(shift, target_date)
        if text:
            result[shift] = [text]

    if result:
        result["_resolved_date"] = target_date

    return result


async def all_shift_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generate an AI summary that covers all closed shifts so far.
    - /all_shift_summary           → most recent date with >= 2 shifts in DB
    - /all_shift_summary 24/02/26  → specific date from DB
    Falls back to in-memory if DB has no data.
    """
    specific_date_requested = bool(context.args)
    target_date = None

    # ── Parse explicit date if given ─────────────────────────────────────────
    if context.args:
        raw = context.args[0].strip()
        parsed = None
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                continue
        if parsed is None:
            await update.message.reply_text(
                "❌ Invalid date format.\n"
                "Use DD/MM/YY — e.g. /all_shift_summary 24/02/26\n"
                "Or DD/MM/YYYY — e.g. /all_shift_summary 24/02/2026"
            )
            return
        target_date = parsed

    # ── Load from DB ─────────────────────────────────────────────────────────
    db_evidence = load_shift_evidence_from_db(target_date)

    # Extract resolved date without mutating the dict
    resolved_date = db_evidence.get("_resolved_date", target_date)

    # Only check integer keys 1, 2, 3 — never the string "_resolved_date" key
    db_shifts = [s for s in (1, 2, 3) if db_evidence.get(s)]

    date_label = resolved_date.strftime("%d/%m/%Y") if resolved_date else "unknown date"

    logger.info(f"all_shift_summary_cmd: resolved_date={resolved_date}, db_shifts={db_shifts}")

    if len(db_shifts) >= 2:
        await update.message.reply_text(
            f"⏳ Generating summary from database for "
            f"{date_label} ({len(db_shifts)} shifts found)..."
        )
        # Temporarily swap DB data into ai_shift_evidence so existing AI function works unchanged
        original_evidence = {k: list(v) for k, v in ai_shift_evidence.items()}
        for shift in (1, 2, 3):
            ai_shift_evidence[shift] = db_evidence.get(shift, [])
        try:
            await generate_multi_shift_summary_and_post(context, db_shifts)
        finally:
            # Always restore original memory even if AI call fails
            for shift in (1, 2, 3):
                ai_shift_evidence[shift] = original_evidence[shift]

    elif specific_date_requested:
        await update.message.reply_text(
            f"⚠️ Only {len(db_shifts)} shift(s) found in the database for {date_label}.\n"
            "At least 2 shifts are required.\n\n"
            "Make sure shift reports were submitted for that date."
        )

    else:
        # No date given, DB empty → fall back to in-memory (original behaviour)
        included_shifts = [s for s in (1, 2, 3) if ai_shift_evidence.get(s)]
        if len(included_shifts) < 2:
            await update.message.reply_text(
                "At least two shift summaries are required.\n"
                "Use /shift_summary for each shift first."
            )
            return
        await update.message.reply_text(
            f"⏳ Generating summary from memory ({len(included_shifts)} shifts)..."
        )
        await generate_multi_shift_summary_and_post(context, included_shifts)


def _parse_hour_arg(args: list) -> tuple[int | None, str]:
    """
    If first arg is a number 0-23, treat as clock hour (e.g. 3 = 3:00-4:00).
    Returns (hour_or_none, report_text).
    """
    if not args:
        return None, ""
    try:
        first = int(args[0])
        if 0 <= first <= 23:
            return first, " ".join(args[1:]).strip()
    except ValueError:
        pass
    return None, " ".join(args).strip()


def _hour_had_production_or_partial(shift: int, hour: int, date_iso: str) -> bool:
    """
    Check if there was production for a specific hour, even if line went off during that hour.
    Returns True if there was any production for that hour.
    Example: line works 3:00-3:35, then OFF - should return True for hour 3.
    """
    conn = None
    cur = None
    try:
        conn = get_clean_db_connection()
        cur = conn.cursor()

        # First try to check hourly production table if it exists
        try:
            cur.execute(
                "SELECT COUNT(*) FROM hourly_production WHERE date = %s AND shift = %s AND hour = %s AND actual_output > 0",
                (date_iso, shift, hour)
            )
            count = cur.fetchone()[0]
            if count > 0:
                return True
        except Exception:
            # hourly_production table might not exist, continue with fallback logic
            pass

        # Fallback: Check if there's any production for the entire shift
        # If shift had production, assume this hour might have contributed
        cur.execute(
            "SELECT actual_output_pack FROM production WHERE date = %s AND shift = %s",
            (date_iso, shift)
        )
        result = cur.fetchone()

        # If shift had any production, assume this hour might have contributed
        # This ensures hourly summaries fire even if line went off during the hour
        return result is not None and result[0] > 0

    except Exception as e:
        logger.error(f"Error checking hourly production: {e}")
        # On error, assume there was production to avoid missing summaries
        return True
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _hour_had_production(shift: int, hour: int, date_iso: str) -> bool:
    """
    Check if there was any production for a specific hour.
    Returns True if there was production data for that hour.
    """
    conn = None
    cur = None
    try:
        conn = get_clean_db_connection()
        cur = conn.cursor()

        # First try to check hourly production table if it exists
        try:
            cur.execute(
                "SELECT COUNT(*) FROM hourly_production WHERE date = %s AND shift = %s AND hour = %s AND actual_output > 0",
                (date_iso, shift, hour)
            )
            count = cur.fetchone()[0]
            if count > 0:
                return True
        except Exception:
            # hourly_production table might not exist, continue with fallback logic
            pass

        # Fallback: Check if there's any production for the shift and assume partial hour production
        cur.execute(
            "SELECT actual_output_pack FROM production WHERE date = %s AND shift = %s",
            (date_iso, shift)
        )
        result = cur.fetchone()

        # If shift had any production, assume this hour might have contributed
        return result is not None and result[0] > 0

    except Exception as e:
        logger.error(f"Error checking hourly production: {e}")
        # On error, assume there was production to avoid missing summaries
        return True
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _shift_had_any_production(shift: int, date_iso: str) -> bool:
    """
    Check if a shift had ANY production (even partial).
    Returns True if there was any production for the shift.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check main production table for any actual output
        cur.execute(
            "SELECT actual_output_pack FROM production WHERE date = %s AND shift = %s",
            (date_iso, shift)
        )
        result = cur.fetchone()

        # If main production record exists with >0 output, return True
        if result is not None and result[0] > 0:
            cur.close()
            conn.close()
            return True

        # Also try to check hourly production table for more granular data if it exists
        try:
            cur.execute(
                "SELECT COUNT(*) FROM hourly_production WHERE date = %s AND shift = %s AND actual_output > 0",
                (date_iso, shift)
            )
            hourly_count = cur.fetchone()[0]
            if hourly_count > 0:
                cur.close()
                conn.close()
                return True
        except Exception:
            # hourly_production table might not exist, that's fine
            pass

        cur.close()
        conn.close()

        return False
    except Exception as e:
        logger.error(f"Error checking shift production: {e}")
        # On error, assume there was production to avoid missing summaries
        return True


async def hourly_summary_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Two ways to use:
    1) Two-step: Send /hourly_summary_ai 11 (or just 11 for 11:00–12:00). Bot asks for the report. Send the report in your next message.
    2) One message: /hourly_summary_ai 11 Date 18/02/26 Shift 2nd ... (hour + full report text)
    No need to start AI audit – this command works on its own.
    """
    if not context.args:
        await update.message.reply_text(
            "📝 *Hourly summary* – two options:\n\n"
            "1️⃣ *Two-step (easiest):*\n"
            "  Send: /hourly_summary_ai 11\n"
            "  (11 = hour 11:00–12:00; use 0–23)\n"
            "  Then send your hourly report in the *next message* (same format as shift: Date, Shift, Product type, Plan, Actual, Downtime, Rejects).\n\n"
            "2️⃣ *One message:*\n"
            "  /hourly_summary_ai 11 Date 18/02/26 Shift 2nd Product type Water ...\n\n"
            "Without hour: last hour is used.",
            parse_mode="Markdown"
        )
        return

    hour_slot, report_text = _parse_hour_arg(context.args)

    if hour_slot is None and report_text:
        # Only report text, no hour → use last hour
        now = now_ethiopia()
        hour_slot = (now.hour - 1) % 24

    if not report_text:
        # User sent e.g. /hourly_summary_ai 11 → wait for next message
        if hour_slot is None:
            now = now_ethiopia()
            hour_slot = (now.hour - 1) % 24
        context.user_data["hourly_summary_pending"] = hour_slot
        hour_label = format_hour_range_12h(hour_slot)
        await update.message.reply_text(
            f"✅ Hour set to *{hour_label}*.\n\n"
            "Now send your hourly report in the *next message* (same format as shift report):\n"
            "Date, Shift, Product type, Shift plan, Actual, Downtime, Rejects.",
            parse_mode="Markdown"
        )
        return

    hour_label = format_hour_range_12h(hour_slot)
    global active_validation_session_key
    try:
        try:
            h_prod_check = parse_report(report_text)
            current_shift_num = h_prod_check["shift"]
        except Exception:
            current_shift_num = get_shift_for_time(now_ethiopia())

        # Extract hour explicitly or use parsed hour_slot
        parsed_hour = hour_slot
        hour_label = f"Shift {current_shift_num}, Hour {parsed_hour}"

        try:
            h_prod = parse_report(report_text)
            h_cat_dt = parse_downtime_categorized(report_text)
            h_downtime = flatten_categorized_downtime(h_cat_dt)
            h_rejects = parse_rejects(report_text)
            h_vos = parse_vos(report_text)
            save_hourly_to_database(
                h_prod, h_downtime, h_rejects,
                hour_number=parsed_hour,
                vos_info=h_vos,
                shift_override=current_shift_num,
            )
        except Exception as e:
            logger.warning(f"Hourly DB save skipped in command: {e}")

        validation = await validate_and_question_hourly(
            context, report_text, current_shift_num, parsed_hour
        )
        await asyncio.sleep(1)

        if validation and validation.get("_blocked"):
            session_key = validation["_session_key"]
            active_validation_session_key = session_key
            validation_sessions[session_key]["_pending_hour"] = parsed_hour
            validation_sessions[session_key]["_report_text"] = report_text
            validation_sessions[session_key]["_hour_label"] = hour_label
            return  # ✅ hold — summary comes after verdict
        else:
            ai_summary = await ai_generate_hourly_summary_from_text(report_text)
            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📝 HOURLY AI SUMMARY ({hour_label})\n\n{ai_summary}",
            )
            await update.message.reply_text(f"✅ Hourly AI summary for {hour_label} posted to group.")
    except Exception as e:
        logger.error(f"Error generating hourly summary: {e}")
        await update.message.reply_text(f"❌ Error generating hourly summary: {e}")


async def shift_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generate shift report(s):
    - If called during Shift 1 or 2: Shows Shift 1 and 2 reports
    - If called during/after Shift 3: Shows all shifts (1, 2, 3)
    Shows 'not provided' for shifts without summaries
    """
    now = now_ethiopia()
    current_shift_by_clock = get_shift_for_time(now)

    if current_shift_by_clock == 3:
        # Shift 3 - show all shifts
        shifts_to_show = [1, 2, 3]
        report_title = "📊 SHIFT REPORTS - ALL SHIFTS\n\n"
    else:
        # Shift 1 or 2 - show shifts 1 and 2
        shifts_to_show = [1, 2]
        report_title = "📊 SHIFT REPORTS - SHIFTS 1 AND 2\n\n"

    report_text = report_title

    for shift in shifts_to_show:
        if daily_ai_shift_summaries.get(shift):
            report_text += f"SHIFT {shift} SUMMARY:\n{daily_ai_shift_summaries[shift]}\n\n"
        else:
            report_text += f"SHIFT {shift} SUMMARY:\n⚠️ Shift summary is not provided.\n\n"

    # No parse_mode - AI content contains _*[] that break Markdown
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=report_text,
    )

    if len(shifts_to_show) == 2:
        await update.message.reply_text("✅ Posted shift reports for Shifts 1 and 2.")
    else:
        await update.message.reply_text("✅ Posted shift reports for all shifts (1, 2, 3).")


async def test_reminder_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test command to verify reminders work immediately"""
    test_text = (
        "🧪 *TEST REMINDER*\n\n"
        "This is a test reminder to verify the bot is working.\n"
        "If you see this, reminders are active and functioning correctly!"
    )
    await send_or_queue_reminder(context, test_text, parse_mode="Markdown")
    await update.message.reply_text("✅ Test reminder sent to group!")


def get_shift_reminders(shift: int) -> list[tuple[str, str]]:
    """
    Exact schedule per shift in Ethiopian clock (12h):
    - Shift start:  Shift Plan :02, Hourly Plan :05, Hourly Summary :55
    - Normal hours: Hourly Plan :02, Hourly Summary :55
    - Last hour:    Hourly Plan :02, Hourly Summary :50, Shift Summary :55
    """
    if shift == 1:  # Ethiopian 00:00–07:00
        return [
            ("12:00 AM", "Daily Production Plan Reminder"),
            ("12:02 AM", "Shift 1 Plan Reminder"),
            ("12:05 AM", "Hour 1 Plan Reminder"),
            ("12:55 AM", "Hour 1 Summary Reminder"),
            ("1:02 AM", "Hour 2 Plan Reminder"),
            ("1:55 AM", "Hour 2 Summary Reminder"),
            ("2:02 AM", "Hour 3 Plan Reminder"),
            ("2:55 AM", "Hour 3 Summary Reminder"),
            ("3:02 AM", "Hour 4 Plan Reminder"),
            ("3:55 AM", "Hour 4 Summary Reminder"),
            ("4:02 AM", "Hour 5 Plan Reminder"),
            ("4:55 AM", "Hour 5 Summary Reminder"),
            ("5:02 AM", "Hour 6 Plan Reminder"),
            ("5:55 AM", "Hour 6 Summary Reminder"),
            ("6:02 AM", "Hour 7 Plan Reminder"),
            ("6:50 AM", "Hour 7 Summary Reminder"),
            ("6:55 AM", "Shift 1 Summary Reminder"),
        ]
    elif shift == 2:  # Ethiopian 07:00–14:00
        return [
            ("7:02 AM", "Shift 2 Plan Reminder"),
            ("7:05 AM", "Hour 1 Plan Reminder"),
            ("7:55 AM", "Hour 1 Summary Reminder"),
            ("8:02 AM", "Hour 2 Plan Reminder"),
            ("8:55 AM", "Hour 2 Summary Reminder"),
            ("9:02 AM", "Hour 3 Plan Reminder"),
            ("9:55 AM", "Hour 3 Summary Reminder"),
            ("10:02 AM", "Hour 4 Plan Reminder"),
            ("10:55 AM", "Hour 4 Summary Reminder"),
            ("11:02 AM", "Hour 5 Plan Reminder"),
            ("11:55 AM", "Hour 5 Summary Reminder"),
            ("12:02 PM", "Hour 6 Plan Reminder"),
            ("12:55 PM", "Hour 6 Summary Reminder"),
            ("1:02 PM", "Hour 7 Plan Reminder"),
            ("1:50 PM", "Hour 7 Summary Reminder"),
            ("1:55 PM", "Shift 2 Summary Reminder"),
        ]
    else:  # shift == 3, Ethiopian 14:00–24:00
        return [
            ("2:02 PM", "Shift 3 Plan Reminder"),
            ("2:05 PM", "Hour 1 Plan Reminder"),
            ("2:55 PM", "Hour 1 Summary Reminder"),
            ("3:02 PM", "Hour 2 Plan Reminder"),
            ("3:55 PM", "Hour 2 Summary Reminder"),
            ("4:02 PM", "Hour 3 Plan Reminder"),
            ("4:55 PM", "Hour 3 Summary Reminder"),
            ("5:02 PM", "Hour 4 Plan Reminder"),
            ("5:55 PM", "Hour 4 Summary Reminder"),
            ("6:02 PM", "Hour 5 Plan Reminder"),
            ("6:55 PM", "Hour 5 Summary Reminder"),
            ("7:02 PM", "Hour 6 Plan Reminder"),
            ("7:55 PM", "Hour 6 Summary Reminder"),
            ("8:02 PM", "Hour 7 Plan Reminder"),
            ("8:55 PM", "Hour 7 Summary Reminder"),
            ("9:02 PM", "Hour 8 Plan Reminder"),
            ("9:55 PM", "Hour 8 Summary Reminder"),
            ("10:02 PM", "Hour 9 Plan Reminder"),
            ("10:55 PM", "Hour 9 Summary Reminder"),
            ("11:02 PM", "Hour 10 Plan Reminder"),
            ("11:50 PM", "Hour 10 Summary Reminder"),
            ("11:55 PM", "Shift 3 Summary Reminder"),
        ]



async def bot_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check bot status and reminder state"""
    now_pc = now_ethiopia()
    now_eth = to_ethiopian_clock(now_pc)
    current_shift_by_clock = get_shift_for_time(now_pc)

    # Get all reminders for current shift
    all_reminders = get_shift_reminders(current_shift_by_clock)

    # Filter to show only future reminders (or current if within 5 minutes)
    # Compare reminder times in Ethiopian clock domain (what get_shift_reminders returns)
    current_hour_24 = now_eth.hour
    current_minute = now_eth.minute

    def time_to_minutes(time_str: str) -> int:
        """Convert '1:02 AM' format to minutes since midnight."""
        parts = time_str.split()
        time_part = parts[0]
        am_pm = parts[1]
        hour, minute = map(int, time_part.split(":"))
        if am_pm == "PM" and hour != 12:
            hour += 12
        elif am_pm == "AM" and hour == 12:
            hour = 0
        return hour * 60 + minute

    current_minutes = current_hour_24 * 60 + current_minute
    upcoming_reminders = []
    for time_str, desc in all_reminders:
        reminder_minutes = time_to_minutes(time_str)
        if reminder_minutes >= current_minutes - 5:  # Show if within 5 min past or future
            upcoming_reminders.append((time_str, desc))

    # If no upcoming reminders, show all reminders for the shift
    if not upcoming_reminders:
        upcoming_reminders = all_reminders

    status_text = (
        f"🤖 *Bot Status*\n\n"
        f"⏰ Current Time (Ethiopian clock): {format_date_time_12h(now_eth)}\n"
        f"⏰ PC Time: {format_date_time_12h(now_pc)}\n"
        f"📅 Current Shift (by clock): {current_shift_by_clock}\n"
        f"🔄 Active Shift (bot state): {current_shift}\n"
        f"🏭 Line State: {line_state}\n"
        f"🤖 AI Audit Block: {'Yes' if ai_reminder_block else 'No'}\n"
        f"📋 Queued Reminders: {len(pending_reminders)}\n"
        f"✅ Reminders Active: {'Yes' if line_state == LINE_STATE_RUNNING and not ai_reminder_block else 'No — reminders are QUEUED'}\n\n"
    )

    if upcoming_reminders:
        status_text += f"⏰ *Shift {current_shift_by_clock} Reminders:*\n"
        for time_str, desc in upcoming_reminders:
            status_text += f"  • {time_str} - {desc}\n"
    else:
        status_text += f"⏰ *Shift {current_shift_by_clock} Reminders:* None scheduled\n"

    if pending_reminders:
        status_text += "\n📬 *Pending reminders:*\n"
        for i, item in enumerate(pending_reminders[:5], 1):  # Show first 5
            mute_type = item.get("mute_type", "unknown")
            shift = item.get("shift", "?")
            status_text += f"  {i}. Shift {shift} ({mute_type})\n"
        if len(pending_reminders) > 5:
            status_text += f"  ... and {len(pending_reminders) - 5} more\n"

    try:
        await update.message.reply_text(status_text, parse_mode="Markdown")
    except Exception:
        # Markdown failed — strip all formatting and send plain
        plain = status_text.replace("*", "").replace("_", "").replace("`", "")
        await update.message.reply_text(plain)


def main():
    # Use Ethiopia timezone so job queue runs at correct local times (aligns with bot_status)
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(Defaults(tzinfo=TZ_ETHIOPIA)).build()

    # Add error handlers to prevent unhandled exceptions
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log Errors caused by Updates."""
        logger.error(f"Exception while handling an update: {context.error}")
        # Don't re-raise the exception to prevent bot from crashing

    app.add_error_handler(error_handler)

    app.add_handler(CommandHandler("start_audit", start_audit))
    app.add_handler(CommandHandler("end_audit", end_audit))
    app.add_handler(CommandHandler("hourly_summary_ai", hourly_summary_ai_cmd))
    app.add_handler(CommandHandler("shift_summary_hourly", shift_summary_from_hourly_cmd))
    app.add_handler(CommandHandler("all_shift_summary_hourly", all_shift_summary_from_hourly_cmd))
    app.add_handler(CommandHandler("shift_summary_hourly_1", shift_summary_hourly_1_cmd))
    app.add_handler(CommandHandler("shift_summary_hourly_2", shift_summary_hourly_2_cmd))
    app.add_handler(CommandHandler("shift_summary_hourly_3", shift_summary_hourly_3_cmd))
    app.add_handler(CommandHandler("shift_summary", shift_summary))
    app.add_handler(CommandHandler("shift_input_1", shift_input_1_cmd))
    app.add_handler(CommandHandler("shift_input_2", shift_input_2_cmd))
    app.add_handler(CommandHandler("shift_input_3", shift_input_3_cmd))
    app.add_handler(CommandHandler("shift_summary_1", shift_summary_1_cmd))
    app.add_handler(CommandHandler("shift_summary_2", shift_summary_2_cmd))
    app.add_handler(CommandHandler("shift_summary_3", shift_summary_3_cmd))
    app.add_handler(CommandHandler("all_shift_summary", all_shift_summary_cmd))
    app.add_handler(CommandHandler("shift_report", shift_report_cmd))
    app.add_handler(CommandHandler("test_reminder", test_reminder_cmd))
    app.add_handler(CommandHandler("bot_status", bot_status_cmd))
    app.add_handler(CommandHandler("line_off", line_off_cmd))
    app.add_handler(CommandHandler("line_on", line_on_cmd))
    app.add_handler(CommandHandler("sanitation_start", sanitation_start_cmd))
    app.add_handler(CommandHandler("sanitation_end", sanitation_end_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # app.post_init = setup_bot_commands
    app.post_init = post_init
    print("Bot running...")
    print(f"Line state: {line_state}, AI block: {ai_reminder_block}")
    print("Reminders are ACTIVE by default. Use /bot_status to check state.")
    app.run_polling()


if __name__ == "__main__":
    main()