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
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
from groq import Groq
import asyncio

# ---------------- CONFIG ----------------
BOT_TOKEN = "8331725306:AAF_g6rT2SVR21wguL-2AYdH2nQHSrDbyhA"
GROUP_CHAT_ID = -5085725916
EFFICIENCY_LIMIT = 75.0

DB_CONFIG = {
    "host": "localhost",
    "database": "aku",
    "user": "admin",
    "password": "admin",
    "port": 5432
}

# ---------------- AI CONFIG ----------------
load_dotenv()

ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
AI_MODEL = "openai/gpt-oss-120b"

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

# Track shift plan reminders sent per shift per day
shift_plan_sent_today = {
    1: None,  # date when sent, or None
    2: None,
    3: None,
}

# Track when line went off (for calculating partial hours)
line_off_since = None  # datetime when line went off


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
    if shift == 1:  # 00:00 - 07:00 = 7 hours
        return 7 * 60
    elif shift == 2:  # 07:00 - 14:00 = 7 hours
        return 7 * 60
    else:  # Shift 3: 14:00 - 00:00 = 10 hours
        return 10 * 60


# ---------------- LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------------- DATABASE ----------------
def get_db_connection():
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


def save_to_database(data, downtime, rejects, shift_override: int | None = None):
    """Save production data. Uses shift_override if provided (for /shift_summary_N)."""
    conn = get_db_connection()
    cur = conn.cursor()
    shift = shift_override if shift_override is not None else data["shift"]
    try:
        # UPSERT: update if (date, shift) already exists
        cur.execute("""
            INSERT INTO production
            (date, shift, product_type, shift_plan_pack, actual_output_pack)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date, shift) DO UPDATE SET
                product_type = EXCLUDED.product_type,
                shift_plan_pack = EXCLUDED.shift_plan_pack,
                actual_output_pack = EXCLUDED.actual_output_pack
            RETURNING id
        """, (data["date"], shift, data["product_type"], data["plan"], data["actual"]))
        production_id = cur.fetchone()[0]

        # Delete old downtime/rejects for this production (for upsert case)
        cur.execute("DELETE FROM downtime_events WHERE production_id = %s", (production_id,))
        cur.execute("DELETE FROM rejects WHERE production_id = %s", (production_id,))

        for d in downtime:
            cur.execute("""
                INSERT INTO downtime_events
                (production_id, description, duration_min)
                VALUES (%s, %s, %s)
            """, (production_id, d["description"], d["duration"]))

        cur.execute("""
            INSERT INTO rejects
            (production_id, preform, bottle, cap, label, shrink)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (production_id, rejects["preform"], rejects["bottle"], rejects["cap"], rejects["label"],
              rejects["shrink"]))

        conn.commit()
        return production_id
    except Exception as e:
        conn.rollback()
        raise e
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

    # Calculate efficiency based on available time
    efficiency = round((actual / plan) * 100, 1) if plan else 0

    return {
        "date": date,
        "shift": shift,
        "product_type": product_type,
        "plan": plan,
        "actual": actual,
        "efficiency": efficiency,
        "available_time": available_time
    }


def parse_downtime(text: str):
    events = []
    t = text.lower()
    pattern = r"(.+?)\s*(\d+)'"
    for desc, mins in re.findall(pattern, t):
        if "shift plan" in desc or "actual output" in desc:
            continue
        clean = desc.replace("vos", "").replace("=", "").strip()
        if len(clean) > 3:
            events.append({"description": clean, "duration": int(mins)})
    return events


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
    Current time using the PC/international clock.

    Important: shift logic is based on Ethiopian CLOCK (international - 6 hours),
    but scheduling runs on the PC clock.
    """
    return datetime.now()


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
) -> None:
    """
    Central place for all scheduled reminders:
    - If muted, store in pending_reminders with timestamp + shift and mute_type
      - If active, send to GROUP_CHAT_ID immediately
    """
    global pending_reminders

    now = now_ethiopia()
    shift_now = get_shift_for_time(now)
    date_now = now.date()

    # If line is OFF or sanitation is running, queue as line-muted
    if line_state != LINE_STATE_RUNNING:
        pending_reminders.append(
            {
                "text": text,
                "parse_mode": parse_mode,
                "created_at": now,
                "shift": shift_now,
                "date": date_now,
                "mute_type": "line",
            }
        )
        logger.info(f"Reminder queued (line muted): Shift {shift_now} at {now.strftime('%H:%M:%S')}")
        return

    # If AI audit is active, queue as AI-muted (but NEVER drop them later)
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
        logger.info(f"Reminder queued (AI muted): Shift {shift_now} at {now.strftime('%H:%M:%S')}")
        return

    # Otherwise send immediately
    logger.info(f"Sending reminder to group: Shift {shift_now} at {now.strftime('%H:%M:%S')}")
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=text,
        parse_mode=parse_mode,
    )


async def flush_pending_reminders(bot, reason: str | None = None) -> None:
    """
    Flush queued reminders.

    - For reason="line": send ONLY same-day, current-shift reminders that
      were muted because the line/sanitation was OFF. Old-shift/day items
      are dropped.
    - For reason="ai": send ALL reminders that were muted because of an
      active AI audit, even if their time has already passed.
    """
    global pending_reminders

    if not pending_reminders:
        return

    now = now_ethiopia()
    current_shift_by_clock = get_shift_for_time(now)
    current_date = now.date()

    to_send = []
    remaining = []

    if reason == "ai":
        # Flush ALL AI-muted reminders, regardless of time/shift.
        for item in pending_reminders:
            if item.get("mute_type") == "ai":
                to_send.append(item)
            else:
                remaining.append(item)
    else:
        # Default / "line": only flush line-muted reminders that still belong
        # to the *current* shift and date. Others are dropped.
        for item in pending_reminders:
            if item.get("mute_type") != "line":
                remaining.append(item)
                continue

            if (
                    item.get("date") == current_date
                    and item.get("shift") == current_shift_by_clock
            ):
                to_send.append(item)
            # else: drop old-shift or old-day line-muted reminders

    pending_reminders = remaining
 # Ensure shift plan/report reminders are sent before hourly reminders
    def _reminder_priority(item):
        text = item.get("text", "")
        if "Plan Reminder" in text and "Hourly" not in text:
            return 0  # Shift plan
        if "Summary Report Reminder" in text:
            return 1  # Shift report
        if "Hourly Plan" in text:
            return 2  # Hourly plan
        if "Hourly Summary" in text:
            return 3  # Hourly summary
        return 0  # Daily plan etc.

    to_send.sort(key=lambda x: (_reminder_priority(x), x.get("created_at", datetime.min)))

 
    for item in to_send:
        await bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=item["text"],
            parse_mode=item.get("parse_mode"),
        )


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
    for text in reversed(ai_shift_evidence[shift]):
        try:
            production_data = parse_report(text)
            downtime = parse_downtime(text)
            rejects = parse_rejects(text)
            break
        except Exception:
            continue
    if production_data:
        try:
            # Use requested shift (not parsed) to avoid wrong date/shift from mixed evidence
            save_to_database(production_data, downtime, rejects, shift_override=shift)
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


async def ai_generate_summary(shift: int):
    evidence = ai_shift_evidence[shift]
    if not evidence:
        return "No evidence found."

    production_data = None
    downtime = []
    rejects = {}

    for text in reversed(evidence):
        try:
            production_data = parse_report(text)
            downtime = parse_downtime(text)
            rejects = parse_rejects(text)
            break
        except:
            continue

    if not production_data:
        return "DATA INCOMPLETE – production report missing."

    # ---------------- CORE CALCULATIONS ----------------
    total_downtime = sum(d["duration"] for d in downtime)
    actual_output = production_data["actual"]
    plan_output = production_data["plan"]

    # Get available time (machine active time) - default to shift duration if not provided
    # If available_time was not parsed (None), use shift duration
    available_time_minutes = production_data.get("available_time")
    if available_time_minutes is None:
        available_time_minutes = get_shift_duration_minutes(shift)

    # Calculate efficiency based on actual vs plan
    efficiency = round((actual_output / plan_output) * 100, 1) if plan_output else 0

    # Calculate downtime ratio based on available time
    downtime_ratio = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes > 0 else 0

    # ---------------- RISK CLASSIFICATION ----------------
    risk_score = 0

    if efficiency < 60:
        risk_score += 3
    elif efficiency < 75:
        risk_score += 2

    if downtime_ratio > 40:
        risk_score += 3
    elif downtime_ratio > 25:
        risk_score += 2

    # Risk assessment based on individual reject types (not summed)
    total_rejects = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
    if actual_output > 0:
        reject_ratio = (total_rejects / actual_output) * 100
        if reject_ratio > 5:
            risk_score += 2
        elif reject_ratio > 2:
            risk_score += 1

    # Mechanical fault detection
    downtime_text = " ".join(d["description"] for d in downtime).lower()

    if "misalignment" in downtime_text or "wear" in downtime_text:
        risk_score += 1
    if "short circuit" in downtime_text or "breaker" in downtime_text:
        risk_score += 1
    if "glue" in downtime_text or "adhesive" in downtime_text:
        risk_score += 1

    if risk_score >= 7:
        risk_level = "CRITICAL"
    elif risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    audit_status = "CLOSED" if shift_closed[shift] else "FOLLOW-UP REQUIRED"

    # ---------------- STRUCTURED DATA FOR AI ----------------
    structured_data = f"""
                            SHIFT: {shift}
                            DATE: {production_data["date"]}
                            PRODUCT: {production_data["product_type"]}
                            PLAN: {production_data["plan"]}
                            ACTUAL: {production_data["actual"]}
                            AVAILABLE_TIME: {available_time_minutes} minutes
                            EFFICIENCY: {efficiency}%
                            TOTAL_DOWNTIME: {total_downtime} minutes
                            DOWNTIME_RATIO: {downtime_ratio}%
                            REJECTS_BREAKDOWN:
                            - Preform: {rejects.get("preform", 0)}
                            - Bottle: {rejects.get("bottle", 0)}
                            - Cap: {rejects.get("cap", 0)}
                            - Label: {rejects.get("label", 0)}
                            SHRINK_LOSS: {rejects.get("shrink", 0)} kg
                            RISK_LEVEL: {risk_level}
                            AUDIT_STATUS: {audit_status}
                            DOWNTIME_BREAKDOWN:
                            {chr(10).join([f"- {d['description']} ({d['duration']} min)" for d in downtime])}
                    """

    # ---------------- AI EXECUTIVE NARRATIVE ----------------
    loop = asyncio.get_running_loop()

    def call_ai():
        return ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
        You are a plant-level executive production analyst writing a professional shift summary report.

       Write a well-structured executive summary that covers:
        - Operational performance against plan
        - Downtime impact and equipment reliability concerns
        - Quality performance based on reject breakdowns
        - Clear conclusions about shift stability

        FORMATTING RULES (strict):
        - Output 3–4 separate paragraphs. Do NOT merge into one long block.
        - Each paragraph: 2–3 sentences only. One idea per paragraph.
        - Insert exactly one blank line between each paragraph.
        - Avoid wall-of-text. Preserve clarity and technical tone.

       WRITING STYLE:
        - Use proper grammar, capitalization, and punctuation.
        - Begin every sentence with a capital letter.
        - Use numeric format for all numbers (e.g., 42%, 64.07%, 240 minutes).
        - Do NOT convert numbers into words.
        - Be analytical, concise, executive-level.
        - Base conclusions strictly on the structured data provided.

        """

                },
                {
                    "role": "user",
                    "content": structured_data
                }
            ],
            temperature=0.2  # ensures numeric formatting and consistency
        )

    response = await loop.run_in_executor(None, call_ai)
    executive_paragraph = response.choices[0].message.content.strip()

    # ---------------- FINAL REPORT (professional layout with icons) ----------------
    downtime_lines = "\n".join([f"  • {d['description']} ({d['duration']} min)" for d in downtime]) if downtime else "  • None"
    final_report = (
            f"✅ STATUS: COMPLETE\n\n"
            f"⚠️ RISK LEVEL: {risk_level}\n\n"
            f"📋 EXECUTIVE SUMMARY\n\n"
            f"{executive_paragraph}\n\n"
            f"────────────────────────────\n\n"
            f"📊 PRODUCTION PERFORMANCE\n\n"
            f"  • Product: {production_data['product_type']}\n"
            f"  • Plan: {production_data['plan']:,} packs\n"
            f"  • Actual: {production_data['actual']:,} packs\n"
            f"  • Available Time: {available_time_minutes} minutes\n"
            f"  • Efficiency: {efficiency}%\n\n"
            f"────────────────────────────\n\n"
            f"⏱️ DOWNTIME ANALYSIS\n\n"
            f"  • Total Downtime: {total_downtime} minutes\n"
            f"  • Downtime Ratio: {downtime_ratio}% of available time\n"
            f"{downtime_lines}\n\n"
            f"────────────────────────────\n\n"
            f"✓ QUALITY METRICS\n\n"
              f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
              f"  • Bottle Rejects: {rejects.get('bottle', 0):,} pcs\n"
              f"  • Cap Rejects: {rejects.get('cap', 0):,} pcs\n"
              f"  • Label Rejects: {rejects.get('label', 0):,} pcs\n"
              f"  • Shrink Loss: {rejects.get('shrink', 0)} kg\n\n"
            f"────────────────────────────\n\n"
            f"📌 AUDIT STATUS: {audit_status}"
    )

    return final_report.strip()


async def ai_generate_hourly_summary_from_text(report_text: str):
    """
    Generate an AI executive-style summary for ONE HOUR of production.
    The input format is the same as for a shift summary:
    - Contains date, shift, shift plan, actual, downtime, rejects, etc.
    """
    try:
        production_data = parse_report(report_text)
    except Exception:
        return "DATA INCOMPLETE – production report missing."

    downtime = parse_downtime(report_text)
    rejects = parse_rejects(report_text)

    total_downtime = sum(d["duration"] for d in downtime)
    actual_output = production_data["actual"]
    plan_output = production_data["plan"]

    # Get available time (machine active time) - default to 60 minutes for hourly if not provided
    # If available_time was not parsed (None), use 60 minutes for hourly summaries
    available_time_minutes = production_data.get("available_time")
    if available_time_minutes is None:
        available_time_minutes = 60

    # Calculate efficiency based on actual vs plan
    efficiency = round((actual_output / plan_output) * 100, 1) if plan_output else 0

    # Calculate downtime ratio based on available time
    downtime_ratio = round((total_downtime / available_time_minutes) * 100, 2) if available_time_minutes > 0 else 0

    risk_score = 0

    if efficiency < 60:
        risk_score += 3
    elif efficiency < 75:
        risk_score += 2

    if downtime_ratio > 40:
        risk_score += 3
    elif downtime_ratio > 25:
        risk_score += 2

    # Risk assessment based on individual reject types (not summed)
    total_rejects = rejects.get("bottle", 0) + rejects.get("cap", 0) + rejects.get("label", 0)
    if actual_output > 0:
        reject_ratio = (total_rejects / actual_output) * 100
        if reject_ratio > 5:
            risk_score += 2
        elif reject_ratio > 2:
            risk_score += 1

    downtime_text = " ".join(d["description"] for d in downtime).lower()

    if "misalignment" in downtime_text or "wear" in downtime_text:
        risk_score += 1
    if "short circuit" in downtime_text or "breaker" in downtime_text:
        risk_score += 1
    if "glue" in downtime_text or "adhesive" in downtime_text:
        risk_score += 1

    if risk_score >= 7:
        risk_level = "CRITICAL"
    elif risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    audit_status = "FOLLOW-UP REQUIRED"

    structured_data = f"""
                        HOUR SHIFT: {production_data["shift"]}
                        DATE: {production_data["date"]}
                        PRODUCT: {production_data["product_type"]}
                        PLAN (hour): {production_data["plan"]}
                        ACTUAL (hour): {production_data["actual"]}
                        AVAILABLE_TIME: {available_time_minutes} minutes
                        EFFICIENCY: {efficiency}%
                        TOTAL_DOWNTIME: {total_downtime} minutes
                        DOWNTIME_RATIO: {downtime_ratio}%
                        REJECTS_BREAKDOWN:
                        - Preform: {rejects.get("preform", 0)}
                        - Bottle: {rejects.get("bottle", 0)}
                        - Cap: {rejects.get("cap", 0)}
                        - Label: {rejects.get("label", 0)}
                        SHRINK_LOSS: {rejects.get("shrink", 0)} kg
                        RISK_LEVEL: {risk_level}
                        AUDIT_STATUS: {audit_status}
                        DOWNTIME_BREAKDOWN:
                        {chr(10).join([f"- {d['description']} ({d['duration']} min)" for d in downtime])}
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

Write a well-structured executive summary that evaluates ONE HOUR of production:
- Operational performance against plan for this hour
- Downtime impact and equipment reliability concerns
- Quality performance based on reject breakdowns
- Clear conclusions about overall hour stability

FORMATTING RULES (strict):
- Output 3–4 separate paragraphs. Do NOT merge into one long block.
- Each paragraph: 2–3 sentences only. One idea per paragraph.
- Insert exactly one blank line between each paragraph.
- Avoid wall-of-text. Preserve clarity and technical tone.

WRITING STYLE:
- Use proper grammar, capitalization, and punctuation.
- Begin every sentence with a capital letter.
- Use numeric format for all numbers (e.g., 42%, 64.07%, 40 minutes).
- Do NOT convert numbers into words.
- Be analytical, concise, executive-level.
- Base conclusions strictly on the structured data provided.
"""
                },
                {
                    "role": "user",
                    "content": structured_data
                },
            ],
            temperature=0.2,
        )

    response = await loop.run_in_executor(None, call_ai)
    executive_paragraph = response.choices[0].message.content.strip()

    downtime_lines = "\n".join([f"  • {d['description']} ({d['duration']} min)" for d in downtime]) if downtime else "  • None"
    final_report = (
            f"✅ STATUS: COMPLETE\n\n"
            f"⚠️ RISK LEVEL: {risk_level}\n\n"
            f"📋 EXECUTIVE SUMMARY\n\n"
            f"{executive_paragraph}\n\n"
            f"────────────────────────────\n\n"
            f"📊 PRODUCTION PERFORMANCE\n\n"
            f"  • Product: {production_data['product_type']}\n"
            f"  • Plan: {production_data['plan']:,} packs\n"
            f"  • Actual: {production_data['actual']:,} packs\n"
            f"  • Available Time: {available_time_minutes} minutes\n"
            f"  • Efficiency: {efficiency}%\n\n"
            f"────────────────────────────\n\n"
            f"⏱️ DOWNTIME ANALYSIS\n\n"
            f"  • Total Downtime: {total_downtime} minutes\n"
            f"  • Downtime Ratio: {downtime_ratio}% of available time\n"
            f"{downtime_lines}\n\n"
            f"────────────────────────────\n\n"
            f"✓ QUALITY METRICS\n\n"
              f"  • Preform Rejects: {rejects.get('preform', 0):,} pcs\n"
              f"  • Bottle Rejects: {rejects.get('bottle', 0):,} pcs\n"
              f"  • Cap Rejects: {rejects.get('cap', 0):,} pcs\n"
              f"  • Label Rejects: {rejects.get('label', 0):,} pcs\n"
              f"  • Shrink Loss: {rejects.get('shrink', 0)} kg\n\n"
            f"────────────────────────────\n\n"
            f"📌 AUDIT STATUS: {audit_status}"
    )

    return final_report.strip()


async def ai_generate_multi_shift_summary(included_shifts: list[int]):
    """
    Use the per-shift AI summaries to create one
    executive overview for the specified shifts.
    """
    if not included_shifts:
        return None

    # Ensure we have summaries for all requested shifts
    for s in included_shifts:
        if not daily_ai_shift_summaries.get(s):
            return None

    blocks = []
    for s in included_shifts:
        blocks.append(f"SHIFT {s} SUMMARY:\n{daily_ai_shift_summaries[s]}")
    day_content = "\n\n".join(blocks)

    included_label = ", ".join(str(s) for s in included_shifts)

    loop = asyncio.get_running_loop()

    def call_ai():
        return ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a plant-level daily production analyst.

You will receive the AI-approved summaries for shifts {included_label} of the same production day.

Produce ONE concise, high-level multi-shift production summary that:
- Compares shifts (performance, downtime, quality)
- Highlights trends across the covered shifts (improving, degrading, stable)
- Flags any systemic risks that persist across shifts
- States the overall stability and performance for these shifts combined.

STRICT RULES:
- Do NOT simply repeat all details; synthesize.
- Use numeric values when referring to performance (e.g. 92%, 350 minutes).
- Be executive-level, suitable for plant and corporate management.
"""
                },
                {
                    "role": "user",
                    "content": day_content
                },
            ],
            temperature=0.2,
        )

    response = await loop.run_in_executor(None, call_ai)
    return response.choices[0].message.content.strip()


async def generate_multi_shift_summary_and_post(
        context: ContextTypes.DEFAULT_TYPE,
        included_shifts: list[int],
) -> None:
    """
    Helper to call the multi-shift AI and post into the group.
    Safe no-op if data is incomplete.
    """
    daily_text = await ai_generate_multi_shift_summary(included_shifts)
    if not daily_text:
        return

    # Build a human-readable label like "Shifts 1 and 2" or "Shifts 1, 2 and 3"
    if len(included_shifts) == 2:
        label = f"Shifts {included_shifts[0]} and {included_shifts[1]}"
    else:
        label = f"Shifts {', '.join(str(s) for s in included_shifts[:-1])} and {included_shifts[-1]}"

    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📘 *MULTI-SHIFT PRODUCTION SUMMARY – {label}*\n\n{daily_text}",
        parse_mode="Markdown",
    )


# ---------------- MESSAGE HANDLER ----------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    # Shift summary two-step (no audit): user sent /shift_input_1 then sends the report next
    pending_shift = context.user_data.get("shift_summary_pending")
    if pending_shift is not None and text and not text.startswith("/"):
        context.user_data.pop("shift_summary_pending", None)
        try:
            # store evidence under requested shift
            ai_shift_evidence[pending_shift].append(text)
            shift_closed[pending_shift] = False

            # Save to DB (upsert)
            try:
                production_data = parse_report(text)
                downtime = parse_downtime(text)
                rejects = parse_rejects(text)
                save_to_database(production_data, downtime, rejects, shift_override=pending_shift)
                logger.info(f"Shift {pending_shift} report saved to database (manual input)")
            except Exception as e:
                logger.warning(f"Manual shift input DB save skipped: {e}")

            ai_text = await ai_generate_summary(pending_shift)
            daily_ai_shift_summaries[pending_shift] = ai_text

            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📊 SHIFT {pending_shift} OFFICIAL SUMMARY\n\n{ai_text}",
            )
            shift_closed[pending_shift] = True
            # await update.message.reply_text(f"✅ Shift {pending_shift} summary posted to group.")
        except Exception as e:
            logger.error(f"Error generating shift summary (manual): {e}")
            await update.message.reply_text(f"❌ Error generating shift summary: {e}")
        return

    # Hourly summary two-step: user previously sent /hourly_summary_ai 11, now sending the report
    pending_hour = context.user_data.get("hourly_summary_pending")
    if pending_hour is not None and text and not text.startswith("/"):
        context.user_data.pop("hourly_summary_pending", None)
        hour_label = format_hour_range_12h(pending_hour)
        try:
            ai_summary = await ai_generate_hourly_summary_from_text(text)
            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=f"📝 HOURLY AI SUMMARY ({hour_label})\n\n{ai_summary}",
            )
            # await update.message.reply_text(f"✅ Hourly AI summary for {hour_label} posted to group.")
        except Exception as e:
            logger.error(f"Error generating hourly summary: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
        return

    if user_id not in active_users:
        return  # Ignore unless audit started

    # ✅ AI AUDIT EVIDENCE COLLECTION (store under the correct shift)
    # During audit, the operator may paste Shift 1/2/3 reports regardless of current clock shift.
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
        # Save to database when a valid report is received during AI audit
        if target_shift in (1, 2, 3):
            try:
                production_data = parse_report(text)
                downtime = parse_downtime(text)
                rejects = parse_rejects(text)
                save_to_database(production_data, downtime, rejects, shift_override=target_shift)
                logger.info(f"Shift {target_shift} report saved to database (AI audit)")
            except Exception as e:
                logger.warning(f"AI audit DB save skipped: {e}")

    try:
        # Generate next AI question for the operator message
        next_ai_question = await generate_ai_questions_for_message(user_id, text)

        if not next_ai_question:
            # AI failed or no question needed, do nothing
            return

        if next_ai_question == "STOP":
            # Only now mark audit completed
            await context.bot.send_message(
                GROUP_CHAT_ID,
                "✅ Audit completed.\nAll observed issues have been addressed or scheduled.\nNo further AI questions."
            )
            # Clean up session
            active_users.discard(user_id)
            user_ai_sessions.pop(user_id, None)
            user_audit_state.pop(user_id, None)

            # If this STOP came from a manual audit session, unblock reminders and flush AI-muted queue
            global ai_reminder_block
            ai_reminder_block = False
            await flush_pending_reminders(context.bot, reason="ai")
            return

        # Post **only the AI question** to the group
        msg = f"❓ AI Question:\n{next_ai_question}"
        await context.bot.send_message(GROUP_CHAT_ID, msg)

        # Count questions
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

    # Check if already sent today (persisted in bot_state)
    if shift_plan_sent_today[shift] == today:
        logger.info(f"Shift {shift} plan reminder already sent today, skipping")
        return

    # Track that shift plan was sent
    shift_plan_sent_today[shift] = today
    bot_state_set(f"shift_plan_sent_{shift}", today.isoformat())

    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📋 *Shift {shift} Plan Reminder*\n\n"
            + "- Product type\n"
            + "- Shift plan (packs)\n"
            + "- Expected manpower / constraints"
    )
    await send_or_queue_reminder(context, text, parse_mode="Markdown")


async def remind_shift_report(context: ContextTypes.DEFAULT_TYPE):
    shift = context.job.data["shift"]
    now = now_ethiopia()
    today_iso = now.date().isoformat()
    key = f"shift_report_{today_iso}_{shift}"
    if bot_state_get(key):
        logger.info(f"Shift {shift} report reminder already sent today, skipping")
        return
    bot_state_set(key, "1")
    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"📊 *Shift {shift} Summary Report Reminder*\n\n"
            + "- Actual output\n"
            + "- Downtime (reason + minutes)\n"
            + "- Rejects (preform, bottle, cap, label)\n"
            + "- Operator remarks"
    )
    await send_or_queue_reminder(context, text, parse_mode="Markdown")


async def remind_hourly_plan(context: ContextTypes.DEFAULT_TYPE):
    """Hourly plan reminder - what to do in this hour. Skip if already sent for this slot (persisted in bot_state)."""
    # Always derive shift/hour from current clock so titles match /bot_status
    now = now_ethiopia()
    shift = get_shift_for_time(now)
    hour = get_current_hour_number(shift, now)
    today_iso = now.date().isoformat()
    key = f"hourly_plan_{today_iso}_{shift}_{hour}"
    if bot_state_get(key):
        logger.info(f"Hourly plan already sent for Shift {shift} Hour {hour} today, skipping")
        return
    bot_state_set(key, "1")
    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + f"⏰ *Hourly Plan Reminder – Shift {shift}, Hour {hour}*\n\n"
            + "Please share the plan for this hour:\n"
            + "- Production target\n"
            + "- Any scheduled maintenance or adjustments\n"
            + "- Expected challenges"
    )
    await send_or_queue_reminder(context, text, parse_mode="Markdown")


async def remind_hourly_summary(context: ContextTypes.DEFAULT_TYPE):
    """Hourly summary reminder - triggers AI hourly summary. Skip if already sent for this slot (persisted in bot_state)."""
    # Always derive shift/hour from current clock so titles match /bot_status
    now = now_ethiopia()
    shift = get_shift_for_time(now)
    hour = get_current_hour_number(shift, now)
    today_iso = now.date().isoformat()
    key = f"hourly_summary_{today_iso}_{shift}_{hour}"
    if bot_state_get(key):
        logger.info(f"Hourly summary already sent for Shift {shift} Hour {hour} today, skipping")
        return
    bot_state_set(key, "1")
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
    await send_or_queue_reminder(context, text, parse_mode="Markdown")


async def remind_daily_production_plan(context: ContextTypes.DEFAULT_TYPE):
    """
    Daily production plan reminder:
    - Primary at the beginning of Shift 1
    - If missed (e.g. bot down), fallback in Shift 2 or 3
    Only fires once per calendar day.
    """
    global daily_plan_last_date

    today = now_ethiopia().date()
    if daily_plan_last_date == today:
        # Already sent today (either in Shift 1 or a later fallback)
        return

    now = now_ethiopia()
    header = f"📅 {format_date_time_12h(now)}\n\n"
    text = (
            header
            + "📆 *Daily Production Plan Reminder*\n\n"
            + "Please share today's overall production plan:\n"
            + "- Products and SKUs by shift\n"
            + "- Target packs per shift\n"
            + "- Any known constraints (utilities, materials, manpower)."
    )
    await send_or_queue_reminder(context, text, parse_mode="Markdown")
    daily_plan_last_date = today
    bot_state_set("daily_plan_last_date", today.isoformat())


async def setup_shift_schedules(app):
    job_queue = app.job_queue

    logger.info("Setting up shift schedules and reminders...")

    # Daily production plan reminder:
    # - Primary: beginning of Shift 1 (Ethiopian 12:00 -> PC 06:00)
    # - Fallbacks: Shift 2 and 3 plan times, only if not already sent that day
    job_queue.run_daily(
        remind_daily_production_plan,
        time=ethiopian_clock_time_to_pc_time(time(0, 0)),
        name="daily_plan_primary_shift1",
    )
    logger.info("Scheduled: Daily plan reminder at Ethiopian 12:00 (PC 06:00)")

    job_queue.run_daily(
        remind_daily_production_plan,
        time=ethiopian_clock_time_to_pc_time(time(7, 0)),  # Shift 2 start
        name="daily_plan_fallback_shift2",
    )
    job_queue.run_daily(
        remind_daily_production_plan,
        time=ethiopian_clock_time_to_pc_time(time(14, 0)),  # Shift 3 start
        name="daily_plan_fallback_shift3",
    )

    # ========== SHIFT 1 (00:00 - 07:00) ==========
    # Shift 1 plan reminder at 00:05
    job_queue.run_daily(
        remind_shift_plan,
        time=ethiopian_clock_time_to_pc_time(time(0, 5)),
        data={"shift": 1},
        name="shift_1_plan"
    )
    logger.info("Scheduled: Shift 1 plan reminder at 00:05")

    # Shift 1 hourly reminders (hours 1-7)
    shift1_hourly_times = [
        (time(0, 8), 1),  # Hour 1 plan
        (time(0, 55), 1),  # Hour 1 summary (5 min before hour end)
        (time(1, 2), 2),  # Hour 2 plan
        (time(1, 55), 2),  # Hour 2 summary (5 min before hour end)
        (time(2, 2), 3),  # Hour 3 plan
        (time(2, 55), 3),  # Hour 3 summary (5 min before hour end)
        (time(3, 2), 4),  # Hour 4 plan
        (time(3, 55), 4),  # Hour 4 summary (5 min before hour end)
        (time(4, 2), 5),  # Hour 5 plan
        (time(4, 55), 5),  # Hour 5 summary (5 min before hour end)
        (time(5, 2), 6),  # Hour 6 plan
        (time(5, 55), 6),  # Hour 6 summary (5 min before hour end)
        (time(6, 2), 7),  # Hour 7 plan
        (time(6, 45), 7),  # Final hourly summary (Shift 1) - 15 min before shift end
    ]

    for t, hour in shift1_hourly_times:
        if t.minute == 2 or (t.hour == 6 and t.minute == 2):  # Plan reminders
            job_queue.run_daily(
                remind_hourly_plan,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 1, "hour": hour},
                name=f"shift1_hour{hour}_plan"
            )
        else:  # Summary reminders
            job_queue.run_daily(
                remind_hourly_summary,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 1, "hour": hour},
                name=f"shift1_hour{hour}_summary"
            )

    # Shift 1 summary reminder: 5 min before shift end
    job_queue.run_daily(
        remind_shift_report,
        time=ethiopian_clock_time_to_pc_time(time(6, 55)),
        data={"shift": 1},
        name="shift_1_report"
    )
    logger.info("Scheduled: Shift 1 summary reminder at 06:55")

    # ========== SHIFT 2 (07:00 - 14:00) ==========
    # Shift 2 plan reminder at 07:05
    job_queue.run_daily(
        remind_shift_plan,
        time=ethiopian_clock_time_to_pc_time(time(7, 5)),
        data={"shift": 2},
        name="shift_2_plan"
    )
    logger.info("Scheduled: Shift 2 plan reminder at 07:05")

    # Shift 2 hourly reminders (hours 1-7)
    shift2_hourly_times = [
        (time(7, 8), 1),  # Hour 1 plan
        (time(7, 55), 1),  # Hour 1 summary (5 min before hour end)
        (time(8, 2), 2),  # Hour 2 plan
        (time(8, 55), 2),  # Hour 2 summary (5 min before hour end)
        (time(9, 2), 3),  # Hour 3 plan
        (time(9, 55), 3),  # Hour 3 summary (5 min before hour end)
        (time(10, 2), 4),  # Hour 4 plan
        (time(10, 55), 4),  # Hour 4 summary (5 min before hour end)
        (time(11, 2), 5),  # Hour 5 plan
        (time(11, 55), 5),  # Hour 5 summary (5 min before hour end)
        (time(12, 2), 6),  # Hour 6 plan
        (time(12, 55), 6),  # Hour 6 summary (5 min before hour end)
        (time(13, 2), 7),  # Hour 7 plan
        (time(13, 45), 7),  # Final hourly summary (Shift 2) - 15 min before shift end
    ]

    for t, hour in shift2_hourly_times:
        if t.minute == 2 or (t.hour == 13 and t.minute == 2):  # Plan reminders
            job_queue.run_daily(
                remind_hourly_plan,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 2, "hour": hour},
                name=f"shift2_hour{hour}_plan"
            )
        else:  # Summary reminders
            job_queue.run_daily(
                remind_hourly_summary,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 2, "hour": hour},
                name=f"shift2_hour{hour}_summary"
            )

    # Shift 2 summary reminder: 5 min before shift end
    job_queue.run_daily(
        remind_shift_report,
        time=ethiopian_clock_time_to_pc_time(time(13, 55)),
        data={"shift": 2},
        name="shift_2_report"
    )
    logger.info("Scheduled: Shift 2 summary reminder at 13:55")

    # ========== SHIFT 3 (14:00 - 24:00 Ethiopian, 10 hours) ==========
    # Shift 3 plan reminder at 14:05 (Ethiopian) -> 20:05 (PC)
    job_queue.run_daily(
        remind_shift_plan,
        time=ethiopian_clock_time_to_pc_time(time(14, 5)),
        data={"shift": 3},
        name="shift_3_plan"
    )
    logger.info("Scheduled: Shift 3 plan reminder at 14:05")

    # Shift 3 hourly reminders (hours 1-10)
    shift3_hourly_times = [
        (time(14, 8), 1), (time(14, 55), 1),
        (time(15, 2), 2), (time(15, 55), 2),
        (time(16, 2), 3), (time(16, 55), 3),
        (time(17, 2), 4), (time(17, 55), 4),
        (time(18, 2), 5), (time(18, 55), 5),
        (time(19, 2), 6), (time(19, 55), 6),
        (time(20, 2), 7), (time(20, 55), 7),
        (time(21, 2), 8), (time(21, 55), 8),
        (time(22, 2), 9), (time(22, 55), 9),
        (time(23, 2), 10), (time(23, 45), 10),  # final hourly summary 15 min before end
    ]

    for t, hour in shift3_hourly_times:
        if t.minute == 2:  # Plan reminders
            job_queue.run_daily(
                remind_hourly_plan,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 3, "hour": hour},
                name=f"shift3_hour{hour}_plan"
            )
        else:  # Summary reminders
            job_queue.run_daily(
                remind_hourly_summary,
                time=ethiopian_clock_time_to_pc_time(t),
                data={"shift": 3, "hour": hour},
                name=f"shift3_hour{hour}_summary"
            )

    # Shift 3 summary reminder: 5 min before shift end
    job_queue.run_daily(
        remind_shift_report,
        time=ethiopian_clock_time_to_pc_time(time(23, 55)),
        data={"shift": 3},
        name="shift_3_report"
    )
    logger.info("Scheduled: Shift 3 summary reminder at 23:55 (Ethiopian clock)")

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


async def send_daily_plan_if_needed(bot, now: datetime) -> bool:
    """Send daily plan reminder if not sent today. Returns True if sent."""
    global daily_plan_last_date
    today = now.date()
    if daily_plan_last_date != today:
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
            await bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=daily_plan_text,
                parse_mode="Markdown"
            )
            daily_plan_last_date = today
            bot_state_set("daily_plan_last_date", today.isoformat())
            logger.info("Daily plan reminder sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send daily plan reminder: {e}")
    return False


async def send_shift_plan_if_needed(bot, current_shift_num: int, now: datetime) -> bool:
    """Send shift plan reminder if not sent for current shift today. Returns True if sent."""
    global shift_plan_sent_today
    today = now.date()
    if shift_plan_sent_today[current_shift_num] != today:
        header = f"📅 {format_date_time_12h(now)}\n\n"
        shift_plan_text = (
                header
                + f"📋 *Shift {current_shift_num} Plan Reminder*\n\n"
                + "- Product type\n"
                + "- Shift plan (packs)\n"
                + "- Expected manpower / constraints"
        )
        try:
            await bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=shift_plan_text,
                parse_mode="Markdown"
            )
            shift_plan_sent_today[current_shift_num] = today
            bot_state_set(f"shift_plan_sent_{current_shift_num}", today.isoformat())
            logger.info(f"Shift {current_shift_num} plan reminder sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send shift plan reminder: {e}")
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


async def send_current_hour_plan(bot, current_shift_num: int, now: datetime) -> bool:
    """Send hourly plan reminder for current hour only. Skip if already sent for this slot (persisted). Returns True if sent."""
    current_hour = get_current_hour_number(current_shift_num, now)
    current_time = now.time()

    # Check if we're still in the window for hourly plan (first 30 minutes of hour)
    hour_start_minutes = (now.hour * 60 + now.minute) % 60
    if hour_start_minutes > 30:  # Past the 30-minute window, don't send
        return False

    today_iso = now.date().isoformat()
    key = f"hourly_plan_{today_iso}_{current_shift_num}_{current_hour}"
    if bot_state_get(key):
        logger.info(f"Hourly plan already sent for Shift {current_shift_num} Hour {current_hour} today, skipping")
        return False

    bot_state_set(key, "1")
    header = f"📅 {format_date_time_12h(now)}\n\n"
    hourly_plan_text = (
            header
            + f"⏰ *Hourly Plan Reminder – Shift {current_shift_num}, Hour {current_hour}*\n\n"
            + "Please share the plan for this hour:\n"
            + "- Production target\n"
            + "- Any scheduled maintenance or adjustments\n"
            + "- Expected challenges"
    )
    try:
        await bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=hourly_plan_text,
            parse_mode="Markdown"
        )
        logger.info(f"Current hour plan reminder sent for Shift {current_shift_num}, Hour {current_hour}")
        return True
    except Exception as e:
        logger.error(f"Failed to send hourly plan reminder: {e}")
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

    # Only send hourly summary if there was MORE than 20 minutes of production
    if total_production_minutes > 20:
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

    # If near shift end, also send shift summary reminder
    if is_near_shift_end(current_shift_num, line_on_time):
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
    """Check if we're within last 15 minutes of shift end."""
    current_time = now.time()
    if current_shift_num == 1:  # PC: ends 13:00
        return time(12, 45) <= current_time < time(13, 0)
    if current_shift_num == 2:  # PC: ends 20:00
        return time(19, 45) <= current_time < time(20, 0)
    # Shift 3 ends at 06:00 (wrap)
    return time(5, 45) <= current_time < time(6, 0)


async def catch_up_missed_reminders(app, current_shift_num: int, now: datetime):
    """
    When bot starts mid-shift, send reminders in order:
    1. Daily plan if not sent today
    2. Shift plan if not sent for current shift
    3. Current hour plan only (if still in first 30 minutes of hour)
    """
    # Order 1: Daily plan reminder
    await send_daily_plan_if_needed(app.bot, now)
    await asyncio.sleep(1)  # Small delay between reminders

    # Order 2: Shift plan reminder
    await send_shift_plan_if_needed(app.bot, current_shift_num, now)
    await asyncio.sleep(1)

    # Order 3: Current hour plan reminder
    await send_current_hour_plan(app.bot, current_shift_num, now)


async def post_init(app):
    global current_shift, daily_plan_last_date

    # Load persisted state so reboot does not duplicate reminders
    load_bot_state_from_db()

    await setup_bot_commands(app)
    await setup_shift_schedules(app)  # <- this sets up automatic reminders

    # Sync current_shift with actual clock time
    now = now_ethiopia()
    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock
    logger.info(f"Bot started: Synced current_shift to {current_shift} (clock time: {now.strftime('%H:%M:%S')})")

    # Catch up on missed reminders for current shift (daily plan, shift plan, current hour plan)
    await catch_up_missed_reminders(app, current_shift, now)

    # Send startup confirmation to group (using plain text to avoid Markdown parsing errors)
    startup_msg = (
        f"🤖 Bot Started Successfully\n\n"
        f"⏰ Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📅 Current Shift: {current_shift}\n"
        f"🏭 Line State: {line_state}\n"
        f"✅ Reminders: ACTIVE\n\n"
    )
    today = now_ethiopia().date()
    if daily_plan_last_date == today:
        startup_msg += "📆 Daily plan reminder was already sent today.\n"
    else:
        startup_msg += "📆 Daily plan reminder sent above.\n"

    startup_msg += "\nAll scheduled reminders are configured and will fire at their designated times.\nUse /bot_status to check current state."

    try:
        await app.bot.send_message(
            chat_id=GROUP_CHAT_ID,
            text=startup_msg
        )
        logger.info("Startup message sent to group")
    except Exception as e:
        logger.error(f"Failed to send startup message: {e}")


# ---------------- LINE / SANITATION CONTROL COMMANDS ----------------
async def line_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since
    line_state = LINE_STATE_OFF
    line_off_since = now_ethiopia()  # Track when line went off
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", line_off_since.isoformat())
    await update.message.reply_text(
        "⚠️ Line set to OFF.\n"
        "All shift and hourly reminders will be queued until the line is ON."
    )


async def line_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, line_off_since, current_shift
    now = now_ethiopia()
    line_was_off_since = line_off_since
    line_state = LINE_STATE_RUNNING
    line_off_since = None  # Reset tracking
    bot_state_set("line_state", line_state)
    bot_state_set("line_off_since", "")

    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock

    await update.message.reply_text(
        "✅ Line set to ON.\n"
        "Processing reminders and checking for missed items..."
    )

    # Order 1: Send daily plan if not sent today (regardless of shift)
    await send_daily_plan_if_needed(context.bot, now)
    await asyncio.sleep(1)

    # Order 2: Send shift plan if not sent for current shift
    await send_shift_plan_if_needed(context.bot, current_shift, now)
    await asyncio.sleep(1)

    # Order 3: Handle partial hours if line was off
    if line_was_off_since:
        off_time = line_was_off_since.time()
        on_time = now.time()
        off_hour = off_time.hour
        on_hour = on_time.hour
        off_minutes = off_time.hour * 60 + off_time.minute
        on_minutes = on_time.hour * 60 + on_time.minute
        downtime_minutes = on_minutes - off_minutes

        # Check if we're still in the same hour
        if off_hour == on_hour:
            # Same hour - calculate production time
            hour_start_minutes = off_hour * 60
            hour_end_minutes = hour_start_minutes + 60
            production_before_off = off_minutes - hour_start_minutes
            production_after_on = hour_end_minutes - on_minutes
            total_production_minutes = production_before_off + production_after_on

            # Only send summary if production was MORE than 20 minutes
            if total_production_minutes > 20:
                # Send partial hour summary
                await handle_partial_hours_on_line_resume(context.bot, current_shift, line_was_off_since, now)
            else:
                # Less than 20 min production - send hourly plan if still in 30-min window
                minutes_into_hour = on_time.minute
                if minutes_into_hour <= 30:
                    await send_current_hour_plan(context.bot, current_shift, now)
        else:
            # Different hour - check if we're still in first 30 minutes of new hour
            minutes_into_hour = on_time.minute
            if minutes_into_hour <= 30:
                # Still in window for hourly plan of new hour
                await send_current_hour_plan(context.bot, current_shift, now)
            # Otherwise, hour is lost - don't send reminders for it
    else:
        # No line off tracking - just send current hour plan if in window
        await send_current_hour_plan(context.bot, current_shift, now)

    # Check if near shift end - send shift summary reminder
    if is_near_shift_end(current_shift, now):
        shift_summary_text = (
            f"📊 *Shift {current_shift} Summary Report Reminder*\n\n"
            "- Actual output\n"
            "- Downtime (reason + minutes)\n"
            "- Rejects (preform, bottle, cap, label)\n"
            "- Operator remarks"
        )
        try:
            await context.bot.send_message(
                chat_id=GROUP_CHAT_ID,
                text=shift_summary_text,
                parse_mode="Markdown"
            )
            logger.info(f"Shift {current_shift} summary reminder sent (near shift end)")
        except Exception as e:
            logger.error(f"Failed to send shift summary reminder: {e}")

    # Flush any queued reminders (but they're filtered by shift already)
    await flush_pending_reminders(context.bot, reason="line")


async def sanitation_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state
    line_state = LINE_STATE_SANITATION
    bot_state_set("line_state", line_state)
    await update.message.reply_text(
        "🧼 Sanitation started.\n"
        "All reminders will be queued until sanitation is finished."
    )


async def sanitation_end_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global line_state, current_shift
    now = now_ethiopia()
    line_state = LINE_STATE_RUNNING
    bot_state_set("line_state", line_state)

    current_shift_by_clock = get_shift_for_time(now)
    current_shift = current_shift_by_clock

    await update.message.reply_text(
        "✅ Sanitation finished.\n"
        "Processing reminders..."
    )

    # Order 1: Send daily plan if not sent today (regardless of shift)
    await send_daily_plan_if_needed(context.bot, now)
    await asyncio.sleep(1)

    # Order 2: Send shift plan if not sent for current shift
    await send_shift_plan_if_needed(context.bot, current_shift, now)
    await asyncio.sleep(1)

    # Order 3: Send current hour plan if still in window (first 30 minutes)
    await send_current_hour_plan(context.bot, current_shift, now)

    # Note: Don't try to catch up on past hours during sanitation (as per user request)

    # Flush any queued reminders
    await flush_pending_reminders(context.bot, reason="line")


async def all_shift_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generate an AI summary that covers all closed shifts so far.
    - If 2 shifts closed: summary of Shift 1 and 2
    - If 3 shifts closed: summary of Shift 1, 2 and 3
    """
    included_shifts = [s for s in (1, 2, 3) if daily_ai_shift_summaries.get(s)]

    if len(included_shifts) < 2:
        await update.message.reply_text(
            "At least two shift summaries are required.\n"
            "Use /shift_summary for each shift first."
        )
        return

    await generate_multi_shift_summary_and_post(context, included_shifts)

    if len(included_shifts) == 2:
        label = f"shifts {included_shifts[0]} and {included_shifts[1]}"
    else:
        label = f"shifts {', '.join(str(s) for s in included_shifts[:-1])} and {included_shifts[-1]}"

    await update.message.reply_text(
        f"✅ Posted multi-shift AI summary for {label}."
    )


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
    try:
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
    Get all reminders for a shift with their times in 12-hour AM/PM format
    in the ETHIOPIAN CLOCK (what your operators use).
    """
    if shift == 1:  # Ethiopian clock: 12:00 AM – 7:00 AM
        return [
            ("12:00 AM", "Daily Production Plan Reminder"),
            ("12:05 AM", "Shift 1 Plan Reminder"),
            ("12:08 AM", "Hour 1 Plan Reminder"),
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
            ("6:45 AM", "Final Hourly Summary (Shift 1)"),
            ("6:55 AM", "Shift 1 Summary Reminder"),
        ]
    elif shift == 2:  # Ethiopian clock: 7:00 AM – 2:00 PM
        return [
            ("7:05 AM", "Shift 2 Plan Reminder"),
            ("7:08 AM", "Hour 1 Plan Reminder"),
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
            ("1:45 PM", "Final Hourly Summary (Shift 2)"),
            ("1:55 PM", "Shift 2 Summary Reminder"),
        ]
    else:  # Ethiopian clock: 2:00 PM – 12:00 AM (10 hours)
        return [
            ("2:05 PM", "Shift 3 Plan Reminder"),
            ("2:08 PM", "Hour 1 Plan Reminder"),
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
            ("11:45 PM", "Final Hourly Summary (Shift 3)"),
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
        f"✅ Reminders Active: {'Yes' if line_state == LINE_STATE_RUNNING and not ai_reminder_block else 'No'}\n\n"
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

    await update.message.reply_text(status_text, parse_mode="Markdown")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start_audit", start_audit))
    app.add_handler(CommandHandler("end_audit", end_audit))
    app.add_handler(CommandHandler("shift_summary", shift_summary))
    app.add_handler(CommandHandler("shift_input_1", shift_input_1_cmd))
    app.add_handler(CommandHandler("shift_input_2", shift_input_2_cmd))
    app.add_handler(CommandHandler("shift_input_3", shift_input_3_cmd))
    app.add_handler(CommandHandler("shift_summary_1", shift_summary_1_cmd))
    app.add_handler(CommandHandler("shift_summary_2", shift_summary_2_cmd))
    app.add_handler(CommandHandler("shift_summary_3", shift_summary_3_cmd))
    app.add_handler(CommandHandler("all_shift_summary", all_shift_summary_cmd))
    app.add_handler(CommandHandler("hourly_summary_ai", hourly_summary_ai_cmd))
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