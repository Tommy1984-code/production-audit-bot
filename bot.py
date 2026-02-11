import re
import logging
import psycopg2
from datetime import datetime, timedelta
from datetime import datetime, time
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


#this works only thing to make sure is there is 3 times we less from local utc always 3+
SHIFT_SCHEDULE = {
    1: {"plan_time": time(6, 54), "report_time": time(14, 10)},
    2: {"plan_time": time(14, 0), "report_time": time(22, 10)},
    3: {"plan_time": time(22, 0), "report_time": time(6, 10)},  # next day
}
# ---------------- AI SUMMARY EVIDENCE ----------------
ai_shift_evidence = {
    1: [],
    2: [],
    3: []
}
# ---------------- SHIFT STATE ----------------
current_shift = 1   # starts at shift 1
shift_closed = {
    1: False,
    2: False,
    3: False
}



# ---------------- LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------- DATABASE ----------------
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_to_database(data, downtime, rejects):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO production
            (date, shift, product_type, shift_plan_pack, actual_output_pack)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (data["date"], data["shift"], data["product_type"], data["plan"], data["actual"]))
        production_id = cur.fetchone()[0]

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
        """, (production_id, rejects["preform"], rejects["bottle"], rejects["cap"], rejects["label"], rejects["shrink"]))

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
    date = datetime.strptime(date_match.group(1), "%d/%m/%y").date() if date_match else datetime.now().date()
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
    efficiency = round((actual / plan) * 100, 1) if plan else 0
    return {"date": date, "shift": shift, "product_type": product_type, "plan": plan, "actual": actual, "efficiency": efficiency}

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

# ---------------- COMMANDS ----------------
async def start_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    active_users.add(user_id)
    user_ai_sessions[user_id] = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
    user_audit_state[user_id] = {"questions": 0, "completed": False, "ended": False}
    await update.message.reply_text("✅ Audit triggered. Send shift reports. Use /end_audit to stop.")

async def end_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    active_users.discard(user_id)
    user_ai_sessions.pop(user_id, None)
    user_audit_state.pop(user_id, None)
    await update.message.reply_text("🛑 Audit ended. AI questioning stopped.")


async def shift_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_shift

    if not context.args:
        await update.message.reply_text("Usage: /shift_summary 1 | 2 | 3")
        return

    try:
        shift = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Shift must be 1, 2, or 3.")
        return

    if shift != current_shift:
        await update.message.reply_text(
            f"❌ Shift {shift} cannot be summarized.\n"
            f"Current active shift is Shift {current_shift}."
        )
        return

    if not ai_shift_evidence[shift]:
        await update.message.reply_text("No shift data available.")
        return

    # Generate AI summary
    ai_text = await ai_generate_summary(shift)

    # Post directly to Telegram group
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📊 *SHIFT {shift} OFFICIAL SUMMARY*\n\n{ai_text}",
        parse_mode="Markdown"
    )

    # Close shift
    shift_closed[shift] = True

    # Clear shift evidence (important)
    ai_shift_evidence[shift] = []

    # Move to next shift
    if current_shift < 3:
        current_shift += 1
    else:
        current_shift = 1
        shift_closed[1] = False
        shift_closed[2] = False
        shift_closed[3] = False

    await update.message.reply_text(
        f"✅ Shift {shift} closed.\n➡️ Shift {current_shift} is now active."
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
    efficiency = production_data["efficiency"]
    total_downtime = sum(d["duration"] for d in downtime)
    total_reject_units = (
        rejects.get("preform", 0)
        + rejects.get("bottle", 0)
        + rejects.get("cap", 0)
        + rejects.get("label", 0)
    )

    actual_output = production_data["actual"]
    reject_rate = round((total_reject_units / actual_output) * 100, 2) if actual_output else 0

    # Assume 8-hour shift = 480 minutes
    downtime_ratio = round((total_downtime / 480) * 100, 2)

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

    if reject_rate > 5:
        risk_score += 2
    elif reject_rate > 2:
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
                            EFFICIENCY: {efficiency}%
                            TOTAL_DOWNTIME: {total_downtime} minutes
                            DOWNTIME_RATIO: {downtime_ratio}%
                            TOTAL_REJECT_UNITS: {total_reject_units}
                            REJECT_RATE: {reject_rate}%
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
        You are a plant-level executive production analyst.

        Write ONE high-level executive paragraph evaluating:
        - Operational performance
        - Downtime impact
        - Equipment reliability risk
        - Quality impact
        - Overall operational stability

        STRICT RULES:
        - Use numeric format for all numbers (e.g., 42%, 64.07%, 240 minutes)
        - Do NOT convert numbers into words
        - Be analytical, concise, professional, and executive-level
        - Base conclusions strictly on the structured data provided
        - Interpret numbers; do not mechanically repeat them
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

    # ---------------- FINAL REPORT ----------------
    final_report = (
            f"*STATUS:* COMPLETE\n\n"
            f"*RISK LEVEL:* {risk_level}\n\n"
            f"*SUMMARY:*\n{executive_paragraph}\n\n"
            f"*PRODUCTION PERFORMANCE:*\n"
            f"- Product: {production_data['product_type']}\n"
            f"- Plan: {production_data['plan']}\n"
            f"- Actual: {production_data['actual']}\n"
            f"- Efficiency: {efficiency}%\n\n"
            f"*DOWNTIME ANALYSIS:*\n"
            f"- Total Downtime: {total_downtime} minutes\n"
            f"- Downtime Ratio: {downtime_ratio}% of shift\n"
            + "\n".join([f"- {d['description']} ({d['duration']} min)" for d in downtime])
            + "\n\n"
              f"*QUALITY IMPACT:*\n"
              f"- Total Reject Units: {total_reject_units}\n"
              f"- Reject Rate: {reject_rate}%\n"
              f"- Preform: {rejects.get('preform', 0)}\n"
              f"- Bottle: {rejects.get('bottle', 0)}\n"
              f"- Cap: {rejects.get('cap', 0)}\n"
              f"- Label: {rejects.get('label', 0)}\n"
              f"- Shrink Loss: {rejects.get('shrink', 0)} kg\n\n"
              f"*AUDIT STATUS:* {audit_status}"
    )

    return final_report.strip()


# ---------------- MESSAGE HANDLER ----------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if user_id not in active_users:
        return  # Ignore unless audit started

    # ✅ STATE-BASED SHIFT EVIDENCE COLLECTION
    # ✅ STATE-BASED SHIFT EVIDENCE COLLECTION (ignore commands)
    if not text.startswith("/") and not shift_closed[current_shift]:
        ai_shift_evidence[current_shift].append(text)

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
        await app.bot.send_message(chat_id, f"📅 Scheduled Audit:\n❓ AI Question:\n{question}\n\n🛠 Operator answer:\n{message_text}")

async def remind_shift_plan(context: ContextTypes.DEFAULT_TYPE):
    shift = context.job.data["shift"]
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📋 *Shift {shift} Plan Reminder*\n\n"
             "- Product type\n"
             "- Shift plan (packs)\n"
             "- Expected manpower / constraints",
        parse_mode="Markdown"
    )

async def remind_shift_report(context: ContextTypes.DEFAULT_TYPE):
    shift = context.job.data["shift"]
    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=f"📊 *Shift {shift} Summary Report Reminder*\n\n"
             "- Actual output\n"
             "- Downtime (reason + minutes)\n"
             "- Rejects (preform, bottle, cap, label)\n"
             "- Operator remarks",
        parse_mode="Markdown"
    )
async def setup_shift_schedules(app):
    job_queue = app.job_queue
    for shift, times in SHIFT_SCHEDULE.items():
        # Plan reminder
        job_queue.run_daily(
            remind_shift_plan,
            time=times["plan_time"],
            data={"shift": shift},
            name=f"shift_{shift}_plan"
        )
        # Report reminder
        job_queue.run_daily(
            remind_shift_report,
            time=times["report_time"],
            data={"shift": shift},
            name=f"shift_{shift}_report"
        )

# ---------------- BOT SETUP ----------------
async def setup_bot_commands(app):
    commands = [
        BotCommand("start_audit", "Start production audit manually"),
        BotCommand("end_audit", "End current audit"),
        BotCommand("shift_summary", "Generate and close current shift summary (PDF)")
    ]
    await app.bot.set_my_commands(commands)

async def post_init(app):
    await setup_bot_commands(app)
    await setup_shift_schedules(app)  # <- this sets up automatic reminders

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start_audit", start_audit))
    app.add_handler(CommandHandler("end_audit", end_audit))
    app.add_handler(CommandHandler("shift_summary", shift_summary))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # app.post_init = setup_bot_commands
    app.post_init = post_init
    print("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
