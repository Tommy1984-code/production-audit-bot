import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from openai import OpenAI
import json

# ---------------- CONFIG ----------------
BOT_TOKEN = "8331725306:AAF_g6rT2SVR21wguL-2AYdH2nQHSrDbyhA"
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free"

# ---------------- LOGGING ----------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ---------------- AI FUNCTION ----------------
def ai_agent_question():
    """
    Ask AI to generate a question automatically.
    Returns JSON {"status": "question", "questions": [...]}
    """
    prompt = """
    You are a friendly AI agent in a Telegram bot.
    Greet the user and ask a simple question to start the conversation.
    Respond ONLY in strict JSON format: {"status":"question","questions":["your question here"]}
    """
    try:
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content.strip()

        # Extract JSON safely
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1:
            json_content = content[first_brace:last_brace+1]
            return json.loads(json_content)
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"AI request failed: {e}\nAI response was:\n{content}")
        return {"status": "ok"}

# ---------------- COMMAND ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    When /start is sent, AI automatically asks a question.
    """
    ai_result = ai_agent_question()

    if ai_result.get("status") == "question":
        questions_text = "\n".join(ai_result.get("questions", []))
        await update.message.reply_text(f"AI asks:\n{questions_text}")
    else:
        await update.message.reply_text("Hello! AI says hi, but has no question right now.")

# ---------------- MAIN ----------------
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    print("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
