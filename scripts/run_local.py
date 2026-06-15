#!/home/karsa-robert/miniconda3/bin/python3
"""STS Runner — beszélgetés a Gemma 4-gyel mikrofonon keresztül.

Konfigurálható VAD paraméterek a lap tetején.
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ═══════════════════════════════════════════════
# VAD (Voice Activity Detection) paraméterek
# ═══════════════════════════════════════════════

# Ennyi ms csend után számít befejezettnek a beszéd.
# Ha gyakran a szavadba vágnak vagy a mondat vége lemarad, növeld!
MIN_SILENCE_MS = 1200  # Növelve 600-ról - várj tovább mielőtt befejezettnek tekinted

# Ennyi ms aktív beszéd kell egy érvényes mondathoz.
# Ha túl alacsony, rövid szavakat (ok, igen, nem) is felismeri.
# Ha túl magas, ezeket eldobja.
MIN_SPEECH_MS = 150  # Optimális rövid válaszokhoz (ok, igen, nem, jó)

# VAD érzékenység (0-1). Alacsonyabb = érzékenyebb.
# Ha a környezeti zaj miatt folyamatosan nyitva marad, növeld.
THRESHOLD = 0.5  # Csökkentve 0.5-ről 0.3-ra - érzékenyebb!

# Ennyi ms audio padding a beszéd ELEJÉRE - ne maradjon le a kezdet!
SPEECH_PAD_MS = 500  # Default: 200-300, növelve hogy ne maradjon le az eleje

# Folytatott beszédnél (gyors válaszok) ennyi ms elég
# Rövid "ok", "igen", "nem" válaszokhoz
MIN_SPEECH_CONTINUATION_MS = 100  # Folytatásnál alacsonyabb threshold

# Ha a válasz után ennyi ms-ig nem beszélsz, újra figyel.
# Növeld, ha lassabban szeretnéd folytatni a beszélgetést.
UNANSWERED_REOPEN_MS = 2000

# ═══════════════════════════════════════════════
# LLM (Nyelvi Modell) beállítások
# ═══════════════════════════════════════════════

# Az LLM API címe (OpenAI-kompatibilis Responses API)
#
# Példák:
#   Helyi:           http://127.0.0.1:8888/v1
#   OpenAI:          http://10.0.64.2:8000/v1
#   OpenRouter:      https://openrouter.ai/api/v1
#   Together AI:     https://api.together.xyz/v1
LLM_API_URL = "http://127.0.0.1:8888/v1"

# A modell neve (az API által támogatott név)
#
# Példák:
#   Helyi Gemma 4:   unsloth/gemma-4-E4B-it-GGUF
#   OpenAI:          Qwen3.6-35B
#   OpenRouter:      anthropic/claude-sonnet-4
#   Together AI:     meta-llama/Llama-4-17B-Instruct
LLM_MODEL = "unsloth/gemma-4-12b-it-GGUF"

# API kulcs (ha üres '' , az api_key.txt-ből olvassa)
# Ha különböző API-t használsz, írd IDE pl.:
# LLM_API_KEY = "sk-proj-..."
LLM_API_KEY = "sk-unsloth-a29763b139e418a3c4a1f43c7ebe05ac"

# STT választás: "parakeet-tdt" vagy "nemotron"
STT_BACKEND = "nemotron"
STT_LANGUAGE = "hu"

# TTS választás: "edge-tts" vagy "omnivoice"
TTS_BACKEND = "omnivoice"

# OmniVoice voice clone referencia (ha TTS_BACKEND = omnivoice)
# Hasznalja a teszt.wav-ot mintakent, hogy konzisztens hangon szolaljon meg
OMNIVOICE_REF_AUDIO = "/home/karsa-robert/Asztal/teszt.wav"
OMNIVOICE_REF_TEXT = "Hello world, this is a test"

# ═══════════════════════════════════════════════
# CHAT / LLM Prompt beállítások
# ═══════════════════════════════════════════════

# Kezdő prompt - állítsd be a kívánt viselkedést
INIT_CHAT_PROMPT = """Te egy barátságos magyar asszisztens vagy.
Mindig magyarul válaszolj, röviden és természetesen.
Segítőkész vagy és udvariasan beszélsz."""

# Automatikusan a felismert nyelvén válaszol (magyar)
ENABLE_LANG_PROMPT = True

# ═══════════════════════════════════════════════

# API kulcs: ha LLM_API_KEY üres, olvassa a fájlból
if LLM_API_KEY:
    api_key = LLM_API_KEY
else:
    with open("/home/karsa-robert/hermes/STS/scripts/api_key.txt") as f:
        api_key = f.read().strip()

env = os.environ.copy()
env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
env["PYTHONUNBUFFERED"] = "1"
# OmniVoice konfiguráció környezeti változókon át
env["OMNIVOICE_REF_AUDIO"] = OMNIVOICE_REF_AUDIO
if OMNIVOICE_REF_TEXT:
    env["OMNIVOICE_REF_TEXT"] = OMNIVOICE_REF_TEXT

cmd = [
    "/home/karsa-robert/miniconda3/bin/speech-to-speech",
    "--mode", "local",
    "--stt", STT_BACKEND,
    "--parakeet_tdt_device", "auto",
    "--tts", TTS_BACKEND,
    "--llm_backend", "responses-api",
    "--model_name", LLM_MODEL,
    "--responses_api_base_url", LLM_API_URL,
    "--responses_api_api_key", api_key,
    "--responses_api_disable_thinking",  # Kikapcsoljuk az érvelési módot!
    "--init_chat_prompt", INIT_CHAT_PROMPT,  # Magyar system prompt
    "--enable_lang_prompt",  # Automatikus nyelv detektálás
    "--min_silence_ms", str(MIN_SILENCE_MS),
    "--min_speech_ms", str(MIN_SPEECH_MS),
    "--min_speech_continuation_ms", str(MIN_SPEECH_CONTINUATION_MS),  # Rövid folytatások
    "--thresh", str(THRESHOLD),
    "--speech_pad_ms", str(SPEECH_PAD_MS),  # Audio padding az elejére
    "--unanswered_reopen_ms", str(UNANSWERED_REOPEN_MS),
    "--log_level", "INFO",  # DEBUG módra állítva a hangdetektálás miatt
]

print("STS — Speech-to-Speech beszélgetés", flush=True)
print(f"  LLM: {LLM_MODEL}", flush=True)
print(f"  STT: {STT_BACKEND} (nyelv: {STT_LANGUAGE})", flush=True)
print(f"  TTS: {TTS_BACKEND}", flush=True)
print(f"  VAD: csend={MIN_SILENCE_MS}ms, min_beszéd={MIN_SPEECH_MS}ms, pad={SPEECH_PAD_MS}ms", flush=True)
print(f"  Prompt: Magyar asszisztens (auto-lang: {ENABLE_LANG_PROMPT})", flush=True)
print("Loading models...", flush=True)
os.execve(cmd[0], cmd, env)
