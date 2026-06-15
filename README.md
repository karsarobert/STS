# STS - Speech-to-Speech magyar hangvezérlés

🎙️ Magyar nyelvű, lokálisan futó Speech-to-Speech rendszer Gemma 4 / Qwen LLM-mel, Nemotron STT-vel és OmniVoice TTS-sel.

## ✨ Jellemzők

- ✅ **Magyar nyelv támogatás** - STT, LLM és TTS magyar nyelven
- ✅ **Offline működés** - minden komponens lokálisan fut
- ✅ **Voice Activity Detection (VAD)** - automatikus beszéd detektálás
- ✅ **Voice Cloning** - OmniVoice TTS saját hanggal
- ✅ **Real-time streaming** - alacsony latency
- ✅ **Konfigurálható VAD paraméterek** - finomhangolható érzékenység

## 🏗️ Architektúra

```
Mikrofon → VAD (Silero) → STT (Nemotron) → LLM (Gemma 4 / Qwen) → TTS (OmniVoice) → Hangszóró
```

### Komponensek:

- **VAD**: Silero VAD (beszéd detektálás)
- **STT**: Nemotron 3.5 ASR (magyar beszédfelismerés)
- **LLM**: Gemma 4 E4B 12B / Qwen 3.6 35B (OpenAI-kompatibilis API)
- **TTS**: OmniVoice (voice cloning támogatással)

## 📋 Előfeltételek

### Hardver:
- **NVIDIA GPU** (min. 12GB VRAM ajánlott)
- **Mikrofon** és **hangszóró**

### Szoftver:
- **Python 3.10+**
- **CUDA 12.x**
- **Docker** (Nemotron STT konténerhez)
- **Conda / Miniconda**

## 🚀 Telepítés

### 1. Speech-to-Speech könyvtár telepítése

```bash
cd kodok/speech-to-speech
pip install -e .
```

### 2. Függőségek telepítése

```bash
pip install -r requirements.txt
```

### 3. Nemotron STT Docker indítása

```bash
# Építsd meg a Nemotron Docker image-et (lásd NEMOTRON_SETUP.md)
docker run -d --gpus all -p 8002:8000 \
  --name nemotron-speech-docker-cuda \
  nemotron-speech-docker-cuda
```

### 4. LLM szerver indítása

**Unsloth / llama.cpp szerver:**
```bash
# Gemma 4 E4B 12B modell letöltése és szerver indítása
# Lásd: https://github.com/unsloth/unsloth
```

**VAGY vLLM szerver:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --api-key sk-unsloth-xxx
```

## 🎮 Használat

### Egyszerű indítás:

```bash
cd scripts
python run_local.py
```

### Konfiguráció (run_local.py tetején):

```python
# VAD beállítások
MIN_SILENCE_MS = 1200       # Csend után befejezettnek tekinti
MIN_SPEECH_MS = 150         # Minimum beszéd hossz
THRESHOLD = 0.3             # VAD érzékenység (0-1)
SPEECH_PAD_MS = 500         # Audio padding az elejére

# LLM beállítások
LLM_API_URL = "http://127.0.0.1:8888/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF"

# STT/TTS backend
STT_BACKEND = "nemotron"
TTS_BACKEND = "omnivoice"

# Voice cloning (opcionális)
OMNIVOICE_REF_AUDIO = "/path/to/reference.wav"
```

## 🧪 Tesztelés

### Mikrofon teszt:
```bash
bash scripts/test_mic_level.sh
```

### Nemotron STT teszt:
```bash
bash scripts/test_nemotron_simple.sh
```

### WebSocket API teszt:
```bash
python scripts/test_ws.py
```

## 📁 Projekt struktúra

```
STS/
├── scripts/
│   ├── run_local.py              # Fő indító szkript
│   ├── test_mic_level.sh         # Mikrofon szint teszt
│   ├── test_nemotron_simple.py   # STT teszt
│   └── test_ws.py                # WebSocket teszt
├── kodok/
│   └── speech-to-speech/         # Speech-to-speech library (submodule)
├── DEVELOPMENT_PLAN.md           # Többszereplős beszélgetés terv
└── README.md                     # Ez a fájl
```

## ⚙️ VAD Paraméterek finomhangolása

### Ha lemarad a mondat eleje:
```python
SPEECH_PAD_MS = 600  # Növeld (alapértelmezett: 300-500)
```

### Ha lemarad a mondat vége:
```python
MIN_SILENCE_MS = 1500  # Növeld (alapértelmezett: 800-1000)
```

### Ha nem ismeri fel a rövid szavakat ("ok", "igen"):
```python
MIN_SPEECH_MS = 120  # Csökkentsd (alapértelmezett: 150-200)
```

### Ha túl érzékeny (zajokra reagál):
```python
THRESHOLD = 0.5  # Növeld (alapértelmezett: 0.3-0.4)
```

### Ha duplikálódik a szöveg:
```python
SPECULATIVE_REOPEN_MS = 300  # Csökkentsd vagy 0-ra állítsd
UNANSWERED_REOPEN_MS = 500   # Csökkentsd
```

## 🐛 Hibaelhárítás

### 1. "GLIBCXX_3.4.32 not found" hiba

**Megoldás:**
```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
python run_local.py
```

Vagy használd a wrapper szkripteket (`.sh` fájlok).

### 2. Nemotron STT nem válaszol

**Ellenőrzés:**
```bash
docker logs nemotron-speech-docker-cuda
curl http://localhost:8002/health
```

**Újraindítás:**
```bash
docker restart nemotron-speech-docker-cuda
```

### 3. LLM angolul válaszol

Ellenőrizd hogy be van-e állítva:
```python
INIT_CHAT_PROMPT = """Te egy barátságos magyar asszisztens vagy.
Mindig magyarul válaszolj."""

ENABLE_LANG_PROMPT = True
```

### 4. OmniVoice modell lassú

Első használatkor letölti a modelleket (~2-3 GB). Utána gyorsabb lesz.

## 🚧 Jövőbeli fejlesztések (DEVELOPMENT_PLAN.md)

- [ ] **Wake Word Detection** - "Asszisztens", "Szia Reachy" ébresztőszó
- [ ] **Passive Listening** - kontextus követés megszólítás nélkül
- [ ] **Speaker Diarization** - többszereplős beszélgetés támogatás
- [ ] **Context Buffer** - utolsó N kijelentés tárolása
- [ ] **Turn-Taking Logic** - intelligens beszédváltás

Részletek: [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md)

## 📊 Teljesítmény

**Tipikus latency breakdown (RTX 3090, Gemma 4 12B):**
- VAD detektálás: ~50-100ms
- STT (Nemotron): ~150-300ms
- LLM (Gemma 4): ~300-800ms
- TTS (OmniVoice): ~500-1500ms
- **Teljes pipeline**: ~1-3 másodperc

## 📝 Licensz

MIT License (lásd a speech-to-speech submodule licenszét is)

## 🙏 Köszönet

- [speech-to-speech](https://github.com/huggingface/speech-to-speech) - Hugging Face
- [Nemotron ASR](https://nvidia.com) - NVIDIA
- [Gemma 4](https://ai.google.dev/gemma) - Google
- [OmniVoice](https://github.com/omnivore-labs/omnivoice) - Voice cloning
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection

## 📧 Támogatás

Issues és PRs welcome! 🎉

---

**Készítve ❤️-vel Magyarországon 🇭🇺**
