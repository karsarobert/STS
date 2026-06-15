# STS Többszereplős Beszélgetés - Fejlesztési Terv

## Célkitűzés
Az STS rendszer képes legyen bekapcsolódni többszereplős beszélgetésekbe, és csak akkor reagáljon, amikor **kifejezetten megszólítják**.

---

## Főbb Kihívások

### 1. **Wake Word Detection (Ébresztőszó)**
- Fel kell ismerni, amikor az AI-t megszólítják
- Példa trigger szavak: "Asszisztens", "Claude", "Robot", "AI"
- Magyar nyelvű wake word detection szükséges

### 2. **Speaker Diarization (Beszélő azonosítás)**
- Ki beszél éppen? (Ember A, Ember B, AI)
- Többszereplős környezetben tudni kell, ki szólt meg kit

### 3. **Context-Aware Listening (Kontextus-tudatos hallgatás)**
- Passzív hallgatás: követi a beszélgetést anélkül, hogy válaszolna
- Context buffer: emlékszik a beszélgetés menetére
- Csak akkor aktiválódik, amikor megszólítják

### 4. **Intelligens Turn-Taking (Beszédváltás)**
- Ne szakítsa félbe az embereket
- Várjon, amíg befejezik
- Többszereplős környezetben ne válaszoljon minden kijelentésre

### 5. **VAD Finomhangolás Többszereplős Környezetre**
- Jelenlegi VAD: egyetlen beszélőre optimalizált
- Új VAD: többszereplős, átfedő beszéd kezelése

---

## Architektúra Módosítások

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Input Stream                        │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │   VAD (Silero)       │
         │   - Multi-speaker    │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────────────┐
         │  Wake Word Detector          │
         │  - "Asszisztens" / "Claude"  │
         │  - Magyar nyelvű             │
         └───────────┬──────────────────┘
                     │
                     ├────► Wake word? NO ─► Passive Listening
                     │                       (context buffer)
                     │
                     └────► Wake word? YES
                              │
                  ┌───────────▼──────────────┐
                  │  STT (Nemotron)          │
                  │  - Full transcription    │
                  └───────────┬──────────────┘
                              │
                  ┌───────────▼──────────────┐
                  │  Context Manager         │
                  │  - Last N utterances     │
                  │  - Speaker tracking      │
                  └───────────┬──────────────┘
                              │
                  ┌───────────▼──────────────┐
                  │  LLM (Qwen)              │
                  │  - Response generation   │
                  └───────────┬──────────────┘
                              │
                  ┌───────────▼──────────────┐
                  │  TTS (OmniVoice)         │
                  └──────────────────────────┘
```

---

## Implementációs Lépések

### Phase 1: Wake Word Detection (Ébresztőszó Detektálás)
**Prioritás: MAGAS**

#### 1.1 Wake Word Engine Választás
**Opciók:**
- **OpenWakeWord** (lightweight, magyar támogatás custom training-gel)
- **Porcupine** (Picovoice - commercial, jó magyar support)
- **Snowboy** (deprecated, de működik)
- **Vosk** + keyword spotting

**Javasolt:** OpenWakeWord + custom magyar wake word model

#### 1.2 Implementáció
```python
# Új modul: wake_word_detector.py
class WakeWordDetector:
    def __init__(self, keywords=["asszisztens", "claude"]):
        self.model = load_openwakeword(keywords)
        self.detection_threshold = 0.5
    
    def detect(self, audio_chunk: np.ndarray) -> bool:
        """Visszaadja, hogy ébresztőszót hallott-e."""
        score = self.model.predict(audio_chunk)
        return score > self.detection_threshold
```

#### 1.3 Integráció
- A VAD **után**, de az STT **előtt** fut
- Ha nincs wake word → eldobja az audiót (vagy passive buffer-be teszi)
- Ha van wake word → továbbítja az STT-nek

---

### Phase 2: Passive Listening Mode (Passzív Hallgatás)
**Prioritás: KÖZEPES**

#### 2.1 Context Buffer Manager
```python
class ContextBufferManager:
    def __init__(self, buffer_size=10, max_age_seconds=300):
        self.buffer = deque(maxlen=buffer_size)  # Utolsó N kijelentés
        self.max_age = max_age_seconds
    
    def add_utterance(self, text: str, speaker_id: int, timestamp: float):
        """Hozzáad egy kijelentést a context buffer-hez."""
        self.buffer.append({
            "text": text,
            "speaker": speaker_id,
            "timestamp": timestamp
        })
    
    def get_context(self) -> str:
        """Visszaadja az utolsó N kijelentést LLM context-ként."""
        return "\n".join([f"Speaker {u['speaker']}: {u['text']}" 
                          for u in self.buffer])
```

#### 2.2 Lightweight STT Passzív Módban
- Nemotron STT **folyamatosan fut** háttérben
- Minden kijelentést transcribe-ol, **de nem válaszol**
- Buffer-be menti a context-et
- Amikor wake word jön → a buffer context-et is elküldi az LLM-nek

**Trade-off:**
- **Folyamatos STT**: Drága (GPU usage), de teljes context
- **Csak wake word után**: Olcsó, de nincs előzmény context

**Javasolt:** Hibrid megközelítés
- Lightweight STT vagy keyword spotting passzív módban
- Full STT csak wake word után

---

### Phase 3: Speaker Diarization (Beszélő Azonosítás)
**Prioritás: ALACSONY (nice-to-have)**

#### 3.1 Pyannote.audio Integráció
```python
from pyannote.audio import Pipeline

class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
    
    def identify_speaker(self, audio: np.ndarray) -> int:
        """Visszaadja a speaker ID-t (0, 1, 2, ...)."""
        diarization = self.pipeline({"waveform": audio, "sample_rate": 16000})
        # Return dominant speaker in this segment
        return get_dominant_speaker(diarization)
```

#### 3.2 Context-ben Használat
- LLM prompt: "Speaker 1 asked you a question..."
- Tudja, hogy ki kérdezte → személyre szabott válaszok

---

### Phase 4: Turn-Taking Logic (Beszédváltás Logika)
**Prioritás: MAGAS**

#### 4.1 Jelenlegi Probléma
- VAD azonnal reagál bármilyen beszédre
- Többszereplős környezetben ez zavaró

#### 4.2 Új VAD Logika
```python
class MultiSpeakerVAD:
    def __init__(self):
        self.vad = SileroVAD()
        self.wake_word_detector = WakeWordDetector()
        self.is_listening_mode = False  # Alapból passzív
        self.conversation_timeout = 5.0  # 5s csend után passzívra vált
    
    def process_audio(self, audio_chunk):
        # 1. VAD detekció
        is_speech = self.vad.detect(audio_chunk)
        
        if not is_speech:
            return None
        
        # 2. Wake word detekció (ha passzív módban van)
        if not self.is_listening_mode:
            has_wake_word = self.wake_word_detector.detect(audio_chunk)
            if has_wake_word:
                self.is_listening_mode = True
                return audio_chunk  # Továbbítás STT-nek
            else:
                # Passzív buffer-be menti (opcionális)
                return None
        
        # 3. Aktív módban: minden beszédet feldolgoz
        else:
            return audio_chunk
    
    def check_timeout(self):
        """Ha 5s csend van, visszavált passzív módba."""
        if time_since_last_speech() > self.conversation_timeout:
            self.is_listening_mode = False
```

#### 4.3 Beszélgetés Flow
```
1. Passzív mód: Csak figyel, nem válaszol
2. Wake word: "Asszisztens, mennyi az idő?"
3. Aktív mód: Feldolgozza, válaszol
4. Követő kérdés (5s-on belül): "És holnap milyen idő lesz?"
   → Még aktív, válaszol (nincs szükség újabb wake word-re)
5. 5s csend után: Visszavált passzív módba
```

---

### Phase 5: LLM Prompt Engineering
**Prioritás: MAGAS**

#### 5.1 Rendszer Prompt Módosítás
```python
SYSTEM_PROMPT = """
Te egy asszisztens vagy, aki egy többszereplős beszélgetésben vesz részt.

FONTOS SZABÁLYOK:
1. Csak akkor válaszolj, ha KÖZVETLENÜL téged szólítanak meg
2. Ha mások beszélgetnek egymással, NE szakítsd félbe őket
3. Rövid, természetes válaszokat adj
4. Ha nem vagy biztos, hogy téged kérdeztek, kérdezz vissza
5. A beszélgetés kontextusa:
{conversation_context}

Felhasználó kérdése:
{user_query}

Válaszod:
"""
```

#### 5.2 Context Injection
- Utolsó 5-10 kijelentés az LLM-nek
- "Speaker 1 said: ...", "Speaker 2 said: ...", "You said: ..."

---

## Konkrét Implementációs Feladatok

### Milestone 1: Wake Word Detection MVP
- [ ] OpenWakeWord telepítés és setup
- [ ] Magyar wake word model training ("asszisztens", "claude")
- [ ] Wake word detector modul írása
- [ ] Integráció a VAD pipeline-ba
- [ ] Teszt: Wake word nélküli beszéd → NEM válaszol
- [ ] Teszt: "Asszisztens, hello" → Válaszol

### Milestone 2: Passive Listening
- [ ] Context Buffer Manager implementáció
- [ ] Lightweight STT háttérben (vagy keyword spotting)
- [ ] Buffer perzisztencia (utolsó N kijelentés tárolása)
- [ ] LLM prompt-ba context injection
- [ ] Teszt: "A asks B a question" → AI nem válaszol
- [ ] Teszt: "A: Asszisztens, te mit gondolsz?" → AI válaszol **és** ismeri a context-et

### Milestone 3: Turn-Taking Logic
- [ ] MultiSpeakerVAD osztály írása
- [ ] Aktív/Passzív mód state machine
- [ ] Conversation timeout logika (5s csend → passzív)
- [ ] Követő kérdések kezelése (wake word nélkül, ha aktív)
- [ ] Teszt: Wake word → kérdés → válasz → követő kérdés (wake word nélkül) → válasz
- [ ] Teszt: 5s csend után passzív módba vált

### Milestone 4: Speaker Diarization (Optional)
- [ ] Pyannote.audio telepítés
- [ ] Speaker embedding extraction
- [ ] Speaker ID tracking
- [ ] LLM context-be speaker info
- [ ] Teszt: "Speaker 1 asked you X, Speaker 2 said Y"

---

## Konfigurációs Paraméterek (run_local.py)

```python
# ═══════════════════════════════════════════════
# WAKE WORD beállítások
# ═══════════════════════════════════════════════

# Wake word(ek) - vesszővel elválasztva
WAKE_WORDS = "asszisztens,claude"

# Wake word érzékenység (0-1). Magasabb = kevésbé érzékeny.
WAKE_WORD_THRESHOLD = 0.5

# ═══════════════════════════════════════════════
# CONTEXT BUFFER beállítások
# ═══════════════════════════════════════════════

# Hány kijelentést tároljon a buffer-ben
CONTEXT_BUFFER_SIZE = 10

# Hány másodpercig érvényes a context (5 perc)
CONTEXT_MAX_AGE_SEC = 300

# Passzív hallgatás engedélyezése (folyamatos STT háttérben)
ENABLE_PASSIVE_LISTENING = True

# ═══════════════════════════════════════════════
# TURN-TAKING beállítások
# ═══════════════════════════════════════════════

# Ennyi csend után vált vissza passzív módba (ms)
ACTIVE_MODE_TIMEOUT_MS = 5000

# Follow-up kérdések engedélyezése wake word nélkül
ALLOW_FOLLOWUP_WITHOUT_WAKE_WORD = True
```

---

## Várható Eredmények

### Előnyök
✅ Természetes többszereplős beszélgetés támogatás  
✅ Nem szakítja félbe az embereket  
✅ Kontextus-tudatos válaszok  
✅ Energiatakarékos (passzív módban nem fut LLM)  
✅ Skálázható több résztvevőre  

### Kihívások
⚠️ Wake word detekció pontossága magyar nyelven  
⚠️ Folyamatos STT háttérben → GPU terhelés  
⚠️ Speaker diarization accuracy  
⚠️ False positive wake word detekciók  

---

## Alternatív Megközelítések

### 1. **Push-to-Talk (PTT) Button**
- Legegyszerűbb: Fizikai/virtuális gomb
- Gomb lenyomva → AI hallgat
- Gomb felengedve → AI válaszol
- **Pro:** 100% pontos, nincs false positive
- **Con:** Nem "hands-free"

### 2. **Visual Cue Detection (Kamera)**
- Kamera követi, ki néz az AI felé
- Ha valaki a kamerába néz → aktiválódik
- **Pro:** Természetes interakció
- **Con:** Privacy concerns, extra hardver

### 3. **BLE/Phone Trigger**
- Mobil app → BLE jelzés az AI-nek
- Telefon gomb → AI aktiválódik
- **Pro:** Egyszerű, megbízható
- **Con:** Extra eszköz szükséges

---

## Következő Lépések

1. **MVP Prototípus**: Wake word detection + passzív mód
2. **Pilot Test**: 2-3 résztvevős beszélgetés tesztelése
3. **Iteráció**: VAD/Wake word threshold finomhangolás
4. **Production**: Speaker diarization + multi-room support

---

## Költségbecslés (Fejlesztési Idő)

| Milestone | Becsült Idő | Nehézség |
|-----------|--------------|----------|
| Wake Word Detection MVP | 3-5 nap | Közepes |
| Passive Listening | 2-3 nap | Könnyű |
| Turn-Taking Logic | 2-3 nap | Könnyű |
| Speaker Diarization | 5-7 nap | Nehéz |
| LLM Prompt Engineering | 1-2 nap | Könnyű |
| **ÖSSZESEN** | **13-20 nap** | - |

---

## Függőségek (Python Packages)

```bash
pip install openwakeword  # Wake word detection
pip install pyannote.audio  # Speaker diarization
pip install webrtcvad  # Alternatív VAD
```

---

## Referenciák

- [OpenWakeWord GitHub](https://github.com/dscripka/openWakeWord)
- [Pyannote Speaker Diarization](https://github.com/pyannote/pyannote-audio)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Multi-Speaker STT Best Practices](https://arxiv.org/abs/2012.12345)
