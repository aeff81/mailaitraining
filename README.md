UNGETESTET!


# Anleitung: LoRA-Trainingsdaten aus E-Mails erstellen und ein Modell trainieren

## Überblick

Dieser Workflow erzeugt aus deinen gesendeten E-Mails Trainingsdaten für ein LoRA-Modell, das später E-Mails in deinem persönlichen Stil schreibt.

```
E-Mails (Maildir) → LLM-Analyse → JSONL-Trainingsdaten → LoRA-Training → Adapter
```

**Wichtige Erkenntnis:** Das Modell, das die E-Mails *analysiert* (via API), muss **nicht** dasselbe sein wie das Modell, das später *trainiert* und *verwendet* wird. Die Analyse erzeugt nur die Trainingsdaten. Das Trainingsmodell lernt daraus den Stil.

---

## 1. Modell-Empfehlung

### Empfohlen: Qwen2.5-7B-Instruct

| Kriterium | Bewertung |
|-----------|-----------|
| Deutsch | Sehr gut (mehrsprachig trainiert) |
| Größe | 7B – mit QLoRA auf ~8-12 GB VRAM trainierbar |
| llama.cpp | Exzellent unterstützt (GGUF verfügbar) |
| Anweisungsfolge | Sehr gut (Instruct-Version) |
| Stil-Anpassung | Gut geeignet |

**Alternative:** `Mistral-7B-Instruct-v0.3` – ebenfalls sehr gut für Deutsch, etabliert in der llama.cpp-Community.

**Wenn weniger VRAM:** `Qwen2.5-3B-Instruct` oder `Llama-3.2-3B-Instruct` (beide ~6 GB mit QLoRA).

**Warum nicht größer?** Für Stil-Anpassungen (Formalität, Ton, Struktur) reicht ein 7B-Modell völlig. Die **Qualität der Trainingsdaten** ist hier entscheidender als die Modellgröße.

---

## 2. Trainingsdaten erzeugen

```bash
cd ~/training

# Alle E-Mails aus dem Maildir-Ordner verarbeiten
python3 02gen_trainingdata.py ~/training/sentfolder/cur

# Oder mit Wildcards (einzelne Dateien)
python3 02gen_trainingdata.py ~/training/sentfolder/cur/*.eml

# Oder eine einzelne Datei
python3 02gen_trainingdata.py ~/training/sentfolder/cur/1572632950.M409971P26447.starbuck,S=677,W=697:2,S
```

**Ausgabe:** `~/training/lora_emails/lora_email_training.jsonl`

Das Skript schreibt nach **jeder** verarbeiteten E-Mail sofort in die Datei (streaming). Bei Abbruch bleiben bereits erzeugte Einträge erhalten.

**Format jeder Zeile:**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**LLM-Endpoint:** `http://127.0.0.1:8081/v1` (llama.cpp-server, OpenAI-kompatibel)

---

## 3. LoRA trainieren

### Voraussetzungen

```bash
pip install torch transformers peft trl datasets bitsandbytes accelerate
```

### Training starten

```bash
cd ~/training

# Qwen2.5-7B-Instruct mit QLoRA (4-Bit) trainieren
python3 03train_lora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 2 \
  --lora-r 16 \
  --lora-alpha 32
```

**Alternative mit lokalem GGUF-Modell:** Das Trainingsskript benötigt ein HuggingFace-Format-Modell (`.safetensors`), nicht GGUF. Falls du nur eine GGUF-Datei hast:

```bash
# GGUF → HF Format konvertieren (im transformers Repo)
python3 convert_gguf_to_hf.py model.gguf --output_dir ./model_hf
```

Oder lade das Originalmodell direkt von HuggingFace (`Qwen/Qwen2.5-7B-Instruct`).

### Wichtige Parameter

| Parameter | Empfehlung | Beschreibung |
|-----------|-----------|--------------|
| `--epochs` | 3-5 | Mehr Epochen = stärkere Anpassung, aber Overfitting-Risiko |
| `--batch-size` | 2 | Bei VRAM-Mangel auf 1 senken |
| `--lora-r` | 16 | Rang; höher = mehr Kapazität, mehr Speicher |
| `--lora-alpha` | 32 | Skalierung; Faustregel: 2×r |
| `--no-4bit` | - | Volle Präzision (mehr VRAM, evtl. bessere Qualität) |

**Ausgabe:** `~/training/lora_emails/lora_model/` mit `adapter_model.safetensors` und `adapter_config.json`

---

## 4. Trainierten Adapter verwenden

### Option A: Adapter dynamisch laden (zum Testen)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter = "~/training/lora_emails/lora_model"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter)

prompt = """### Instruction:
Schreibe eine E-Mail in der Kategorie 'beruflich' mit folgendem Stil: Formalität=formell, Ton=höflich, Struktur=strukturiert

### Input:
Von: Absender Name
Betreff: Krankmeldung
Kontext: Melde mich heute krank

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option B: Adapter mergen (für llama.cpp)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(model, "~/training/lora_emails/lora_model")
model = model.merge_and_unload()  # Adapter dauerhaft einbauen

model.save_pretrained("~/training/lora_emails/merged_model")
tokenizer.save_pretrained("~/training/lora_emails/merged_model")
```

Danach zu GGUF konvertieren:

```bash
# llama.cpp konvertieren (mit HF-Modell)
python3 llama.cpp/convert_hf_to_gguf.py ~/training/lora_emails/merged_model \
  --outfile ~/training/lora_emails/merged_model.gguf \
  --outtype q8_0
```

### Option C: LoRA direkt in llama.cpp laden (ohne mergen)

Der llama.cpp-server unterstützt LoRA-Adapters direkt:

```bash
llama-server \
  -m model.gguf \
  --lora ~/training/lora_emails/lora_model/adapter_model.safetensors \
  --lora-scale 1.0 \
  --port 8081
```

**Hinweis:** llama.cpp erwartet LoRA im GGUF-Format. Konvertiere den Adapter falls nötig:

```bash
python3 llama.cpp/convert_lora_to_gguf.py \
  ~/training/lora_emails/lora_model \
  --outfile ~/training/lora_emails/lora_model.gguf \
  --base-gguf model.gguf
```

---

## 5. Modell-Fragen: Was ist mit welchem Modell kompatibel?

| Situation | Möglich? |
|-----------|----------|
| Training + Inference mit demselben Modell | ✅ Ja (empfohlen) |
| QLoRA: großes Modell in 4-Bit trainieren | ✅ Ja |
| Kleines Modell trainieren und auch verwenden | ✅ Ja |
| Adapter von Model A auf Model B anwenden | ❌ Nein |
| Analyse-Modell ≠ Trainingsmodell | ✅ Ja (Analyse erzeugt nur Daten) |

---

## 6. Typische Probleme

### "Keine E-Mails gefunden"
- Pfad prüfen: `ls ~/training/sentfolder/cur | head`
- Maildir-Dateien haben keine `.eml`-Endung (z.B. `1572632950.M409971P26447.starbuck,S=677,W=697:2,S`)
- Wildcards in Anführungszeichen setzen, wenn die Shell sie nicht auflöst

### "LLM-Fehler" in der Ausgabe
- llama.cpp-server prüfen: `curl http://127.0.0.1:8081/v1/models`
- Endpoint in `02gen_trainingdata.py` anpassen (Zeile ~33)

### Kategorien zeigen alle "unknown"
- LLM antwortet nicht im JSON-Format → Prompt oder `response_format` prüfen
- Logs des LLM-Servers ansehen

### CUDA Out of Memory beim Training
- `--batch-size 1` verwenden
- `--lora-r 8` statt 16
- Sicherstellen, dass QLoRA aktiv ist (nicht `--no-4bit`)

### Adapter lädt nicht in llama.cpp
- Adapter muss zu GGUF konvertiert werden (`convert_lora_to_gguf.py`)
- Basis-Modell muss exakt dasselbe sein wie beim Training

---

## 7. Dateien

| Datei | Zweck |
|-------|-------|
| `~/training/02gen_trainingdata.py` | Erzeugt Trainingsdaten aus E-Mails |
| `~/training/03train_lora.py` | Trainiert den LoRA-Adapter |
| `~/training/lora_emails/lora_email_training.jsonl` | Trainingsdaten |
| `~/training/lora_emails/lora_model/` | Trainierter Adapter |
| `~/training/Anleitung.md` | Diese Anleitung |

---

## 8. Schnellstart (Zusammenfassung)

```bash
cd ~/training

# 1. Trainingsdaten erzeugen
python3 02gen_trainingdata.py ~/training/sentfolder/cur

# 2. Daten prüfen
wc -l ~/training/lora_emails/lora_email_training.jsonl

# 3. LoRA trainieren
pip install torch transformers peft trl datasets bitsandbytes accelerate
python3 03train_lora.py --model Qwen/Qwen2.5-7B-Instruct

# 4. Adapter testen (siehe Option A oben)
# 5. Für llama.cpp: Adapter mergen oder konvertieren (Option B/C)
```
