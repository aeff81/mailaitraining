#!/usr/bin/env python3
"""
Skript zum Erzeugen von LoRA-Trainingsdaten für E-Mail-Schreibstile.

Liest E-Mails aus einem Maildir-Ordner, analysiert sie mit einem LLM
und erzeugt Trainingsdaten für eine Low-Rank Adaptation, die es ermöglicht,
E-Mails im eigenen Stil zu schreiben.

Ziel: Nach dem Training kann ein Agent E-Mails schreiben wie:
- "Schreibe eine Mail an meinen Chef und melde mich für heute krank"
- "Schreibe an den Herstellersupport von Produkt X"
- "Schreibe meiner Mutter, dass es mir gut geht"
- "Schreibe an Firma X und kündige meinen Vertrag dort"
"""

import os
import sys
import json
import re
import glob
import email
import imaplib
import time
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("⚠️  Bitte installieren Sie: pip install requests")
    sys.exit(1)

# LLM API Konfiguration
LLM_BASE_URL = "http://127.0.0.1:8081/v1"
LLM_MODEL = "gpt-4o"  # oder dein Modell
LLM_TIMEOUT = 120

# Pfad zum Maildir-Ordner (default ~/training/sentfolder/cur)
MAILDIR_DEFAULT = Path.home() / "training/sentfolder/cur"
OUTPUT_DIR = Path.home() / "training/lora_emails"

# LoRA Trainingsformat: jede Zeile ist ein JSON-Objekt mit "prompt" und "response"
# Diese werden später für LoRA-Training verwendet


def expand_paths(args: list[str]) -> list[Path]:
    """
    Expandiert ~ und Wildcards (*, ?, []) in den Argumenten.
    Funktioniert auch wenn die Shell die Wildcards nicht aufgelöst hat.
    """
    paths = []
    for arg in args:
        expanded = str(Path(arg).expanduser())
        matches = sorted(glob.glob(expanded))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            # Kein Treffer: Pfad trotzdem behalten (für klare Fehlermeldung)
            paths.append(Path(expanded))
    return paths


def find_email_files(path: Path) -> list[Path]:
    """
    Findet alle E-Mail-Dateipfade im Ordner oder gibt die einzelne Datei zurück.
    Liest noch keinen Inhalt – das passiert erst beim Verarbeiten (RAM-freundlich).
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        print(f"❌ Ordner nicht gefunden: {path}")
        return []
    
    files = [p for p in path.iterdir() if p.is_file()]
    print(f"✅ Gefunden {len(files)} E-Mails in {path}")
    return files


def parse_email_content(raw_content: str) -> dict:
    """
    Versucht, den Inhalt einer Maildatei zu parsen.
    """
    try:
        msg = email.message_from_string(raw_content)
        
        result = {
            'subject': msg.get('subject', ''),
            'from': msg.get('from', ''),
            'to': msg.get('to', ''),
            'date': msg.get('date', ''),
            'body': '',
            'raw': raw_content[:2000]  # Nur erste 2000 Zeichen für Analyse
        }
        
        # Extrahiere Body (Text-Parts)
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    result['body'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True)
            if body:
                result['body'] = body.decode('utf-8', errors='ignore')
        
        return result
    except Exception as e:
        print(f" ⚠️ Fehler beim Parsen der Mail: {e}")
        return {'error': str(e)}


def analyze_email_llm(parsed: dict) -> dict:
    """
    Sendet die Mail an das LLM zur Analyse und Kategorisierung.
    """
    prompt = f"""
Analysiere die folgende E-Mail und antworte AUSSCHLIESSLICH mit einem JSON-Objekt
mit genau diesen Feldern:

{{
  "category": "privat" | "beruflich" | "geschäftlich",
  "style": {{
    "formality": "sehr formell" | "formell" | "neutral" | "informell",
    "tone": "freundlich" | "direkt" | "höflich" | "dringlich" | "sachlich",
    "structure": "kurz und knapp" | "detailliert" | "emotional" | "strukturiert"
  }},
  "write_instruction": "Eine kurze Anleitung, wie man eine ähnliche Mail schreibt."
}}

Kategorien:
- privat: persönlich, Familie, Freunde
- beruflich: Arbeitsplatz, Kollegen, Arbeitsabläufe
- geschäftlich: Kunden, Lieferanten, Verträge, Finanzen

E-Mail-Inhalt:
---
Von: {parsed.get('from', 'unknown')}
Betreff: {parsed.get('subject', 'ohne Betreff')}
Datum: {parsed.get('date', 'unbekannt')}
---
{parsed.get('body', '')}
"""
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Du bist ein E-Mail-Analyse-System. Antworte immer mit gültigem JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }
    
    response = request_llm(payload)
    if not response.get('success'):
        return response
    
    # LLM-Antwort (JSON-String) in strukturiertes Dict umwandeln
    content = response.get('content', '')
    try:
        parsed_response = json.loads(content)
        return {
            'success': True,
            'category': parsed_response.get('category', 'unknown'),
            'style': parsed_response.get('style', {}),
            'write_instruction': parsed_response.get('write_instruction', '')
        }
    except json.JSONDecodeError:
        # Fallback: Wenn kein gültiges JSON, versuche Kategorie mit Regex zu extrahieren
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', content)
        return {
            'success': True,
            'category': category_match.group(1) if category_match else 'unknown',
            'style': {},
            'write_instruction': content
        }


def request_llm(payload: dict) -> dict:
    """Sendet eine Anfrage an das LLM API."""
    try:
        url = f"{LLM_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        data = json.dumps(payload)
        resp = requests.post(url, headers=headers, data=data, timeout=LLM_TIMEOUT)
        
        if resp.status_code == 200:
            result = resp.json()
            if 'choices' in result and len(result['choices']) > 0:
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content']
                }
            else:
                return {'success': False, 'error': 'Keine Antwort vom LLM erhalten'}
        else:
            return {'success': False, 'error': f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def process_mail(file_path: str, content: str) -> dict:
    """Verarbeitet eine einzelne Mail: parsen, analysieren, formatieren."""
    print(f"  📧 {Path(file_path).name}")
    
    parsed = parse_email_content(content)
    if 'error' in parsed:
        return {'error': parsed['error']}
    
    analysis = analyze_email_llm(parsed)
    if not analysis.get('success'):
        error_msg = analysis.get('error', 'Unbekannter LLM-Fehler')
        print(f"    ⚠️  LLM-Fehler: {error_msg}")
        return {'error': error_msg}
    
    # Formatierung für LoRA-Training
    # Wir erstellen Trainingspaare: INSTRUCTION -> OUTPUT
    category = analysis.get('category', 'unknown')
    style = analysis.get('style', {})
    instruction = analysis.get('write_instruction', '')
    
    # Prompt: Wie schreibe ich eine Mail in dieser Kategorie/Stil?
    # Response: Die Analyse und Musterbeschreibung
    
    training_entry = {
        "instruction": f"Schreibe eine E-Mail in der Kategorie '{category}' mit folgendem Stil: Formalität={style.get('formality', 'neutral')}, Ton={style.get('tone', 'neutral')}, Struktur={style.get('structure', 'neutral')}",
        "input": f"Von: {parsed.get('from', '')}\nBetreff: {parsed.get('subject', '')}\nKontext: {parsed.get('body', '')[:300]}",
        "output": f"{instruction}\n\nKategorie: {category}\nStil: Formalität={style.get('formality', 'neutral')}, Ton={style.get('tone', 'neutral')}, Struktur={style.get('structure', 'neutral')}\n\nBeispiel aus dieser Mail:\n{parsed.get('body', '')[:800]}"
    }
    
    return training_entry


def main():
    """Hauptfunktion."""
    # Pfade aus Argumenten (Wildcards werden expandiert), sonst Default
    args = sys.argv[1:]
    paths = expand_paths(args) if args else [MAILDIR_DEFAULT]
    print(f"📂 Pfade: {', '.join(str(p) for p in paths)}")
    
    # Ausgabe-Verzeichnis erstellen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "lora_email_training.jsonl"
    print(f"💾 Trainingsdaten: {output_file}")
    
    # Alle Dateipfade sammeln (noch ohne Inhalt zu laden)
    email_files = []
    for path in paths:
        email_files.extend(find_email_files(path))
    
    if not email_files:
        print("❌ Keine E-Mails gefunden. Abbruch.")
        return
    
    print(f"\n🔄 Verarbeite {len(email_files)} E-Mails...")
    print("=" * 60)
    
    success_count = 0
    categories = {}
    
    # Datei am Anfang leeren, dann jede Zeile sofort schreiben (RAM-freundlich)
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in email_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as mf:
                    content = mf.read()
            except Exception as e:
                print(f"  ⚠️ Fehler beim Lesen {file_path}: {e}")
                continue
            
            result = process_mail(str(file_path), content)
            if 'error' in result:
                print(f"  ⚠️  Fehler bei {file_path.name}: {result['error']}")
                continue
            
            # Direkt in die Datei schreiben und sofort auf Platte flushen
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            success_count += 1
            
            # Nur Zähler im RAM halten, nicht die kompletten Einträge
            match = re.search(r'Kategorie: (\w+)', result["output"])
            if match:
                cat = match.group(1)
                categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n✅ Erfolgreich verarbeitet: {success_count}/{len(email_files)}")
    print(f"💾 Trainingsdaten gespeichert: {output_file}")
    
    print("\n📊 Kategorien:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")


if __name__ == "__main__":
    main()
