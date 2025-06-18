import sys
import os
import json
import time
import threading
import queue
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import collections
import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
                               QFileDialog, QSpinBox, QCheckBox, QComboBox, QGroupBox,
                               QGridLayout, QSlider, QTabWidget, QLineEdit) 
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QFont, QPixmap, QTextCursor

import pyaudio
import wave
import numpy as np

try:
    import whisper
    import tempfile
    import io
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    from mss import mss
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pynput.keyboard import Key, Controller as KeyController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SlideContent:
    slide_number: int
    title: str
    content: str
    full_text: str

@dataclass
class TransitionDecision:
    should_transition: bool
    target_slide: int
    confidence: float
    reasoning: str

class AudioRecorder(QThread):
    audio_data = Signal(bytes)
    
    def __init__(self):
        super().__init__()
        self.recording = False
        self.audio_queue = queue.Queue()
        
    def run(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                          input=True, frames_per_buffer=CHUNK)
            logger.info("Audio recording started")
            while self.recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_data.emit(data)
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            p.terminate()
            logger.info("Audio recording stopped.")
    
    def start_recording(self):
        self.recording = True
        self.start()
    
    def stop_recording(self):
        self.recording = False

class SpeechTranscriber(QThread):
    transcript_updated = Signal(str)
    
    def __init__(self, model_name: str = "base"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.running = False
        
        self.sample_rate = 16000
        self.buffer_duration_seconds = 10  
        self.process_interval_seconds = 2
        self.max_buffer_size = self.sample_rate * self.buffer_duration_seconds
        
        self.audio_buffer = collections.deque(maxlen=self.max_buffer_size)
        self.buffer_lock = threading.Lock()
        self.last_transcript = ""
        
    def initialize_model(self):
        if not WHISPER_AVAILABLE:
            logger.error("Whisper library not found. Please install with 'pip install openai-whisper'.")
            return False
        try:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            return False
    
    def add_audio_data(self, audio_data: bytes):
        with self.buffer_lock:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            self.audio_buffer.extend(audio_np)
    
    def run(self):
        if not self.model:
            logger.error("Transcription model not initialized. Stopping thread.")
            return
            
        self.running = True
        logger.info("Speech transcription thread started.")
        
        while self.running:
            time.sleep(self.process_interval_seconds)
            
            with self.buffer_lock:
                if len(self.audio_buffer) < self.sample_rate:
                    continue
                buffer_snapshot = np.array(self.audio_buffer)

            try:
                audio_float = buffer_snapshot.astype(np.float32) / 32768.0
                
                result = self.model.transcribe(audio_float, language="en", fp16=False)
                full_text = result["text"].strip()
                
                if full_text and len(full_text) > len(self.last_transcript):
                    new_text_start = full_text.rfind(self.last_transcript)
                    if new_text_start != -1:
                        new_part = full_text[new_text_start + len(self.last_transcript):].strip()
                    else:
                        new_part = " ".join(full_text.split()[-10:])

                    if new_part:
                        self.transcript_updated.emit(new_part)

                self.last_transcript = full_text
                        
            except Exception as e:
                logger.error(f"Transcription error: {e}")

        logger.info("Speech transcription thread stopped.")

    def stop(self):
        self.running = False
        self.last_transcript = ""
        self.audio_buffer.clear()

class PDFProcessor:    
    @staticmethod
    def extract_slides(pdf_path: str) -> List[SlideContent]:
        if not PDF_AVAILABLE:
            logger.error("PyMuPDF not available")
            return []
        slides = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                lines = text.split('\n')
                title = lines[0] if lines else f"Slide {page_num + 1}"
                slide = SlideContent(
                    slide_number=page_num + 1,
                    title=title.strip(),
                    content='\n'.join(lines[1:]) if len(lines) > 1 else "",
                    full_text=text
                )
                slides.append(slide)
            doc.close()
            logger.info(f"Extracted {len(slides)} slides from PDF")
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
        return slides

class ScreenOCR(QThread):
    current_slide_detected = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.screen_region = None
    
    def set_screen_region(self, x: int, y: int, width: int, height: int):
        self.screen_region = {"top": y, "left": x, "width": width, "height": height}
    
    def run(self):
        if not OCR_AVAILABLE:
            logger.warning("OCR components not available - screen OCR disabled")
            return
        self.running = True
        try:
            with mss() as sct:
                while self.running:
                    try:
                        if self.screen_region:
                            screenshot = sct.grab(self.screen_region)
                        else:
                            screenshot = sct.grab(sct.monitors[1])
                        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            self.current_slide_detected.emit(text.strip())
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"OCR error: {e}")
                        time.sleep(1)
        except Exception as e:
            logger.error(f"Screen capture initialization error: {e}")
    
    def stop(self):
        self.running = False

class AISlideAgent:
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.slides: List[SlideContent] = []
        self.current_slide = 1
        self.transcript_history = collections.deque(maxlen=10)
        self.transition_history = []
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def set_slides(self, slides: List[SlideContent]):
        self.slides = slides
        
    def add_transcript(self, text: str):
        self.transcript_history.append({
            'timestamp': time.time(),
            'text': text
        })
    
    def analyze_transition(self) -> TransitionDecision:
            if not self.api_key or not self.slides or not self.transcript_history:
                return TransitionDecision(False, self.current_slide, 0.0, "Prerequisites not met (API key, slides, or transcript).")
            
            full_deck_context_parts = []
            for slide in self.slides:
                marker = " (YOU ARE HERE)" if slide.slide_number == self.current_slide else ""
                
                slide_text_block = f"""--- Slide {slide.slide_number}{marker}: {slide.title} ---
    {slide.full_text}"""
                full_deck_context_parts.append(slide_text_block)

            full_deck_context = "\n\n".join(full_deck_context_parts)
            
            now = time.time()
            transcript_log = "\n".join(
                [f"[T-{now - item['timestamp']:.1f}s] {item['text']}" for item in self.transcript_history]
            )
            
            prompt = f"""
You are an expert AI assistant for presentations. Your task is to analyze a speaker's live speech and decide if they intend to change slides.

### PRESENTATION CONTEXT
- Total slides: {len(self.slides)}
- The speaker is currently on slide number: {self.current_slide}

### FULL SLIDE DECK CONTENT
Here is the text content for all slides in the presentation. Read this carefully to understand the entire presentation flow.
{full_deck_context}
--- END OF SLIDE DECK ---

### SPEAKER'S RECENT UTTERANCES
(Entries are timestamped relative to now)
---
{transcript_log}
---

### TASK
Based on all the context above, analyze the speaker's most recent utterances.
1.  **Evaluate Intent:** Has the speaker finished discussing the *current slide's* content? Are they introducing a topic from another slide?
2.  **Look for Cues:** Identify transition phrases ("moving on," "next," "so that brings us to...") or references to content on other slides (e.g., "as we saw on the first slide...", "the slide on future work shows...").
3.  **Be Conservative:** Do NOT trigger a transition for questions, brief tangents, or if the speaker is still elaborating on the current slide's points. It's better to miss a transition than to switch prematurely.
4.  **Determine Target:** If a transition is warranted, determine the target slide number. It's usually the next slide ({self.current_slide + 1}), but the speaker might explicitly refer to another slide by number or by its content/title.
5.  **Provide Confidence:** Rate your confidence on a scale of 0.0 to 1.0. High confidence (>0.8) means a very clear verbal cue was given. Medium confidence (0.6-0.8) means the topic seems concluded. Low confidence means it's ambiguous.

### RESPONSE FORMAT
Respond ONLY with a valid JSON object in the following format:
{{
    "should_transition": boolean,
    "target_slide": integer,
    "confidence": float,
    "reasoning": "A brief, one-sentence explanation for your decision."
}}
    """

            try:
                headers = {
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                        "maxOutputTokens": 256
                    }
                }
                
                response = requests.post(self.base_url, headers=headers, json=data)
                response.raise_for_status()
                
                result_data = response.json()
                content = result_data['candidates'][0]['content']['parts'][0]['text']
                result = json.loads(content)
                
                decision = TransitionDecision(
                    should_transition=result.get('should_transition', False),
                    target_slide=result.get('target_slide', self.current_slide),
                    confidence=result.get('confidence', 0.0),
                    reasoning=result.get('reasoning', '')
                )
                
                self.transition_history.append({
                    'timestamp': time.time(),
                    'decision': decision,
                    'transcript': transcript_log
                })
                return decision
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}. Response content: {content if 'content' in locals() else 'No content'}")
                return TransitionDecision(False, self.current_slide, 0.0, f"JSON Error: {e}")
            except requests.RequestException as e:
                logger.error(f"API request error: {e}")
                return TransitionDecision(False, self.current_slide, 0.0, f"API Error: {e}")
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing API response structure: {e}. Response: {result_data}")
                return TransitionDecision(False, self.current_slide, 0.0, f"API Response Error: {e}")
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                return TransitionDecision(False, self.current_slide, 0.0, f"General Error: {e}")


class SlideController:    
    def __init__(self):
        self.keyboard = None
        if PYNPUT_AVAILABLE:
            try:
                self.keyboard = KeyController()
            except Exception as e:
                logger.error(f"Failed to initialize keyboard controller: {e}")
        self.last_action_time = 0
        self.min_action_interval = 0.01
    
    def advance_slide(self) -> bool:
        if not self.keyboard: return False
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval: return False
        try:
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)
            self.last_action_time = current_time
            logger.info("Slide advanced forward")
            return True
        except Exception as e:
            logger.error(f"Slide control error: {e}")
            return False
    
    def go_back_slide(self) -> bool:
        if not self.keyboard: return False
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval: return False
        try:
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)
            self.last_action_time = current_time
            logger.info("Slide moved backward")
            return True
        except Exception as e:
            logger.error(f"Slide control error: {e}")
            return False
    
    def jump_to_slide(self, current_slide: int, target_slide: int, max_attempts: int = 15) -> bool:
        if current_slide == target_slide: return True
        if not self.keyboard: return False
        steps_needed = target_slide - current_slide
        logger.info(f"Jumping from slide {current_slide} to {target_slide} ({steps_needed} steps)")
        steps_needed = max(-max_attempts, min(max_attempts, steps_needed))
        success_count = 0
        action = self.advance_slide if steps_needed > 0 else self.go_back_slide
        for _ in range(abs(steps_needed)):
            if action():
                success_count += 1
                time.sleep(0.1)
            else:
                break
        logger.info(f"Successfully executed {success_count} of {abs(steps_needed)} slide transitions")
        return success_count > 0

class MainWindow(QMainWindow):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Slide Switcher")
        self.setGeometry(100, 100, 1000, 700)
        
        self.audio_recorder = AudioRecorder()
        self.transcriber = SpeechTranscriber()
        self.screen_ocr = ScreenOCR()
        self.ai_agent = AISlideAgent()
        self.slide_controller = SlideController()
        
        self.slides = []
        self.is_recording = False
        self.settings_file = Path("slide_switcher_settings.json")
        
        self.setup_ui()
        self.load_settings()
        self.setup_connections()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ai_analysis)

    def load_settings(self):
        if not self.settings_file.exists():
            return
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            self.api_key_input.setText(settings.get('api_key', ''))
            
            model_index = self.model_combo.findText(settings.get('model', ''))
            if model_index >= 0: self.model_combo.setCurrentIndex(model_index)
            
            whisper_index = self.whisper_model_combo.findText(settings.get('whisper_model', 'base'))
            if whisper_index >= 0: self.whisper_model_combo.setCurrentIndex(whisper_index)

            self.confidence_slider.setValue(settings.get('confidence_threshold', 75))
            self.auto_advance_checkbox.setChecked(settings.get('auto_advance', True))
                    
            logger.info("Settings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def save_settings(self):
        try:
            settings = {
                'api_key': self.api_key_input.text().strip(),
                'model': self.model_combo.currentText(),
                'whisper_model': self.whisper_model_combo.currentText(),
                'confidence_threshold': self.confidence_slider.value(),
                'auto_advance': self.auto_advance_checkbox.isChecked()
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        tab_widget = QTabWidget()
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout(control_group)
        self.load_pdf_btn = QPushButton("Load PDF")
        self.start_btn = QPushButton("Start Assisting")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.load_pdf_btn, 0, 0)
        control_layout.addWidget(self.start_btn, 0, 1)
        control_layout.addWidget(self.stop_btn, 0, 2)
        
        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        self.slide_count_label = QLabel("Slides: 0")
        self.current_slide_label = QLabel("Current Slide: 1")
        self.confidence_label = QLabel("AI Confidence: 0%")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        status_layout.addWidget(self.slide_count_label, 0, 0)
        status_layout.addWidget(self.current_slide_label, 0, 1)
        status_layout.addWidget(self.confidence_label, 1, 0)
        status_layout.addWidget(self.confidence_bar, 1, 1)
        
        transcript_group = QGroupBox("Live Transcript")
        transcript_layout = QVBoxLayout(transcript_group)
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_display)
        
        ai_group = QGroupBox("AI Analysis")
        ai_layout = QVBoxLayout(ai_group)
        self.ai_decision_display = QTextEdit()
        self.ai_decision_display.setReadOnly(True)
        self.ai_decision_display.setMaximumHeight(100)
        ai_layout.addWidget(self.ai_decision_display)
        
        main_layout.addWidget(control_group)
        main_layout.addWidget(status_group)
        main_layout.addWidget(transcript_group)
        main_layout.addWidget(ai_group)
        
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        api_group = QGroupBox("API Configuration")
        api_layout = QGridLayout(api_group)
        api_layout.addWidget(QLabel("Gemini API Key:"), 0, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_key_input, 0, 1)
        
        api_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemini-2.5-flash-latest", "gemini-2.5-pro-latest"])
        api_layout.addWidget(self.model_combo, 1, 1)
        
        whisper_group = QGroupBox("Speech Recognition (Whisper)")
        whisper_layout = QGridLayout(whisper_group)
        whisper_layout.addWidget(QLabel("Whisper Model:"), 0, 0)
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.whisper_model_combo.setCurrentText("base")
        whisper_layout.addWidget(self.whisper_model_combo, 0, 1)

        sens_group = QGroupBox("Detection Settings")
        sens_layout = QGridLayout(sens_group)
        sens_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(75)
        sens_layout.addWidget(self.confidence_slider, 0, 1)
        self.auto_advance_checkbox = QCheckBox("Automatically advance slides when confident")
        self.auto_advance_checkbox.setChecked(True)
        sens_layout.addWidget(self.auto_advance_checkbox, 1, 0, 1, 2)
        
        settings_layout.addWidget(api_group)
        settings_layout.addWidget(whisper_group)
        settings_layout.addWidget(sens_group)
        settings_layout.addStretch()
        
        tab_widget.addTab(main_tab, "Main")
        tab_widget.addTab(settings_tab, "Settings")
        
        layout = QVBoxLayout(central_widget)
        layout.addWidget(tab_widget)
        
    def setup_connections(self):
        self.load_pdf_btn.clicked.connect(self.load_pdf)
        self.start_btn.clicked.connect(self.start_assistance)
        self.stop_btn.clicked.connect(self.stop_assistance)
        
        self.audio_recorder.audio_data.connect(self.transcriber.add_audio_data)
        self.transcriber.transcript_updated.connect(self.update_transcript)
        
        self.api_key_input.textChanged.connect(self.save_settings)
        self.model_combo.currentTextChanged.connect(self.save_settings)
        self.whisper_model_combo.currentTextChanged.connect(self.save_settings)
        self.confidence_slider.valueChanged.connect(self.save_settings)
        self.auto_advance_checkbox.toggled.connect(self.save_settings)
        
    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.slides = PDFProcessor.extract_slides(file_path)
            self.ai_agent.set_slides(self.slides)
            self.slide_count_label.setText(f"Slides: {len(self.slides)}")
            self.ai_agent.current_slide = 1
            self.current_slide_label.setText(f"Current Slide: {self.ai_agent.current_slide}")
            logger.info(f"Loaded {len(self.slides)} slides.")
    
    def start_assistance(self):
        self.transcriber.model_name = self.whisper_model_combo.currentText()
        if not self.transcriber.initialize_model():
            self.ai_decision_display.setText("ERROR: Could not initialize the Whisper speech recognition model.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.ai_decision_display.setText("ERROR: Gemini API key is missing. Please set it in the Settings tab.")
            return
            
        self.ai_agent.api_key = api_key
        self.ai_agent.model = self.model_combo.currentText()
        
        self.audio_recorder.start_recording()
        self.transcriber.start()
        
        self.update_timer.start(4000)
        
        self.is_recording = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.transcript_display.clear()
        self.ai_decision_display.setText("Assistant started. Listening for speech...")
        logger.info("Assistance started.")
    
    def stop_assistance(self):
        self.audio_recorder.stop_recording()
        self.transcriber.stop()
        self.update_timer.stop()
        
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.ai_decision_display.setText("Assistant stopped.")
        logger.info("Assistance stopped.")
    
    def update_transcript(self, text: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        timestamped_text = f"[{timestamp}] {text}"
        
        self.transcript_display.moveCursor(QTextCursor.End)
        self.transcript_display.insertPlainText(timestamped_text + "\n")
        self.transcript_display.ensureCursorVisible()
        self.ai_agent.add_transcript(text)
    
    def update_ai_analysis(self):
        if not self.is_recording or not self.ai_agent.transcript_history:
            return
        
        decision = self.ai_agent.analyze_transition()
        
        confidence_pct = int(decision.confidence * 100)
        self.confidence_label.setText(f"AI Confidence: {confidence_pct}%")
        self.confidence_bar.setValue(confidence_pct)
        
        self.ai_decision_display.setText(
            f"Transition: {decision.should_transition} | "
            f"Target: {decision.target_slide} | "
            f"Reason: {decision.reasoning}"
        )
        
        if (self.auto_advance_checkbox.isChecked() and 
            decision.should_transition and 
            decision.confidence >= (self.confidence_slider.value() / 100.0)):
            
            current_slide = self.ai_agent.current_slide
            target_slide = decision.target_slide
            
            if current_slide != target_slide:
                if self.slide_controller.jump_to_slide(current_slide, target_slide):
                    self.ai_agent.current_slide = target_slide
                    self.current_slide_label.setText(f"Current Slide: {target_slide}")
                    logger.info(f"AI triggered jump from slide {current_slide} to {target_slide}")
                else:
                    logger.warning(f"AI failed to jump from slide {current_slide} to {target_slide}")

    def closeEvent(self, event):
        self.stop_assistance()
        event.accept()

def check_dependencies():
    missing = []
    if not WHISPER_AVAILABLE: missing.append("openai-whisper")
    if not PDF_AVAILABLE: missing.append("PyMuPDF")
    if not PYNPUT_AVAILABLE: missing.append("pynput")
    if not OCR_AVAILABLE: missing.append("pytesseract, mss, Pillow")
    return missing

def main():
    app = QApplication(sys.argv)
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"ERROR: Missing required dependencies: {', '.join(missing_deps)}")
        print(f"Please install them via: pip install {' '.join(missing_deps)}")
        return 1
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()