from faster_whisper import WhisperModel
import sounddevice as sd
import queue
import numpy as np
from pydub import AudioSegment
import os
from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

AUDIO_FILE = "audio_sample.mp3"

# model = WhisperModel(model_size_or_path='small', device='cpu', compute_type='int8')

# segments, info = model.transcribe(AUDIO_FILE, beam_size=1, vad_filter=True)

# allsegs = " ".join(seg.text for seg in segments)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    callback=audio_callback
)

stream.start()

BUFFER_SECONDS = 3
SAMPLE_RATE = 16000
buffer = np.zeros((0, 1))

def get_audio_chunk():
    global buffer
    while not audio_queue.empty():
        buffer = np.vstack([buffer, audio_queue.get()])

    if len(buffer) >= BUFFER_SECONDS * SAMPLE_RATE:
        chunk = buffer[:BUFFER_SECONDS * SAMPLE_RATE]
        buffer = buffer[BUFFER_SECONDS * SAMPLE_RATE:]
        return chunk
    return None

from faster_whisper import WhisperModel
import tempfile
import soundfile as sf

model = WhisperModel("small", compute_type="int8")

AudioSegment.from_file(AUDIO_FILE).export("audio_sample.wav", format="wav")

def transcribe_chunk(audio_chunk):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio_chunk, 16000)
        segments, _ = model.transcribe(f.name)
        return " ".join(seg.text for seg in segments)


class LiveIntent(BaseModel):
    intent: str
    sentiment: str
    # entities: List[str]

prompt = ChatPromptTemplate.from_template("""
You are an intent and sentiment extraction engine.

text : {text}

You need to extract the intent, sentiment and entities from the text.
Extract:
    -intent (greeting, farewell, inquiry, complaint, suggestion)
    -sentiment (good, bad, neutral)

{format_instructions}

""")

llm = ChatOllama(model="tinyllama", temperature=0) 

parser = PydanticOutputParser(pydantic_object=LiveIntent)

chain = prompt | llm | parser

chain.invoke({
    'text': allsegs,
    'format_instructions': parser.get_format_instructions()
})


def clean_text(text):
    return text.lower().replace("uh", "").replace("um", "").strip()

def decide_live_action(intent):
    if intent.intent == "pricing_objection":
        return "Explain ROI before discount"
    if intent.intent == "complaint":
        return "Acknowledge + empathize"
    if intent.intent == "close":
        return "Ask for confirmation"
    return None

def show_recommendation(text):
    print("SALES TIP:", text)


while True:
    chunk = get_audio_chunk()
    if chunk is None:
        continue

    transcript = transcribe_chunk(chunk)
    if not transcript.strip():
        continue

    cleaned = clean_text(transcript)

    intent = intent_chain.invoke({
        "text": cleaned,
        "format_instructions": parser.get_format_instructions()
    })

    recommendation = decide_live_action(intent)
    if recommendation:
        show_recommendation(recommendation)
