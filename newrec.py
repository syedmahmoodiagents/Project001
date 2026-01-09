
import re
import soundfile as sf
import pyttsx3

from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama

AUDIO_FILE = "customer.wav"

# SYNTHETIC_CUSTOMER_TEXT = (
#     "Honestly, I like what you’re offering, "
#     "but the price feels too high for us right now."
# )


# def text_to_speech(text: str, output_file: str):
#     engine = pyttsx3.init()
#     engine.save_to_file(text, output_file)
#     engine.runAndWait()

# print("Generating customer audio...")
# text_to_speech(SYNTHETIC_CUSTOMER_TEXT, AUDIO_FILE)


class ASR:
    def __init__(self):
        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments)

asr = ASR()

print("Transcribing audio...")
raw_text = asr.transcribe(AUDIO_FILE)
print("RAW TRANSCRIPT:", raw_text)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(uh|um|you know|like)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_text = clean_text(raw_text)
print("CLEANED TEXT:", cleaned_text)


class IntentOutput(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]

parser = PydanticOutputParser(pydantic_object=IntentOutput)
# Intent Extractor

llm = ChatOllama(model="tinyllama", temperature=0)



intent_prompt = ChatPromptTemplate.from_template("""
You are an intent and sentiment classifier for sales calls.

Text:
{text}

Classify:
- intent (pricing_objection, interest, complaint, purchase_intent, other)
- sentiment (positive, neutral, negative)
- entities (keywords)

{format_instructions}
""")

intent_chain = intent_prompt | llm | parser

print("Extracting intent and sentiment...")

try:
    intent_result = intent_chain.invoke({
        "text": cleaned_text,
        "format_instructions": parser.get_format_instructions()
    })
    print("INTENT RESULT:", intent_result)
except Exception as e:
    print("Intent parsing failed:", e)
    # Attempt to get raw model response and parse it; fall back to defaults.
    try:
        raw_resp = (intent_prompt | llm).invoke({
            "text": cleaned_text,
            "format_instructions": parser.get_format_instructions()
        })
        print("RAW MODEL RESPONSE:", raw_resp)
        raw_text = None
        if hasattr(raw_resp, "content"):
            raw_text = raw_resp.content
        elif isinstance(raw_resp, str):
            raw_text = raw_resp
        else:
            raw_text = str(raw_resp)

        try:
            intent_result = parser.parse(raw_text)
            print("Parsed intent from raw response:", intent_result)
        except Exception as e2:
            print("Parser failed on raw response:", e2)
            intent_result = IntentOutput(intent="other", sentiment="neutral", entities=[])
    except Exception as e3:
        print("LLM invocation failed while recovering intent:", e3)
        intent_result = IntentOutput(intent="other", sentiment="neutral", entities=[])


# Decision Logic (Rules)

def decide_action(intent_data: IntentOutput) -> str:
    if intent_data.intent == "pricing_objection":
        if intent_data.sentiment == "negative":
            return "Empathize with concern, then explain ROI before discount"
        return "Explain pricing structure clearly"

    if intent_data.intent == "complaint":
        return "Acknowledge issue and ask clarifying question"

    if intent_data.intent == "purchase_intent":
        return "Move to close and discuss onboarding"

    return "Provide general clarification"

action = decide_action(intent_result)
print("DECISION:", action)


recommendation_prompt = ChatPromptTemplate.from_template("""
You are a sales coach.

Based on:
Intent: {intent}
Sentiment: {sentiment}
Entities: {entities}

Give ONE short recommendation for the sales agent.
""")

recommendation_chain = recommendation_prompt | llm

recommendation = recommendation_chain.invoke({
    "intent": intent_result.intent,
    "sentiment": intent_result.sentiment,
    "entities": ", ".join(intent_result.entities)
})



print("Customer said:", raw_text)
print("Detected intent:", intent_result.intent)
print("Detected sentiment:", intent_result.sentiment)
print("Sales action:", action)
print("Sales recommendation:", recommendation.content)

