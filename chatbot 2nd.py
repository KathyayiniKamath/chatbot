import requests
import speech_recognition as sr
from gtts import gTTS
import pygame
from io import BytesIO
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
API_URL = "https://api-inference.huggingface.co/models/timpal0l/mdeberta-v3-base-squad2"
HEADERS = {"Authorization": "Bearer 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"}
def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        print("Recognition...")
        try:
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
            text_to_audio(text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print("Sorry, Google Speech Recognition service is not available.", e)

def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def load_context_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def answer_question(question, context):
    with open("questions.txt", "r") as file:
        with open("answers.txt", "r") as file1:
            lines = file1.readlines()
            questions = [line.strip() for line in file]
            answers = [line.strip() for line in lines]
    
    vectorizer = CountVectorizer().fit_transform([question] + questions)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    max_similarity_index = np.argmax(similarities)
    
    if similarities[max_similarity_index] > 0.5:
        return answers[max_similarity_index]
    else:
        response = query({
            "inputs": {
                "question": question,
                "context": context
            },
        })
        return response.get("answer", "I'm sorry, I don't understand that question.")

if __name__ == "__main__":
    context_file = "context.txt"  # Change this to the path of your context file
    context = load_context_from_file(context_file)
    
    while True:
        text = convert_speech_to_text()
        if text:
            response = answer_question(text, context)
            print("Response:", response)
            text_to_audio(response)
