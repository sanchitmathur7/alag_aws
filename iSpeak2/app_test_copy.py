import os

# Set cache paths
os.environ['HF_HOME'] = '/slowfs/amstpubs1/sanchit/iSpeak/iSpeak_Final/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/slowfs/amstpubs1/sanchit/iSpeak/iSpeak_Final/cache/'

import csv
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from gtts import gTTS
import torch
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from transformers import AutoProcessor, BarkModel
import scipy
from nltk.tokenize import sent_tokenize
from docx import Document
from werkzeug.utils import secure_filename
import random
import json
from model import NeuralNet as TTSNeuralNet
from nltk_utils import bag_of_words, tokenize
from flask_cors import CORS
import inflect
import edge_tts  # Import Edge TTS

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
CORS(app)

# Create an inflect engine
p = inflect.engine()

def convert_numbers_to_words(text):
   tokens = text.split()
   converted_tokens = [p.number_to_words(token) if token.isdigit() else token for token in tokens]
   return ' '.join(converted_tokens)

# Device configuration:
cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Tacotron2 model (using CPU)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts").to(cpu_device)

# Load pre-trained HiFIGAN vocoder model (using CPU)
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder").to(cpu_device)

# Load Bark processor and model (using GPU)
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(gpu_device)

# Path to the CSV files
csv_file = 'replacement.csv'
users_file = 'users.csv'
replacement_df = pd.read_csv("replacement_words.csv")

# Paths for the new feature
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
CSV_FILE = 'replacements_for_transcript.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['CSV_FILE'] = CSV_FILE

# Ensure the folders exist
if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
   os.makedirs(PROCESSED_FOLDER)

# Function to update the CSV file with new values
def update_csv(original_text, replacement_text):
   with open(csv_file, 'a', newline='') as file:
       writer = csv.writer(file)
       writer.writerow([original_text, replacement_text])

# Function to check if user is logged in
def login_required(f):
   def wrap(*args, **kwargs):
       if 'logged_in' in session:
           return f(*args, **kwargs)
       else:
           return redirect(url_for('login'))
   wrap.__name__ = f.__name__
   return wrap

@app.route('/')
@login_required
def index():
   return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
   if request.method == 'POST':
       username = request.form['username']
       password = request.form['password']
       
       with open(users_file, 'r') as file:
           reader = csv.DictReader(file)
           for row in reader:
               if row['username'] == username and row['password'] == password:
                   session['logged_in'] = True
                   return redirect(url_for('index'))
       return "Login failed. Check your username and password."
   return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
   session.pop('logged_in', None)
   return redirect(url_for('login'))

@app.route('/generate_audio', methods=['POST'])
@login_required
def generate_audio():
   text = request.form['text']
   voice = request.form['voice']
   preset = request.form.get('preset')

   # Replace words in the text with their replacements based on the CSV file
   for index, row in replacement_df.iterrows():
       text = text.replace(row['original_text'], row['replacement_text'])

   if voice == 'tacotron2':
       audio_path = generate_audio_tacotron2(text)
   elif voice == 'gtts':
       audio_path = generate_audio_gtts(text)
   elif voice == 'bark_suno':
       audio_path = generate_audio_bark(text, preset)
   elif voice == 'edge_tts':
       audio_path = generate_audio_edge_tts(text)
   return render_template('preview.html', audio_path=audio_path)

@app.route('/download_audio', methods=['POST'])
@login_required
def download_audio():
   audio_path = request.form['audio_path']
   return send_file(audio_path, as_attachment=True)

def generate_unique_filename(extension):
   return f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

def generate_audio_tacotron2(text):
   text = convert_numbers_to_words(text)

   with torch.no_grad():
       mel_output, _, _ = tacotron2.encode_text(text)
       mel_output = mel_output.cpu()

   with torch.no_grad():
       waveforms = hifi_gan.decode_batch(mel_output)
       waveforms = waveforms.squeeze(1).cpu().numpy()

   audio_path = f"static/{generate_unique_filename('wav')}"
   torchaudio.save(audio_path, torch.tensor(waveforms), 22050)
   return audio_path

def generate_audio_gtts(text):
   tts = gTTS(text=text, lang='en')
   audio_path = f"static{generate_unique_filename('mp3')}"
   tts.save(audio_path)
   return audio_path

def generate_audio_bark(text, preset):
   sentences = sent_tokenize(text)
   audio_list = []

   for sentence in sentences:
       inputs = processor(sentence, voice_preset=preset)
       for k, v in inputs.items():
           inputs[k] = v.to(gpu_device)
       audio_array = model.generate(**inputs)
       audio_array = audio_array.cpu().numpy().squeeze()
       audio_list.append(audio_array)

       silence_duration = int(model.generation_config.sample_rate * 0.25)
       silence = np.zeros(silence_duration)
       audio_list.append(silence)

   concatenated_audio = np.concatenate(audio_list)
   stereo_audio = np.stack([concatenated_audio, concatenated_audio], axis=1)

   audio_path = f"static/{generate_unique_filename('wav')}"
   if not os.path.exists('static'):
       os.makedirs('static')
   scipy.io.wavfile.write(audio_path, rate=model.generation_config.sample_rate, data=stereo_audio)
   return audio_path

def generate_audio_edge_tts(text):
   audio_path = f"static/{generate_unique_filename('wav')}"
   
   # Define the voice parameters as needed
   voice_params = {
       "voice": "en-US-JennyNeural",  # Adjust the voice to your preference
       "rate": "0%",
       "pitch": "0%",
       "volume": "1.0"
   }
   
   # Generate audio using Edge TTS
   async def edge_tts_synthesis():
       communicate = edge_tts.Communicate(text, **voice_params)
       await communicate.save(audio_path)

   # Run the asynchronous function to save audio
   import asyncio
   asyncio.run(edge_tts_synthesis())
   
   return audio_path

@app.route('/form')
@login_required
def form():
   return render_template('form.html')

@app.route('/submit_form', methods=['POST'])
@login_required
def submit_form():
   original_text = request.form['original_text']
   replacement_text = request.form['replacement_text']
   update_csv(original_text, replacement_text)
   return redirect(url_for('index'))

@app.route('/bulk_audio', methods=['GET', 'POST'])
@login_required
def bulk_audio():
   if request.method == 'POST':
       doc_file = request.files['doc_file']
       voice = request.form['voice']
       preset = request.form.get('preset')

       # Save the uploaded DOC file
       doc_path = f"uploads/{doc_file.filename}"
       doc_file.save(doc_path)

       # Extract paragraphs from the DOC file
       paragraphs = extract_paragraphs_from_doc(doc_path)

       # Generate audio for each paragraph
       audio_files = []
       for i, paragraph in enumerate(paragraphs):
           if voice == 'tacotron2':
               audio_path = generate_audio_tacotron2(paragraph)
           elif voice == 'gtts':
               audio_path = generate_audio_gtts(paragraph)
           elif voice == 'bark_suno':
               audio_path = generate_audio_bark(paragraph, preset)
           elif voice == 'edge_tts':
               audio_path = generate_audio_edge_tts(paragraph)
           audio_files.append(audio_path)

       # Create a zip file containing all audio files
       zip_path = f"processed/{generate_unique_filename('zip')}"
       with zipfile.ZipFile(zip_path, 'w') as zipf:
           for audio_file in audio_files:
               zipf.write(audio_file, os.path.basename(audio_file))

       return send_file(zip_path, as_attachment=True)

   return render_template('bulk_audio.html')

def extract_paragraphs_from_doc(doc_path):
   doc = Document(doc_path)
   return [para.text for para in doc.paragraphs if para.text.strip()]


if __name__ == '__main__':
   if not os.path.exists('uploads'):
       os.makedirs('uploads')
   if not os.path.exists('processed'):
       os.makedirs('processed')
   app.run(host='0.0.0.0', port=2000, debug=False)