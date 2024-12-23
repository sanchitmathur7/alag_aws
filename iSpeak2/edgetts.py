from flask import Blueprint, render_template, request, jsonify, send_file
import edge_tts
import asyncio
import tempfile
import os
import zipfile
from werkzeug.utils import secure_filename
from docx import Document
import csv
from datetime import datetime
import socket
import pandas as pd
import pytz
from pytz import timezone

edgetts_bp = Blueprint('edgetts', __name__)

# Path to the analytics file
ANALYTICS_FILE = "analytics.csv"

# Initialize the analytics file
if not os.path.exists(ANALYTICS_FILE):
   with open(ANALYTICS_FILE, mode='w', newline='') as csvfile:
       fieldnames = [
           "Date", "Time", "User_IP", "Action", "Num_Audios", 
           "Total_Audio_Duration", "Notes"
       ]
       writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       writer.writeheader()

async def get_voices():
   voices = await edge_tts.list_voices()
   return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in voices}

async def text_to_speech(text, voice, rate, pitch):
   voice_short_name = voice.split(" - ")[0]
   rate_str = f"{rate:+d}%"
   pitch_str = f"{pitch:+d}Hz"
   communicate = edge_tts.Communicate(text, voice_short_name, rate=rate_str, pitch=pitch_str)

   with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
       tmp_path = tmp_file.name
       await communicate.save(tmp_path)

   return tmp_path

def apply_replacements(text, csv_path='replacement_words.csv'):
   try:
       if os.path.exists(csv_path):
           with open(csv_path, mode='r') as csvfile:
               reader = csv.DictReader(csvfile)
               for row in reader:
                   original_text = row.get('original_text', '').strip()
                   replacement_text = row.get('replacement_text', '').strip()
                   if original_text and replacement_text:
                       text = text.replace(original_text, replacement_text)
   except Exception as e:
       print(f"Error applying replacements: {e}")

   return text

def log_analytics(action, num_audios, total_duration, notes=""):
       # Get the current time in UTC
   utc_now = datetime.now(pytz.utc)

   # Convert UTC to IST
   ist = timezone('Asia/Kolkata')
   ist_now = utc_now.astimezone(ist)

   # Format the IST date and time
   date = ist_now.strftime("%Y-%m-%d")
   time = ist_now.strftime("%H:%M:%S")

   # Get the user's IP address
   user_ip = request.remote_addr

   # Append the data to the analytics file
   with open(ANALYTICS_FILE, mode='a', newline='') as csvfile:
       writer = csv.DictWriter(csvfile, fieldnames=[
           "Date", "Time", "User_IP", "Action", "Num_Audios", 
           "Total_Audio_Duration", "Notes"
       ])
       writer.writerow({
           "Date": date,
           "Time": time,
           "User_IP": user_ip,
           "Action": action,
           "Num_Audios": num_audios,
           "Total_Audio_Duration": total_duration,
           "Notes": notes
       })

@edgetts_bp.route('/edgetts', methods=['GET'])
def edgetts_interface():
   voices = asyncio.run(get_voices())
   return render_template('edgetts.html', voices=voices)

@edgetts_bp.route('/convert', methods=['POST'])
def convert_text():
   text = request.form.get('text')
   voice = request.form.get('voice')
   rate = int(request.form.get('rate', 0))
   pitch = int(request.form.get('pitch', 0))

   # Apply replacements
   text = apply_replacements(text)

   audio_path = asyncio.run(text_to_speech(text, voice, rate, pitch))

   # Log analytics for single audio
   if audio_path:
       # Assuming an average of 150 words per minute for audio duration estimation
       words = len(text.split())
       duration = words / 150
       log_analytics(action="Single Audio", num_audios=1, total_duration=duration)

       return send_file(audio_path, mimetype="audio/mpeg")
   return jsonify({"error": "Conversion failed."}), 500

@edgetts_bp.route('/bulk_convert', methods=['POST'])
def bulk_convert():
   file = request.files.get('docx_file')
   voice = request.form.get('voice')
   rate = int(request.form.get('rate', 0))
   pitch = int(request.form.get('pitch', 0))

   if not file or not file.filename.endswith('.docx'):
       return jsonify({"error": "Please upload a valid .docx file"}), 400

   # Create a temporary directory to save audio files
   temp_dir = tempfile.mkdtemp()

   # Save the uploaded file to the temporary directory
   docx_path = os.path.join(temp_dir, secure_filename(file.filename))
   file.save(docx_path)

   # Process the .docx file and generate audios for each paragraph
   document = Document(docx_path)
   audio_files = []
   total_words = 0

   for i, paragraph in enumerate(document.paragraphs):
       if paragraph.text.strip():  # Skip empty paragraphs
           # Apply replacements
           paragraph_text = apply_replacements(paragraph.text)
           total_words += len(paragraph_text.split())
           audio_path = asyncio.run(text_to_speech(paragraph_text, voice, rate, pitch))
           new_audio_path = os.path.join(temp_dir, f"slide_{i + 1}.mp3")
           os.rename(audio_path, new_audio_path)
           audio_files.append(new_audio_path)

   # Create a zip file containing all audio files
   zip_path = os.path.join(temp_dir, "bulk_audio.zip")
   with zipfile.ZipFile(zip_path, 'w') as zipf:
       for audio_file in audio_files:
           zipf.write(audio_file, os.path.basename(audio_file))
           os.remove(audio_file)  # Cleanup each audio file after adding to zip

   # Log analytics for bulk conversion
   total_duration = total_words / 150  # Estimate based on average speaking rate
   log_analytics(action="Bulk Audio", num_audios=len(audio_files), total_duration=total_duration)


   return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name="bulk_audio.zip")
   

@edgetts_bp.route('/analytics_data', methods=['GET'])
def get_analytics_data():
   if os.path.exists(ANALYTICS_FILE):
       # Read CSV file
       df = pd.read_csv(ANALYTICS_FILE)
       
       # Convert the DataFrame to a dictionary
       data = df.to_dict(orient='records')
       return jsonify(data)
   else:
       return jsonify({"error": "Analytics file not found."}), 404

@edgetts_bp.route('/analytics', methods=['GET'])
def analytics_page():
   if os.path.exists(ANALYTICS_FILE):
       df = pd.read_csv(ANALYTICS_FILE)
       data = df.to_dict(orient='records')
       return render_template('analytics.html', data=data)
   else:
       return render_template('analytics.html', data=[])

@edgetts_bp.route('/download_analytics', methods=['GET'])
def download_analytics():
   if os.path.exists(ANALYTICS_FILE):
       return send_file(
           ANALYTICS_FILE,
           mimetype='text/csv',
           as_attachment=True,
           download_name='analytics.csv'
       )
   else:
       return jsonify({"error": "Analytics file not found."}), 404