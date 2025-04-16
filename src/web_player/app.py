import io
import sys
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_from_directory, Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mess_ai.audio.player import MusicPlayer

app = Flask(__name__)
player = MusicPlayer(wav_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/smd/wav-44')))

@app.route('/')
def index():
    wav_files = player.list_files()
    return render_template('index.html', wav_files=[f.name for f in wav_files])

@app.route('/audio/<filename>')
def audio(filename):
    # Serve the audio file to the browser
    return send_from_directory(player.wav_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')