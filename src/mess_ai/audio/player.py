import io
import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MusicLibrary:
    def __init__(self, wav_dir='data/smd/wav-44'):
        self.wav_dir = Path(wav_dir)
        self.current_file = None #path
        self.data = None
        self.sample_rate = 44100
        self.is_playing = False
        
    def list_files(self):
        """List all WAV files in the directory"""
        return list(self.wav_dir.glob('*.wav'))
    
    def load_file(self, file_path):
        """Load a WAV file"""
        self.current_file = Path(file_path)
        self.data, self.sample_rate = sf.read(file_path)
        return self.data, self.sample_rate
    
    def plot_waveform(self, save_to_buffer=False):
        """Plot the waveform of the loaded audio"""
        if self.data is not None:

            duration = len(self.data) / self.sample_rate
            time = np.linspace(0, duration, len(self.data))
            plt.figure(figsize=(10, 4))
            plt.plot(time,self.data)
            plt.title(f'Waveform: {self.current_file.name if self.current_file else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            if save_to_buffer:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                return buf
            else:
                plt.show()
                return None
        else:
            print("No audio file loaded")
            return None

if __name__ == "__main__":
    library = MusicLibrary()
    wav_files = library.list_files()
    print("Available WAV files:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file.name}")