import os
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import threading

class MusicPlayer:
    def __init__(self, wav_dir='data/smd/wav-44'):
        self.wav_dir = Path(wav_dir)
        self.current_file = None
        self.data = None
        self.sample_rate = 44100
        self.is_playing = False
        self._play_thread = None
        
    def list_files(self):
        """List all WAV files in the directory"""
        return list(self.wav_dir.glob('*.wav'))
    
    def load_file(self, file_path):
        """Load a WAV file"""
        self.current_file = file_path
        self.data, self.sample_rate = sf.read(file_path)
        return self.data, self.sample_rate
    
    def play(self):
        """Play the loaded audio file in a separate thread"""
        if self.data is not None and not self.is_playing:
            self._play_thread = threading.Thread(target=self._play_audio)
            self.is_playing = True
            self._play_thread.start()
        else:
            print("No audio file loaded or already playing")
    
    def _play_audio(self):
        """Internal method to play audio"""
        sd.play(self.data, self.sample_rate)
        sd.wait()
        self.is_playing = False
    
    def stop(self):
        """Stop the currently playing audio"""
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            if self._play_thread is not None:
                self._play_thread.join()
    
    def plot_waveform(self):
        """Plot the waveform of the loaded audio"""
        if self.data is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data)
            plt.title(f'Waveform: {self.current_file.name}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.show()
        else:
            print("No audio file loaded")

if __name__ == "__main__":
    # Create an instance of the MusicPlayer
    player = MusicPlayer()
    
    # List available WAV files
    wav_files = player.list_files()
    print("Available WAV files:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file.name}")
    
    # Example usage with play and stop:
    #player.load_file(wav_files[0])  # Load first file
    #player.play()  # Start playing
    
    # You can now stop the playback at any time with:
    # player.stop() 