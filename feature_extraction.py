import numpy as np
import librosa 
import warnings 
import os 
from scipy.io import wavfile
import matplotlib.pyplot as plt

class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def preprocess_audio(self, audio):
        #Convert to mono if stereo
        if len(audio.shape)>1 and audio.shape[1] > 1:
            audio = np.mean(audio,axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
        
        return audio
    
    def extract_mfcc(self, audio, n_mfcc=13, normalize=True):
        audio = self.preprocess_audio(audio)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        if normalize:
            mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        return mfccs
    
    def extract_mel_spectrogram(self, audio, n_mels=128, normalize=True):
        """Parameters: 
        audio : np.ndarray
            Audio signal
        n_mels : int
            Number of mel bands
        normalize : bool
            Normalize the mel spectrogram
        Returns:
            mel_spec_db : ndarray
                mel spec features of shape (n_mels, time)
        """
        audio = self.preprocess_audio(audio)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if normalize:
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db, axis=1, keepdims=True)) / (np.std(mel_spec_db, axis=1, keepdims=True) + 1e-8)
        return mel_spec_db
    
    def extract_chroma(self, audio, n_chroma=12, normalize=True):
        """
        Parameters: 
        audio : np.ndarray
            Audio signal
        n_chroma : int
            Number of chroma bands
        normalize : bool
            Normalize the chroma features
        Returns:
        chroma : ndarray
            chroma features of shape (n_chroma, time)
        """
        audio = self.preprocess_audio(audio)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_chroma=n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        if normalize:
            chroma = (chroma - np.mean(chroma, axis=1, keepdims=True)) / (np.std(chroma, axis=1, keepdims=True) + 1e-8)
        return chroma
    
    def extract_spectral_contrast(self, audio, n_bands=6, normalize=True):
        """Parameters: 
        audio : np.ndarray
            Audio signal
        n_bands : int
            Number of spectral bands
        normalize : bool
            Normalize the spectral contrast features
        
            Returns:
            conrast : ndarray
            spectral contrast features
        """
        audio = self.preprocess_audio(audio)
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_bands=n_bands,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        if normalize:
            contrast = (contrast - np.mean(contrast, axis=1, keepdims=True)) / (np.std(contrast, axis=1, keepdims=True) + 1e-8)
            
        return contrast
    
    def extract_onset_strength(self, audio, normalize=True):
        audio = self.preprocess_audio(audio)
        onset_strength = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        if normalize:
            onset_strength = (onset_strength - np.mean(onset_strength)) / (np.std(onset_strength) + 1e-8)
        return onset_strength
    
    def extract_tempogram(self, audio, win_length=384, normalize=True):
        audio = self.preprocess_audio(audio)
        onset_strength = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_strength,
            sr=self.sample_rate,
            win_length=win_length,
            hop_length=self.hop_length  
        )
        if normalize:
            tempogram = (tempogram - np.mean(tempogram, axis=1, keepdims=True)) / (np.std(tempogram, axis=1, keepdims=True) + 1e-8)
        return tempogram
    
    def extract_all_features(self, audio):
        """
        Parameters: 
        audio : np.ndarray
            Audio signal
        Returns:
            features : dict
                Dictionary of extracted features
        """
        audio = self.preprocess_audio(audio)

        return {
            'mfccs': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'onset_strength': self.extract_onset_strength(audio),
            'tempogram': self.extract_tempogram(audio)
        }
        
    def visualize_features(self, features, feature_type=None, figsize=(12,8)):
        """
        Visualize extracted features
        
        Parameters:
        -----------
        features : dict or ndarray
            Features to visualize
        feature_type : str, optional
            Type of feature if features is an ndarray
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if isinstance(features, dict):
            #Create a figure with subplots for each feature
            n_features = len(features)
            fig, axes = plt.subplots(n_features, 1, figsize=figsize)
            for i, (feat_name, feat_data) in enumerate(features.items()):
                if n_features == 1:
                    ax = axes
                else:
                    ax = axes[i]
                img = librosa.display.specshow(
                    feat_data, x_axis='time', ax=ax
                )
                ax.set_title(feat_name)
                fig.colorbar(img, ax=ax, format='%+2.0f dB' if 'mel_spectrogram' in feat_name else '%+2.0f')
        else:
            #Single feature
            fig, ax = plt.subplots(figsize=figsize)
            img = librosa.display.specshow(
                features, x_axis='time', ax=ax
            )
            ax.set_title(feature_type if feature_type else 'Feature')
            fig.colorbar(img, ax=ax, format='%+2.0f dB' if feature_type == 'mel_spectrogram' else '%+2.0f')
            
        plt.tight_layout()
        return fig
    
    
