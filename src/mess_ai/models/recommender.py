# src/mess_ai/models/recommender.py
import random
import torch
import numpy as np
from pathlib import Path
import os

class MusicRecommender:
    def __init__(self, feature_extractor=None, embedding_dim=128):
        self.feature_extractor = feature_extractor
        self.embedding_dim = embedding_dim
        self.learning_rate = 0.05
        self.alpha = 0.6  # Balance between user and session influence
        
        # Initialize song library (empty at first)
        self.song_library = {}  # filename -> embedding
        
        # Initialize user and session vectors
        self.theta_u = torch.zeros(embedding_dim)  # long-term taste
        self.s_t = torch.zeros(embedding_dim)  # session-level preference
        
    def load_or_create_embeddings(self, audio_dir):
        """Load or create embeddings for all audio files in directory"""
        audio_dir = Path(audio_dir)
        
        # Check if we have a feature extractor
        if self.feature_extractor:
            print("Extracting real audio features...")
            self.song_library = self.feature_extractor.build_embeddings(audio_dir)
        else:
            # Fallback to random vectors
            print("No feature extractor, using random embeddings...")
            for audio_file in audio_dir.glob('*.wav'):
                if audio_file.name not in self.song_library:
                    self.song_library[audio_file.name] = torch.randn(self.embedding_dim)
        
        print(f"Loaded embeddings for {len(self.song_library)} songs")
    
    def update_vectors(self, filename, feedback):
        """Update user and session vectors based on user feedback
        
        Args:
            filename (str): Filename of the song
            feedback (str): Either "play" or "skip"
        """
        if filename not in self.song_library:
            print(f"Warning: {filename} not in library")
            return
            
        phi_i = self.song_library[filename]
        sign = 1.0 if feedback == "play" else -1.0
        self.theta_u += sign * self.learning_rate * phi_i  
        self.s_t += sign * self.learning_rate * phi_i
        
    def find_similar_tracks(self, file_path, n=3):
        """Find similar tracks based on song embeddings and user/session vectors
        
        Args:
            file_path (str): Path to the query audio file
            n (int): Number of recommendations to return
            
        Returns:
            list: List of tuples (file_path, similarity_score)
        """
        query_path = Path(file_path)
        query_filename = query_path.name
        
        # Load embeddings if library is empty
        if not self.song_library:
            self.load_or_create_embeddings(query_path.parent)
        
        # If we've never seen this file, create an embedding
        if query_filename not in self.song_library:
            self.song_library[query_filename] = torch.randn(self.embedding_dim)
        
        # Update vectors based on current play
        self.update_vectors(query_filename, "play")
        
        # Calculate scores
        scores = {}
        for filename, phi_i in self.song_library.items():
            if filename == query_filename:  # Skip the query file
                continue
                
            score = self.alpha * torch.dot(self.theta_u, phi_i) + \
                   (1 - self.alpha) * torch.dot(self.s_t, phi_i)
            scores[filename] = score.item()
        
        # Sort and return top n
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Convert to expected format (file_path, score)
        query_dir = query_path.parent
        recommendations = [(str(query_dir / filename), score) for filename, score in sorted_recs]
        
        return recommendations

    def record_skip(self, filename):
        """Record a song skip to adjust recommendations"""
        self.update_vectors(filename, "skip")
        
    def reset_session(self):
        """Reset session vector for a new listening session"""
        self.s_t = torch.zeros(self.embedding_dim)

    def decay_session(self, decay_factor=0.95):
        """Apply decay to session vector over time"""
        self.s_t *= decay_factor

    def add_prompt(self, prompt_name, weight=0.5):
        """Blend a prompt vector into the session vector"""
        if prompt_name in self.prompt_vectors:
            self.s_t = weight * self.prompt_vectors[prompt_name] + (1-weight) * self.s_t

    def save_user_preferences(self, user_id):
        """Save user preference vector to disk"""
        torch.save(self.theta_u, f"user_prefs/{user_id}.pt")
        
    def load_user_preferences(self, user_id):
        """Load user preference vector from disk"""
        path = f"user_prefs/{user_id}.pt"
        if os.path.exists(path):
            self.theta_u = torch.load(path)