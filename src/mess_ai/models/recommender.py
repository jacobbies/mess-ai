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
        