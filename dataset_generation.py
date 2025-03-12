import os
import numpy as np
import json
import datetime
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from feature_extraction import AudioFeatureExtractor

class AudioModelDatasetGenerator:
    """Generate datasets for training audio models"""
    
    def __init__(self, dataset, output_dir='./model_datasets'):
        """
        Initialize the dataset generator
        
        Parameters:
        -----------
        dataset : SaarlandMusicDataset
            The music dataset object
        output_dir : str
            Directory to save generated datasets
        """
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create feature extractor
        self.feature_extractor = AudioFeatureExtractor()
    
    def generate_classification_dataset(self, label_fn, recording_ids=None, feature_type='mel_spectrogram', 
                                       segment_duration=3.0, overlap=0.5, min_segments=1, **feature_kwargs):
        """
        Generate a dataset for classification tasks
        
        Parameters:
        -----------
        label_fn : callable
            Function that takes a recording_id and returns a label
        recording_ids : list, optional
            List of recording IDs to include. If None, uses all recordings with 44kHz audio.
        feature_type : str
            Type of feature to extract ('mfcc', 'mel_spectrogram', etc.)
        segment_duration : float
            Duration of each audio segment in seconds
        overlap : float
            Overlap between segments (0.0-1.0)
        min_segments : int
            Minimum number of segments required to include a recording
        **feature_kwargs : dict
            Additional parameters to pass to the feature extraction method
            
        Returns:
        --------
        dataset_path : str
            Path to the generated dataset
        """
        if recording_ids is None:
            # Use all recordings with 44kHz audio
            recording_ids = [rec_id for rec_id, data in self.dataset.recordings.items() 
                           if data['wav-44']]
        
        # Prepare dataset containers
        features = []
        labels = []
        recording_map = []  # To track which recording each segment came from
        
        # Create timestamp for dataset
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset directory
        dataset_name = f"classification_{feature_type}_{timestamp}"
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Log file for tracking progress and issues
        log_file = os.path.join(dataset_dir, "generation_log.txt")
        
        with open(log_file, 'w') as log:
            log.write(f"Dataset generation started at {timestamp}\n")
            log.write(f"Feature type: {feature_type}\n")
            log.write(f"Segment duration: {segment_duration} seconds\n")
            log.write(f"Overlap: {overlap}\n")
            log.write(f"Minimum segments: {min_segments}\n")
            log.write(f"Feature kwargs: {feature_kwargs}\n\n")
            
            log.write(f"Processing {len(recording_ids)} recordings...\n\n")
            
            # Process each recording
            for rec_id in tqdm(recording_ids, desc="Generating dataset"):
                try:
                    # Get the label for this recording
                    label = label_fn(rec_id)
                    
                    # Skip if label is None (useful for filtering)
                    if label is None:
                        log.write(f"Skipping {rec_id}: Label is None\n")
                        continue
                    
                    # Load the recording
                    recording = self.dataset.get_recording(rec_id)
                    recording.load_audio(sample_rate='44')
                    
                    if recording.audio_44 is None:
                        log.write(f"Skipping {rec_id}: No 44kHz audio available\n")
                        continue
                    
                    audio = recording.audio_44
                    sr = recording.sr_44
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1 and audio.shape[1] > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Calculate segment parameters
                    samples_per_segment = int(segment_duration * sr)
                    hop_length = int(samples_per_segment * (1 - overlap))
                    
                    # Extract segments
                    segments = []
                    for start in range(0, len(audio) - samples_per_segment + 1, hop_length):
                        segment = audio[start:start + samples_per_segment]
                        segments.append(segment)
                    
                    if len(segments) < min_segments:
                        log.write(f"Skipping {rec_id}: Not enough segments ({len(segments)})\n")
                        continue
                    
                    # Extract features for each segment
                    for i, segment in enumerate(segments):
                        # Get the appropriate feature extraction method
                        if feature_type == 'mfcc':
                            feature = self.feature_extractor.extract_mfcc(segment, **feature_kwargs)
                        elif feature_type == 'mel_spectrogram':
                            feature = self.feature_extractor.extract_mel_spectrogram(segment, **feature_kwargs)
                        elif feature_type == 'chroma':
                            feature = self.feature_extractor.extract_chroma(segment, **feature_kwargs)
                        elif feature_type == 'spectral_contrast':
                            feature = self.feature_extractor.extract_spectral_contrast(segment, **feature_kwargs)
                        elif feature_type == 'tempogram':
                            feature = self.feature_extractor.extract_tempogram(segment, **feature_kwargs)
                        else:
                            raise ValueError(f"Unknown feature type: {feature_type}")
                        
                        features.append(feature)
                        labels.append(label)
                        recording_map.append(f"{rec_id}_segment_{i}")
                    
                    log.write(f"Processed {rec_id}: Added {len(segments)} segments with label '{label}'\n")
                    
                except Exception as e:
                    log.write(f"Error processing {rec_id}: {str(e)}\n")
        
        # Convert labels to numeric if they're strings
        if isinstance(labels[0], str):
            unique_labels = sorted(set(labels))
            label_to_index = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_index[label] for label in labels])
            
            # Save label mapping
            label_mapping = {i: label for label, i in label_to_index.items()}
            with open(os.path.join(dataset_dir, "label_mapping.json"), 'w') as f:
                json.dump(label_mapping, f, indent=2)
        else:
            numeric_labels = np.array(labels)
        
        # Convert to numpy arrays
        features = np.array(features)
        recording_map = np.array(recording_map)
        
        # Save the dataset
        np.save(os.path.join(dataset_dir, "features.npy"), features)
        np.save(os.path.join(dataset_dir, "labels.npy"), numeric_labels)
        np.save(os.path.join(dataset_dir, "recording_map.npy"), recording_map)
        
        # Save metadata
        metadata = {
            "dataset_type": "classification",
            "feature_type": feature_type,
            "segment_duration": segment_duration,
            "overlap": overlap,
            "sample_rate": 44100,
            "num_samples": len(features),
            "feature_shape": features[0].shape,
            "num_classes": len(set(numeric_labels)),
            "class_distribution": {int(label): int(np.sum(numeric_labels == label)) for label in set(numeric_labels)},
            "feature_kwargs": feature_kwargs,
            "creation_date": timestamp
        }
        
        with open(os.path.join(dataset_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate a visualization of the dataset
        self._visualize_dataset(features, numeric_labels, dataset_dir, feature_type)
        
        print(f"Dataset saved to {dataset_dir}")
        print(f"Total samples: {len(features)}")
        print(f"Feature shape: {features[0].shape}")
        print(f"Number of classes: {len(set(numeric_labels))}")
        
        return dataset_dir
    
    def generate_sequence_dataset(self, sequence_fn, recording_ids=None, feature_type='mel_spectrogram',
                                 segment_duration=3.0, hop_length=0.5, **feature_kwargs):
        """
        Generate a dataset for sequence prediction tasks
        
        Parameters:
        -----------
        sequence_fn : callable
            Function that takes a recording_id and returns a sequence of values
        recording_ids : list, optional
            List of recording IDs to include. If None, uses all recordings with 44kHz audio.
        feature_type : str
            Type of feature to extract ('mfcc', 'mel_spectrogram', etc.)
        segment_duration : float
            Duration of each audio segment in seconds
        hop_length : float
            Hop length between segments in seconds
        **feature_kwargs : dict
            Additional parameters to pass to the feature extraction method
            
        Returns:
        --------
        dataset_path : str
            Path to the generated dataset
        """
        if recording_ids is None:
            # Use all recordings with 44kHz audio
            recording_ids = [rec_id for rec_id, data in self.dataset.recordings.items() 
                           if data['wav-44']]
        
        # Prepare dataset containers
        features = []
        sequences = []
        recording_map = []  # To track which recording each segment came from
        
        # Create timestamp for dataset
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset directory
        dataset_name = f"sequence_{feature_type}_{timestamp}"
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Log file for tracking progress and issues
        log_file = os.path.join(dataset_dir, "generation_log.txt")
        
        with open(log_file, 'w') as log:
            log.write(f"Dataset generation started at {timestamp}\n")
            log.write(f"Feature type: {feature_type}\n")
            log.write(f"Segment duration: {segment_duration} seconds\n")
            log.write(f"Hop length: {hop_length} seconds\n")
            log.write(f"Feature kwargs: {feature_kwargs}\n\n")
            
            log.write(f"Processing {len(recording_ids)} recordings...\n\n")
            
            # Process each recording
            for rec_id in tqdm(recording_ids, desc="Generating dataset"):
                try:
                    # Get the sequence for this recording
                    sequence = sequence_fn(rec_id)
                    
                    # Skip if sequence is None (useful for filtering)
                    if sequence is None:
                        log.write(f"Skipping {rec_id}: Sequence is None\n")
                        continue
                    
                    # Load the recording
                    recording = self.dataset.get_recording(rec_id)
                    recording.load_audio(sample_rate='44')
                    
                    if recording.audio_44 is None:
                        log.write(f"Skipping {rec_id}: No 44kHz audio available\n")
                        continue
                    
                    audio = recording.audio_44
                    sr = recording.sr_44
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1 and audio.shape[1] > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Calculate segment parameters
                    samples_per_segment = int(segment_duration * sr)
                    hop_samples = int(hop_length * sr)
                    
                    # Extract segments
                    segments = []
                    for start in range(0, len(audio) - samples_per_segment + 1, hop_samples):
                        segment = audio[start:start + samples_per_segment]
                        segments.append(segment)
                    
                    # Extract features for each segment
                    for i, segment in enumerate(segments):
                        # Get the appropriate feature extraction method
                        if feature_type == 'mfcc':
                            feature = self.feature_extractor.extract_mfcc(segment, **feature_kwargs)
                        elif feature_type == 'mel_spectrogram':
                            feature = self.feature_extractor.extract_mel_spectrogram(segment, **feature_kwargs)
                        elif feature_type == 'chroma':
                            feature = self.feature_extractor.extract_chroma(segment, **feature_kwargs)
                        elif feature_type == 'spectral_contrast':
                            feature = self.feature_extractor.extract_spectral_contrast(segment, **feature_kwargs)
                        elif feature_type == 'tempogram':
                            feature = self.feature_extractor.extract_tempogram(segment, **feature_kwargs)
                        else:
                            raise ValueError(f"Unknown feature type: {feature_type}")
                        
                        features.append(feature)
                        sequences.append(sequence)
                        recording_map.append(f"{rec_id}_segment_{i}")
                    
                    log.write(f"Processed {rec_id}: Added {len(segments)} segments\n")
                    
                except Exception as e:
                    log.write("fix this")