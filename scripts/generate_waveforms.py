#!/usr/bin/env python3
"""
Generate static waveform images for all audio files in the dataset.

This script pre-generates waveform visualizations for all tracks,
saving them as PNG files for serving as static assets.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_waveform(audio_path: Path, output_path: Path, 
                     figsize=(12, 3), color='#00d4ff', 
                     bg_color='#1a1a1a', max_points=10000):
    """
    Generate a waveform visualization for an audio file.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save PNG waveform
        figsize: Figure size in inches
        color: Waveform color
        bg_color: Background color
        max_points: Maximum points to plot (for performance)
    """
    try:
        # Load audio
        audio, sr = sf.read(str(audio_path))
        
        # Handle stereo by converting to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Create time axis
        time_seconds = np.arange(len(audio)) / sr
        
        # Downsample for performance
        downsample_factor = max(1, len(audio) // max_points)
        audio_down = audio[::downsample_factor]
        time_down = time_seconds[::downsample_factor]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Plot waveform
        ax.plot(time_down, audio_down, color=color, linewidth=0.5, alpha=0.8)
        ax.fill_between(time_down, audio_down, alpha=0.3, color=color)
        
        # Styling
        ax.set_xlim(0, time_seconds[-1])
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
        ax.set_ylabel('Amplitude', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.2, color='white')
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        
        plt.tight_layout()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), facecolor=bg_color, dpi=100)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating waveform for {audio_path}: {e}")
        return False


def main():
    """Generate waveforms for all audio files."""
    # Paths
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "data" / "smd" / "wav-44"
    output_dir = project_root / "data" / "processed" / "waveforms"
    
    logger.info(f"Generating waveforms from {audio_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Generate waveforms
    success_count = 0
    for audio_path in tqdm(audio_files, desc="Generating waveforms"):
        output_path = output_dir / f"{audio_path.stem}.png"
        
        # Skip if already exists
        if output_path.exists():
            logger.debug(f"Skipping {audio_path.name} - waveform already exists")
            success_count += 1
            continue
        
        if generate_waveform(audio_path, output_path):
            success_count += 1
    
    logger.info(f"Successfully generated {success_count}/{len(audio_files)} waveforms")
    logger.info(f"Waveforms saved to {output_dir}")
    
    # Print example nginx/S3 configuration
    print("\n" + "="*60)
    print("DEPLOYMENT NOTES:")
    print("="*60)
    print("\n1. For local development:")
    print(f"   Waveforms are served directly from: {output_dir}/")
    print("\n2. For AWS S3 deployment:")
    print(f"   aws s3 sync {output_dir}/ s3://your-bucket/waveforms/ \\")
    print("     --cache-control 'max-age=31536000,public,immutable'")
    print("\n3. Set environment variables for production:")
    print("   export WAVEFORM_BASE_URL='https://your-cdn.cloudfront.net/waveforms/'")
    print("   export AUDIO_BASE_URL='https://your-cdn.cloudfront.net/audio/'")
    print("="*60)


if __name__ == "__main__":
    main()