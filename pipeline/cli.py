#!/usr/bin/env python3
"""
MESS-AI CLI for feature extraction and pipeline operations.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extraction import FeatureExtractor, pipeline_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_features(args):
    """Extract MERT features from audio files."""
    setup_logging(args.verbose)
    
    # Show configuration
    if args.show_config:
        pipeline_config.print_config()
        print("\nDevice Info:")
        import pprint
        pprint.pprint(pipeline_config.get_device_info())
        return
    
    # Initialize extractor
    extractor = FeatureExtractor(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Extract features
    if args.audio_file:
        # Single file
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        print(f"Extracting features from: {audio_path}")
        extractor.extract_track_features(
            audio_path,
            output_dir=args.output_dir,
            track_id=args.track_id
        )
        print(f"Features saved to: {args.output_dir or extractor.output_dir}")
        
    elif args.audio_dir:
        # Directory of files
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            sys.exit(1)
        
        print(f"Extracting features from directory: {audio_dir}")
        extractor.extract_dataset_features(
            audio_dir,
            output_dir=args.output_dir or extractor.output_dir,
            file_pattern=args.pattern
        )
        
    else:
        # Use default audio directory from config
        print(f"Extracting features from default directory: {pipeline_config.audio_dir}")
        extractor.extract_dataset_features(
            pipeline_config.audio_dir,
            output_dir=args.output_dir or pipeline_config.output_dir,
            file_pattern=args.pattern
        )


def validate_config(args):
    """Validate current configuration."""
    setup_logging(args.verbose)
    
    try:
        pipeline_config.validate_config()
        print("✅ Configuration is valid")
        
        if args.verbose:
            pipeline_config.print_config()
            print("\nDevice Info:")
            import pprint
            pprint.pprint(pipeline_config.get_device_info())
            print("\nPath Info:")
            pprint.pprint(pipeline_config.get_path_info())
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MESS-AI Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extract --show-config                    # Show current configuration
  %(prog)s extract --audio-file track.wav          # Extract from single file
  %(prog)s extract --audio-dir /path/to/audio      # Extract from directory
  %(prog)s extract                                  # Extract from default directory
  %(prog)s validate --verbose                      # Validate configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract MERT features')
    extract_parser.add_argument('--audio-file', type=str, help='Single audio file to process')
    extract_parser.add_argument('--audio-dir', type=str, help='Directory of audio files to process')
    extract_parser.add_argument('--output-dir', type=str, help='Output directory for features')
    extract_parser.add_argument('--track-id', type=str, help='Custom track ID for single file')
    extract_parser.add_argument('--pattern', type=str, default='*.wav', help='File pattern (default: *.wav)')
    extract_parser.add_argument('--model', type=str, help='Override MERT model name')
    extract_parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help='Override device')
    extract_parser.add_argument('--show-config', action='store_true', help='Show configuration and exit')
    extract_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    extract_parser.set_defaults(func=extract_features)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    validate_parser.set_defaults(func=validate_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()