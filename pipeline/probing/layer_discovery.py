"""
MERT Layer Discovery System
Systematically discover what each MERT layer encodes for music similarity.
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.append('/Users/jacobbieschke/mess-ai/pipeline')

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerDiscoverySystem:
    """Discover what musical aspects each MERT layer encodes."""
    
    def __init__(self):
        self.data_dir = Path("/Users/jacobbieschke/mess-ai/data")
        self.features_dir = self.data_dir / "processed" / "features" / "raw"
        self.targets_dir = self.data_dir / "processed" / "proxy_targets"
        
        # Validated layer mappings (from our experiments)
        self.validated_layers = {
            1: {
                'aspect': 'timbral_similarity',
                'r2_score': 0.92,
                'target': 'spectral_centroid',
                'description': 'Instrumental timbre and spectral characteristics',
                'confidence': 'high'
            }
        }
        
        # Aspects to test systematically
        self.testable_aspects = {
            'brightness': 'spectral_centroid',
            'tempo_stability': 'tempo',
            'dynamic_contrast': 'dynamic_range',
            'harmonic_richness': 'harmonic_complexity',
            'attack_sharpness': 'onset_slopes',
            'rhythmic_density': 'onset_density'
        }
    
    def load_mert_features(self, audio_files: List[str]) -> Dict[int, np.ndarray]:
        """Load MERT features for all layers."""
        all_features = {layer: [] for layer in range(13)}
        successful_files = []
        
        for audio_file in audio_files:
            audio_name = Path(audio_file).stem
            feature_file = self.features_dir / f"{audio_name}.npy"
            
            if feature_file.exists():
                raw_features = np.load(feature_file)
                
                # Extract each layer and average across segments and time
                for layer in range(13):
                    layer_features = raw_features[:, layer, :, :].mean(axis=(0, 1))
                    all_features[layer].append(layer_features)
                
                successful_files.append(audio_file)
        
        # Convert to numpy arrays
        for layer in range(13):
            all_features[layer] = np.array(all_features[layer])
        
        logger.info(f"Loaded features for {len(successful_files)} files")
        return all_features, successful_files
    
    def load_targets(self, audio_files: List[str]) -> Dict[str, np.ndarray]:
        """Load proxy targets for testing."""
        targets = {
            'spectral_centroid': [],
            'tempo': [],
            'dynamic_range': [],
            'harmonic_complexity': []
        }
        successful_files = []
        
        for audio_file in audio_files:
            audio_name = Path(audio_file).stem
            target_file = self.targets_dir / f"{audio_name}_targets.npz"
            
            if target_file.exists():
                data = np.load(target_file, allow_pickle=True)
                
                # Extract scalar targets
                rhythm_data = data['rhythm'].item()
                timbre_data = data['timbre'].item()
                dynamics_data = data['dynamics'].item()
                harmony_data = data['harmony'].item()
                
                targets['spectral_centroid'].append(np.mean(timbre_data['spectral_centroid']))
                targets['tempo'].append(rhythm_data['tempo'][0])
                targets['dynamic_range'].append(dynamics_data['dynamic_range'][0])
                targets['harmonic_complexity'].append(harmony_data['harmonic_complexity'][0])
                
                successful_files.append(audio_file)
        
        # Convert to numpy arrays
        for aspect in targets:
            targets[aspect] = np.array(targets[aspect])
        
        return targets, successful_files
    
    def probe_layer_for_aspect(self, layer_features: np.ndarray, target: np.ndarray, 
                              aspect_name: str, layer_num: int) -> Dict:
        """Probe a specific layer for a musical aspect."""
        
        if len(layer_features) < 5:
            return {'r2_mean': -999, 'r2_std': 0, 'status': 'insufficient_data'}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(layer_features)
        
        # Cross-validated ridge regression
        probe = Ridge(alpha=1.0, random_state=42)
        
        try:
            cv_scores = cross_val_score(
                probe, features_scaled, target,
                cv=min(5, len(layer_features) - 1),
                scoring='r2'
            )
            
            return {
                'r2_mean': np.mean(cv_scores),
                'r2_std': np.std(cv_scores),
                'cv_scores': cv_scores.tolist(),
                'status': 'success'
            }
        except Exception as e:
            return {'r2_mean': -999, 'r2_std': 0, 'status': f'error: {e}'}
    
    def discover_layer_functions(self, n_samples: int = 30) -> Dict:
        """Systematically discover what each layer encodes."""
        
        # Get audio files
        audio_dir = Path("/Users/jacobbieschke/mess-ai/data/smd/wav-44")
        audio_files = sorted([str(f) for f in audio_dir.glob("*.wav")])[:n_samples]
        
        logger.info(f"Running layer discovery with {len(audio_files)} samples")
        
        # Load data
        mert_features, successful_audio = self.load_mert_features(audio_files)
        targets, successful_targets = self.load_targets(audio_files)
        
        # Find common files
        common_files = set(successful_audio) & set(successful_targets)
        logger.info(f"Common successful files: {len(common_files)}")
        
        if len(common_files) < 5:
            logger.error("Insufficient data for discovery")
            return {}
        
        # Filter to common files
        audio_indices = [i for i, f in enumerate(successful_audio) if f in common_files]
        target_indices = [i for i, f in enumerate(successful_targets) if f in common_files]
        
        filtered_mert = {}
        for layer in mert_features:
            filtered_mert[layer] = mert_features[layer][audio_indices]
        
        filtered_targets = {}
        for aspect in targets:
            filtered_targets[aspect] = targets[aspect][target_indices]
        
        # Test each layer for each aspect
        results = {}
        
        for aspect_name, target_values in filtered_targets.items():
            logger.info(f"\\n=== Testing {aspect_name.upper()} ===")
            
            aspect_results = {}
            best_layer = None
            best_score = -999
            
            for layer in range(13):
                layer_result = self.probe_layer_for_aspect(
                    filtered_mert[layer], target_values, aspect_name, layer
                )
                
                aspect_results[layer] = layer_result
                
                if layer_result['r2_mean'] > best_score:
                    best_score = layer_result['r2_mean']
                    best_layer = layer
                
                logger.info(f"Layer {layer:2d}: R² = {layer_result['r2_mean']:6.4f} ± {layer_result['r2_std']:.4f}")
            
            results[aspect_name] = {
                'layer_results': aspect_results,
                'best_layer': best_layer,
                'best_score': best_score
            }
            
            logger.info(f"Best layer for {aspect_name}: {best_layer} (R² = {best_score:.4f})")
        
        return results
    
    def update_validated_layers(self, discovery_results: Dict):
        """Update validated layer mappings based on discovery results."""
        
        for aspect, result in discovery_results.items():
            best_score = result['best_score']
            best_layer = result['best_layer']
            
            # Only validate if R² > 0.5 (strong predictive power)
            if best_score > 0.5:
                confidence = 'high' if best_score > 0.8 else 'medium'
                
                self.validated_layers[best_layer] = {
                    'aspect': aspect,
                    'r2_score': best_score,
                    'target': aspect,
                    'description': f"Encodes {aspect} information",
                    'confidence': confidence
                }
                
                logger.info(f"✅ Validated Layer {best_layer} for {aspect} (R² = {best_score:.4f})")
    
    def save_discovery_results(self, results: Dict, output_path: str = None):
        """Save discovery results to file."""
        if output_path is None:
            output_path = "/Users/jacobbieschke/mess-ai/pipeline/probing/layer_discovery_results.json"
        
        output_data = {
            'validated_layers': self.validated_layers,
            'discovery_results': results,
            'summary': {
                'total_layers_tested': 13,
                'validated_layers': len(self.validated_layers),
                'best_aspects': {
                    aspect: {
                        'layer': result['best_layer'],
                        'r2': result['best_score']
                    }
                    for aspect, result in results.items()
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Discovery results saved to {output_path}")
    
    def get_layer_recommendation_config(self) -> Dict:
        """Get configuration for layer-based recommendations."""
        config = {}
        
        for layer, info in self.validated_layers.items():
            if info['confidence'] == 'high':
                config[layer] = {
                    'use_for_similarity': True,
                    'aspect': info['aspect'],
                    'description': info['description'],
                    'weight': 1.0 if info['r2_score'] > 0.8 else 0.8
                }
        
        return config


def main():
    """Run layer discovery experiment."""
    
    discovery = LayerDiscoverySystem()
    
    # Run discovery
    results = discovery.discover_layer_functions(n_samples=35)
    
    # Update validated layers
    discovery.update_validated_layers(results)
    
    # Save results
    discovery.save_discovery_results(results)
    
    # Print summary
    print("\\n" + "="*60)
    print("LAYER DISCOVERY SUMMARY")
    print("="*60)
    
    for aspect, result in results.items():
        print(f"{aspect.upper():20}: Layer {result['best_layer']:2d} "
              f"(R² = {result['best_score']:6.4f})")
    
    print("\\n" + "="*60)
    print("VALIDATED LAYERS FOR RECOMMENDATIONS")
    print("="*60)
    
    config = discovery.get_layer_recommendation_config()
    
    if config:
        for layer, info in config.items():
            print(f"Layer {layer:2d}: {info['aspect']} (weight: {info['weight']:.1f})")
    else:
        print("No layers validated for recommendations yet.")
        print("Need aspects with R² > 0.5 for validation.")
    
    print("\\n✅ Layer discovery completed!")
    return results


if __name__ == "__main__":
    results = main()