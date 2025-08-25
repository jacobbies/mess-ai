"""
Intelligent Query Engine for Classical Music Similarity
Uses empirically validated MERT layers for natural language music queries.
"""

import sys
from pathlib import Path

# Add paths
sys.path.append('/Users/jacobbieschke/mess-ai/pipeline')

import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import logging
from dataclasses import dataclass

from query.layer_based_recommender import LayerBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Parsed query with detected musical aspects and weights."""
    aspects: Dict[str, float]
    track_constraint: Optional[str] = None
    n_results: int = 5
    raw_query: str = ""


class IntelligentQueryEngine:
    """Natural language query engine for classical music similarity."""
    
    def __init__(self):
        self.recommender = LayerBasedRecommender()
        
        # Query patterns for musical aspects
        self.aspect_patterns = {
            'timbral_similarity': [
                r'\b(timbre|timbral|tone|sound quality|instrumental sound)\b',
                r'\b(bright|dark|warm|cold|harsh|smooth|metallic|woody)\b',
                r'\b(similar sound|same instrument|instrumental character)\b'
            ],
            'spectral_brightness': [
                r'\b(bright|brilliant|sparkling|shimmering|luminous)\b',
                r'\b(dark|dull|muffled|warm|deep)\b',
                r'\b(spectral|frequency|brightness|clarity)\b'
            ],
            'timbral_texture': [
                r'\b(texture|rough|smooth|grainy|silky|velvety)\b',
                r'\b(dense|sparse|thick|thin|layered)\b',
                r'\b(instrumental texture|orchestration)\b'
            ],
            'acoustic_structure': [
                r'\b(resonance|reverb|space|acoustic|room|hall)\b',
                r'\b(echoing|resonant|dry|wet|spatial)\b',
                r'\b(acoustic properties|sound space)\b'
            ],
            'temporal_patterns': [
                r'\b(rhythm|rhythmic|tempo|beat|pulse)\b',
                r'\b(fast|slow|quick|steady|irregular|syncopated)\b',
                r'\b(temporal|time|duration|pacing)\b'
            ],
            'musical_phrasing': [
                r'\b(phrase|phrasing|melodic line|musical sentence)\b',
                r'\b(expressive|articulation|legato|staccato)\b',
                r'\b(musical structure|form|development)\b'
            ]
        }
        
        # Intensity modifiers
        self.intensity_patterns = {
            'very': 1.5,
            'extremely': 2.0,
            'slightly': 0.5,
            'somewhat': 0.7,
            'quite': 1.2,
            'really': 1.3,
            'absolutely': 2.0,
            'totally': 1.8,
            'incredibly': 2.0
        }
        
        logger.info(f"Initialized query engine with {len(self.recommender.track_names)} tracks")
    
    def parse_query(self, query: str) -> QueryIntent:
        """Parse natural language query into structured intent."""
        
        query_lower = query.lower()
        aspects = {}
        
        # Check for each musical aspect
        for aspect, patterns in self.aspect_patterns.items():
            aspect_score = 0.0
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    aspect_score += len(matches) * 0.3
                    
                    # Check for intensity modifiers before the match
                    for intensity, multiplier in self.intensity_patterns.items():
                        if re.search(rf'\b{intensity}\s+.*{pattern}', query_lower):
                            aspect_score *= multiplier
                            break
            
            if aspect_score > 0:
                aspects[aspect] = min(aspect_score, 2.0)  # Cap at 2.0
        
        # Normalize aspect weights to sum to 1.0
        if aspects:
            total_weight = sum(aspects.values())
            aspects = {k: v/total_weight for k, v in aspects.items()}
        else:
            # Default to timbral similarity if no aspects detected
            aspects = {'timbral_similarity': 1.0}
        
        # Extract number of results
        n_results = 5
        num_match = re.search(r'\b(\d+)\s*(results?|recommendations?|tracks?|pieces?)\b', query_lower)
        if num_match:
            n_results = min(int(num_match.group(1)), 20)  # Cap at 20
        
        # Look for track constraints
        track_constraint = None
        track_patterns = [
            r'\blike\s+([\w\s\-_]+?)(?:\s|$)',
            r'\bsimilar to\s+([\w\s\-_]+?)(?:\s|$)',
            r'\bcompare to\s+([\w\s\-_]+?)(?:\s|$)',
        ]
        
        for pattern in track_patterns:
            match = re.search(pattern, query)
            if match:
                potential_track = match.group(1).strip()
                # Try to find matching track
                matching_tracks = [
                    track for track in self.recommender.track_names 
                    if potential_track.lower() in track.lower()
                ]
                if matching_tracks:
                    track_constraint = matching_tracks[0]
                    break
        
        return QueryIntent(
            aspects=aspects,
            track_constraint=track_constraint,
            n_results=n_results,
            raw_query=query
        )
    
    def execute_query(self, query: str) -> Tuple[QueryIntent, List[Tuple[str, float, Dict[str, float]]]]:
        """Execute natural language query and return recommendations."""
        
        # Parse query intent
        intent = self.parse_query(query)
        logger.info(f"Parsed query: {intent.aspects}")
        
        # Get reference track
        if intent.track_constraint:
            reference_track = intent.track_constraint
        else:
            # Use first track as default reference
            reference_track = self.recommender.track_names[0]
        
        # Get recommendations based on aspects
        recommendations = self.recommender.multi_aspect_recommendation(
            reference_track,
            aspect_weights=intent.aspects,
            n_recommendations=intent.n_results,
            exclude_query=(intent.track_constraint is not None)
        )
        
        return intent, recommendations
    
    def explain_query_results(self, intent: QueryIntent, recommendations: List) -> str:
        """Generate human-readable explanation of query results."""
        
        explanation_parts = []
        
        # Explain detected aspects
        if len(intent.aspects) == 1:
            aspect, weight = list(intent.aspects.items())[0]
            explanation_parts.append(f"Focused on **{aspect.replace('_', ' ')}** similarity")
        else:
            aspect_desc = ", ".join([
                f"{aspect.replace('_', ' ')} ({weight:.1%})" 
                for aspect, weight in intent.aspects.items()
            ])
            explanation_parts.append(f"Combined multiple aspects: {aspect_desc}")
        
        # Explain reference
        if intent.track_constraint:
            explanation_parts.append(f"Using '{intent.track_constraint}' as reference")
        else:
            explanation_parts.append("Using default reference track")
        
        # Summary
        explanation = ". ".join(explanation_parts) + "."
        
        return explanation
    
    def get_query_suggestions(self) -> List[str]:
        """Get example queries to help users."""
        return [
            "Find tracks with similar timbre to Beethoven",
            "Show me 8 pieces with bright, sparkling sound",
            "What has similar acoustic structure?",
            "Find very rhythmic pieces like this one",
            "Show tracks with smooth, warm timbre",
            "Give me 3 pieces with similar musical phrasing",
            "Find extremely dark and resonant recordings",
            "What sounds similar in terms of texture and brightness?"
        ]
    
    def search_tracks(self, search_term: str) -> List[str]:
        """Search for tracks by name/composer."""
        search_lower = search_term.lower()
        
        matches = []
        for track in self.recommender.track_names:
            track_lower = track.lower()
            if search_lower in track_lower:
                matches.append(track)
        
        # Sort by relevance (shorter names first, exact matches)
        matches.sort(key=lambda x: (len(x), x))
        return matches[:10]


def demo():
    """Demo the intelligent query engine."""
    
    query_engine = IntelligentQueryEngine()
    
    if not query_engine.recommender.track_names:
        print("No tracks loaded! Check your feature files.")
        return
    
    print("🎼 Intelligent Classical Music Query Engine")
    print("=" * 60)
    
    # Demo queries
    demo_queries = [
        "Find tracks with very bright, sparkling timbre",
        "Show me 3 pieces with similar acoustic structure to Beethoven",
        "What has smooth, warm timbral texture?",
        "Find extremely rhythmic pieces with dense texture"
    ]
    
    for query in demo_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 40)
        
        try:
            intent, recommendations = query_engine.execute_query(query)
            explanation = query_engine.explain_query_results(intent, recommendations)
            
            print(f"📝 Analysis: {explanation}")
            print("\n🎵 Recommendations:")
            
            for i, (track, score, breakdown) in enumerate(recommendations, 1):
                print(f"{i}. {track}")
                print(f"   Combined similarity: {score:.4f}")
                
                # Show top contributing aspects
                top_aspects = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:2]
                aspects_str = ", ".join([f"{asp.replace('_', ' ')}: {val:.3f}" for asp, val in top_aspects])
                print(f"   Top aspects: {aspects_str}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Query engine demo completed!")
    
    # Show example suggestions
    print("\n💡 Try these example queries:")
    suggestions = query_engine.get_query_suggestions()
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"{i}. {suggestion}")


if __name__ == "__main__":
    demo()