"""
Example usage of the UnifiedMusicRecommender

This shows how developers can easily switch between different
recommendation strategies with the same interface.
"""

from mess_ai.models import UnifiedMusicRecommender

def main():
    # Initialize the unified recommender
    recommender = UnifiedMusicRecommender(features_dir="data/processed/features")
    
    track_id = "Bach_BWV849-01_001_20090916-SMD"  # Example track
    
    print("=== Unified Music Recommender Demo ===\n")
    
    # Check what's available
    info = recommender.get_strategy_info()
    print("Available strategies:", info["available_strategies"])
    print("Available modes:", info["available_modes"])
    print()
    
    # 1. Basic similarity recommendations
    print("1. Similarity-based recommendations:")
    recs = recommender.recommend(track_id, strategy="similarity", n_recommendations=3)
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 2. Diverse recommendations with MMR
    print("2. Diverse recommendations (MMR):")
    recs = recommender.recommend(track_id, strategy="diverse", mode="mmr", n_recommendations=3)
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 3. Cluster-based diversity
    print("3. Diverse recommendations (Cluster-based):")
    recs = recommender.recommend(track_id, strategy="diverse", mode="cluster", n_recommendations=3)
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 4. Popular recommendations
    print("4. Popularity-boosted recommendations:")
    recs = recommender.recommend(track_id, strategy="popular", n_recommendations=3)
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 5. Random recommendations
    print("5. Random recommendations:")
    recs = recommender.recommend(track_id, strategy="random", n_recommendations=3, seed=42)
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 6. Hybrid recommendations
    print("6. Hybrid recommendations (60% similarity + 40% diverse):")
    recs = recommender.recommend(
        track_id, 
        strategy="hybrid", 
        n_recommendations=3,
        weights={"similarity": 0.6, "diverse": 0.4}
    )
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()
    
    # 7. Configure strategies on the fly
    print("7. Configuring strategies dynamically:")
    recommender.configure_strategy("diverse", lambda_param=0.8)  # More diversity
    recs = recommender.recommend(track_id, strategy="diverse", mode="mmr", n_recommendations=3)
    print("   MMR with higher diversity (lambda=0.8):")
    for track, score in recs:
        print(f"   {track}: {score:.3f}")
    print()

def api_integration_example():
    """Example of how this would integrate with your API"""
    
    # In your API endpoint:
    def get_recommendations_endpoint(
        track_id: str,
        strategy: str = "similarity",
        mode: str = None,
        n_recommendations: int = 5
    ):
        recommender = UnifiedMusicRecommender()
        
        # Single line to get any type of recommendation!
        recommendations = recommender.recommend(
            track_id=track_id,
            strategy=strategy,
            mode=mode,
            n_recommendations=n_recommendations
        )
        
        return {
            "track_id": track_id,
            "strategy": strategy,
            "mode": mode,
            "recommendations": [
                {"track_id": track, "score": score}
                for track, score in recommendations
            ]
        }
    
    # Usage examples:
    print("=== API Integration Examples ===")
    
    track = "Bach_BWV849-01_001_20090916-SMD"
    
    # Different strategies with same interface
    examples = [
        ("similarity", None),
        ("diverse", "mmr"),
        ("diverse", "cluster"),
        ("popular", None),
        ("hybrid", None)
    ]
    
    for strategy, mode in examples:
        result = get_recommendations_endpoint(track, strategy, mode, 2)
        print(f"\nStrategy: {strategy}" + (f" (mode: {mode})" if mode else ""))
        for rec in result["recommendations"]:
            print(f"  {rec['track_id']}: {rec['score']:.3f}")

if __name__ == "__main__":
    main()
    print("\n" + "="*50 + "\n")
    api_integration_example()