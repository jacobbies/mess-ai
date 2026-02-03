from typing import Optional


class S3Keys:
    """Centralized S3 key namespace - no state needed, hence staticmethods."""

    # === EMBEDDINGS ===

    @staticmethod
    def embedding_smd(track_id: str) -> str:
        return f"embeddings/smd-embeddings/aggregated/{track_id}.npy"

    #@staticmethod
    #def embedding_maestro(track_id: str) -> str:
    #    return f"embeddings/maestro-embeddings/aggregated/{track_id}.npy"

    # === AUDIO ===

    @staticmethod
    def audio_smd(track_id: str) -> str:
        return f"audio/wav44/{track_id}.wav"

    @staticmethod
    def audio_maestro(track_id: str) -> str:
        return f"audio/maestro/{track_id}.wav"

    # === LIST PREFIXES ===

    @staticmethod
    def embeddings_prefix_smd() -> str:
        return "embeddings/smd-embeddings/aggregated/"

    #@staticmethod
    #def embeddings_prefix_maestro() -> str:
    #    return "embeddings/maestro-embeddings/aggregated/"

    # === PARSING ===

    @staticmethod
    def parse_song_id(s3_key: str) -> str:
        """Extract song_id from embedding S3 key."""
        if 'smd-embeddings' in s3_key:
            filename = s3_key.split('/')[-1].replace('.npy', '')
            return f"smd/{filename}"

        if 'maestro-embeddings' in s3_key:
            filename = s3_key.split('/')[-1].replace('.npy', '')
            return f"maestro/{filename}"

        raise ValueError(f"Unknown embedding format: {s3_key}")


if __name__ == '__main__':
    track = "Bach_BWV849-01_001_20090916-SMD"

    print("SMD Embedding:", S3Keys.embedding_smd(track))
    print("SMD Audio:", S3Keys.audio_smd(track))
    print("SMD Prefix:", S3Keys.embeddings_prefix_smd())

    key = S3Keys.embedding_smd(track)
    print("Parse:", S3Keys.parse_song_id(key))