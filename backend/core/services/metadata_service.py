"""
Metadata management service.
Handles all metadata-related business logic.
"""
from typing import Dict, List, Optional
import logging

from mess_ai.models.metadata import TrackMetadata

logger = logging.getLogger(__name__)


class MetadataService:
    """Service for managing track metadata."""
    
    def __init__(self, metadata_dict: Dict[str, TrackMetadata]):
        """Initialize with metadata dictionary."""
        self.metadata_dict = metadata_dict
        logger.info(f"MetadataService initialized with {len(metadata_dict)} tracks")
    
    def get_all_tracks(
        self,
        composer: Optional[str] = None,
        era: Optional[str] = None,
        form: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all tracks with optional filtering.
        
        Args:
            composer: Filter by composer name
            era: Filter by musical era
            form: Filter by musical form
            search: Search in title, composer, or tags
            
        Returns:
            List of track dictionaries with metadata
        """
        track_ids = list(self.metadata_dict.keys())
        tracks_with_metadata = []
        
        for track_id in track_ids:
            metadata = self.metadata_dict.get(track_id)
            
            # Apply filters
            if not self._passes_filters(metadata, composer, era, form, search):
                continue
            
            # Convert to response format
            track_data = self._format_track_metadata(metadata, track_id)
            tracks_with_metadata.append(track_data)
        
        # Sort by composer, then by opus/title
        tracks_with_metadata.sort(key=lambda x: (
            x.get('composer', ''),
            x.get('opus', ''),
            x.get('movement', ''),
            x.get('title', '')
        ))
        
        return tracks_with_metadata
    
    def get_track_metadata(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed metadata for a specific track.
        
        Args:
            track_id: The track identifier
            
        Returns:
            Track metadata dictionary or None if not found
        """
        metadata = self.metadata_dict.get(track_id)
        if not metadata:
            return None
        
        return {
            "track_id": metadata.track_id,
            "filename": metadata.filename,
            "title": metadata.title,
            "composer": metadata.composer,
            "composer_full": metadata.composer_full,
            "opus": metadata.opus,
            "movement": metadata.movement,
            "movement_name": metadata.movement_name,
            "era": metadata.era,
            "form": metadata.form,
            "key_signature": metadata.key_signature,
            "tempo_marking": metadata.tempo_marking,
            "performer_id": metadata.performer_id,
            "performer_name": metadata.performer_name,
            "instrument": metadata.instrument,
            "recording_date": metadata.recording_date.isoformat() if metadata.recording_date else None,
            "year_composed": metadata.year_composed,
            "duration_seconds": metadata.duration_seconds,
            "dataset_source": metadata.dataset_source,
            "tags": metadata.tags
        }
    
    def get_composers(self) -> List[Dict]:
        """
        Get all composers with track counts.
        
        Returns:
            List of composer dictionaries with statistics
        """
        composer_stats = {}
        
        for metadata in self.metadata_dict.values():
            composer = metadata.composer
            if composer not in composer_stats:
                composer_stats[composer] = {
                    "composer": composer,
                    "composer_full": metadata.composer_full,
                    "era": metadata.era,
                    "track_count": 0
                }
            composer_stats[composer]["track_count"] += 1
        
        # Convert to list and sort by track count
        composers = list(composer_stats.values())
        composers.sort(key=lambda x: x["track_count"], reverse=True)
        
        return composers
    
    def get_tags(self) -> List[Dict]:
        """
        Get all unique tags with counts.
        
        Returns:
            List of tag dictionaries with counts
        """
        tag_counts = {}
        
        for metadata in self.metadata_dict.values():
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by count
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"tag": tag, "count": count} for tag, count in sorted_tags]
    
    def get_track_count(self) -> int:
        """Get total number of tracks."""
        return len(self.metadata_dict)
    
    def _passes_filters(
        self,
        metadata: Optional[TrackMetadata],
        composer: Optional[str],
        era: Optional[str],
        form: Optional[str],
        search: Optional[str]
    ) -> bool:
        """Check if track passes all filters."""
        if not metadata:
            return False
        
        # Composer filter
        if composer:
            if (composer.lower() not in metadata.composer.lower() and 
                composer.lower() not in metadata.composer_full.lower()):
                return False
        
        # Era filter
        if era and metadata.era != era:
            return False
        
        # Form filter
        if form and metadata.form != form:
            return False
        
        # Search filter
        if search:
            search_lower = search.lower()
            if not any([
                search_lower in metadata.title.lower(),
                search_lower in metadata.composer.lower(),
                search_lower in metadata.composer_full.lower(),
                any(search_lower in tag for tag in metadata.tags)
            ]):
                return False
        
        return True
    
    def _format_track_metadata(self, metadata: Optional[TrackMetadata], track_id: str) -> Dict:
        """Format track metadata for API response."""
        if metadata:
            return {
                "track_id": track_id,
                "title": metadata.title,
                "composer": metadata.composer,
                "composer_full": metadata.composer_full,
                "era": metadata.era,
                "form": metadata.form,
                "key_signature": metadata.key_signature,
                "opus": metadata.opus,
                "movement": metadata.movement,
                "filename": metadata.filename,
                "tags": metadata.tags,
                "recording_date": metadata.recording_date.isoformat() if metadata.recording_date else None
            }
        else:
            # Fallback if metadata not found
            return {
                "track_id": track_id,
                "title": track_id.replace('_', ' ').replace('-', ' '),
                "filename": f"{track_id}.wav"
            }