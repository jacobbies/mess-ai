"""
Music metadata models for the MESS-AI system.
"""
from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TrackMetadata(BaseModel):
    """Complete metadata for a music track."""
    
    model_config = ConfigDict(
        json_encoders={
            date: lambda v: v.isoformat() if v else None
        }
    )
    
    # Core identifiers
    track_id: str = Field(..., description="Unique identifier (filename without extension)")
    filename: str = Field(..., description="Original filename with extension")
    
    # Musical information
    title: str = Field(..., description="Human-readable piece title")
    composer: str = Field(..., description="Composer name (last name for sorting)")
    composer_full: str = Field(..., description="Full composer name")
    opus: Optional[str] = Field(None, description="Opus or catalog number (e.g., BWV 849, Op. 27)")
    movement: Optional[str] = Field(None, description="Movement number or name")
    movement_name: Optional[str] = Field(None, description="Movement descriptive name (e.g., Allegro)")
    
    # Musical characteristics
    era: Optional[Literal["Medieval", "Renaissance", "Baroque", "Classical", "Romantic", "Modern", "Contemporary"]] = None
    form: Optional[str] = Field(None, description="Musical form (e.g., Sonata, Prelude, Waltz)")
    key_signature: Optional[str] = Field(None, description="Key signature (e.g., C# minor, Eb major)")
    tempo_marking: Optional[str] = Field(None, description="Tempo marking (e.g., Andante, Presto)")
    
    # Performance information
    performer_id: Optional[str] = Field(None, description="Performer identifier from filename")
    performer_name: Optional[str] = Field(None, description="Full performer name")
    instrument: str = Field(default="Piano", description="Primary instrument")
    ensemble: Optional[str] = Field(None, description="Ensemble name if applicable")
    
    # Recording information
    recording_date: Optional[date] = Field(None, description="Recording date")
    year_composed: Optional[int] = Field(None, description="Year the piece was composed")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    
    # Dataset information
    dataset_source: str = Field(default="SMD", description="Source dataset (SMD, MAESTRO, etc.)")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    
    # Search and categorization
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    difficulty_level: Optional[Literal["Beginner", "Intermediate", "Advanced", "Professional"]] = None
    popular_rank: Optional[int] = Field(None, description="Popularity ranking within dataset")
    
    @field_validator('tags', mode='before')
    @classmethod
    def ensure_tags_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        return v
    
    @field_validator('composer')
    @classmethod
    def extract_last_name(cls, v):
        """Ensure composer field contains last name for sorting."""
        if not v:
            return v
        # Handle special cases like "Bach" vs "Johann Sebastian Bach"
        parts = v.strip().split()
        if len(parts) > 1 and not v.startswith(('van ', 'von ', 'de ', 'di ')):
            return parts[-1]
        return v


class ComposerInfo(BaseModel):
    """Information about a composer."""
    name: str
    full_name: str
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    era: str
    nationality: Optional[str] = None
    track_count: int = 0


class SearchFilters(BaseModel):
    """Search filters for the music catalog."""
    composer: Optional[str] = None
    era: Optional[str] = None
    form: Optional[str] = None
    key_signature: Optional[str] = None
    tags: Optional[List[str]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    dataset_source: Optional[str] = None


# Mapping of common composer names to full names and metadata
COMPOSER_INFO = {
    "Bach": {
        "full_name": "Johann Sebastian Bach",
        "birth_year": 1685,
        "death_year": 1750,
        "era": "Baroque",
        "nationality": "German"
    },
    "Bartok": {
        "full_name": "Béla Bartók",
        "birth_year": 1881,
        "death_year": 1945,
        "era": "Modern",
        "nationality": "Hungarian"
    },
    "Beethoven": {
        "full_name": "Ludwig van Beethoven",
        "birth_year": 1770,
        "death_year": 1827,
        "era": "Classical",
        "nationality": "German"
    },
    "Brahms": {
        "full_name": "Johannes Brahms",
        "birth_year": 1833,
        "death_year": 1897,
        "era": "Romantic",
        "nationality": "German"
    },
    "Chopin": {
        "full_name": "Frédéric Chopin",
        "birth_year": 1810,
        "death_year": 1849,
        "era": "Romantic",
        "nationality": "Polish"
    },
    "Haydn": {
        "full_name": "Franz Joseph Haydn",
        "birth_year": 1732,
        "death_year": 1809,
        "era": "Classical",
        "nationality": "Austrian"
    },
    "Liszt": {
        "full_name": "Franz Liszt",
        "birth_year": 1811,
        "death_year": 1886,
        "era": "Romantic",
        "nationality": "Hungarian"
    },
    "Mozart": {
        "full_name": "Wolfgang Amadeus Mozart",
        "birth_year": 1756,
        "death_year": 1791,
        "era": "Classical",
        "nationality": "Austrian"
    },
    "Rachmaninoff": {
        "full_name": "Sergei Rachmaninoff",
        "birth_year": 1873,
        "death_year": 1943,
        "era": "Romantic",
        "nationality": "Russian"
    },
    "Rachmaninov": {  # Alternative spelling
        "full_name": "Sergei Rachmaninoff",
        "birth_year": 1873,
        "death_year": 1943,
        "era": "Romantic",
        "nationality": "Russian"
    },
    "Ravel": {
        "full_name": "Maurice Ravel",
        "birth_year": 1875,
        "death_year": 1937,
        "era": "Modern",
        "nationality": "French"
    },
    "Skryabin": {
        "full_name": "Alexander Scriabin",
        "birth_year": 1872,
        "death_year": 1915,
        "era": "Romantic",
        "nationality": "Russian"
    }
}


# Mapping of opus numbers to piece titles
OPUS_TITLES = {
    # Bach
    "BWV849": "Well-Tempered Clavier, Book 1: Prelude and Fugue No. 4 in C# minor",
    "BWV871": "Well-Tempered Clavier, Book 2: Prelude and Fugue No. 2 in C minor",
    "BWV875": "Well-Tempered Clavier, Book 2: Prelude and Fugue No. 6 in D minor",
    "BWV888": "Prelude and Fugue in A major",
    
    # Bartók
    "SZ080": "Out of Doors",
    
    # Beethoven
    "Op027No1": "Piano Sonata No. 13 in E-flat major 'Quasi una fantasia'",
    "Op031No2": "Piano Sonata No. 17 in D minor 'Tempest'",
    "WoO080": "32 Variations in C minor",
    
    # Brahms
    "Op005": "Piano Sonata No. 3 in F minor",
    "Op010No1": "Ballade No. 1 in D minor 'Edward'",
    "Op010No2": "Ballade No. 2 in D major",
    
    # Chopin
    "Op010-03": "Étude Op. 10, No. 3 in E major 'Tristesse'",
    "Op010-04": "Étude Op. 10, No. 4 in C# minor 'Torrent'",
    "Op026No1": "Polonaise No. 1 in C# minor",
    "Op026No2": "Polonaise No. 2 in E-flat minor",
    "Op028": "24 Preludes",
    "Op029": "Impromptu No. 1 in A-flat major",
    "Op048No1": "Nocturne No. 13 in C minor",
    "Op066": "Fantaisie-Impromptu in C# minor",
    
    # Haydn
    "Hob017No4": "String Quartet Op. 17, No. 4",
    "HobXVINo52": "Piano Sonata No. 52 in E-flat major",
    
    # Liszt
    "AnnesDePelerinage-LectureDante": "Années de pèlerinage: Après une Lecture du Dante",
    "KonzertetuedeNo2LaLeggierezza": "Trois études de concert: No. 2 'La leggierezza'",
    
    # Mozart
    "KV265": "12 Variations on 'Ah vous dirai-je, Maman'",
    "KV398": "Piano Sonata No. 12 in F major (fragment)",
    
    # Rachmaninoff
    "Op036": "Piano Sonata No. 2 in B-flat minor",
    "Op039No1": "Étude-Tableau Op. 39, No. 1 in C minor",
    
    # Ravel
    "JeuxDEau": "Jeux d'eau",
    "ValsesNoblesEtSentimentales": "Valses nobles et sentimentales",
    
    # Scriabin
    "Op008No8": "Étude Op. 8, No. 8 in A-flat major"
}