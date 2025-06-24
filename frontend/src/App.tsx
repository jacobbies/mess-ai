import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Moon, Sun, Music2, RefreshCw } from 'lucide-react';
import { Button } from './components/ui/button';
import MusicPlayer from './components/MusicPlayer';
import TrackList from './components/TrackList';
import RecommendationList from './components/RecommendationList';
import Waveform from './components/Waveform';
import { apiClient, Recommendation, TrackMetadata } from './lib/api';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'true' || 
           window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  
  const [tracks, setTracks] = useState<TrackMetadata[]>([]);
  const [currentTrack, setCurrentTrack] = useState<string | null>(null);
  const [currentTrackId, setCurrentTrackId] = useState<string | null>(null);
  const [currentTrackMetadata, setCurrentTrackMetadata] = useState<TrackMetadata | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [referenceTrack, setReferenceTrack] = useState<string>('');
  const [referenceMetadata, setReferenceMetadata] = useState<any>(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  useEffect(() => {
    loadTracks();
  }, []);

  const loadTracks = async () => {
    try {
      setIsLoadingTracks(true);
      setError(null);
      const response = await apiClient.getTracks();
      setTracks(response.tracks);
      console.log('Loaded tracks:', response.tracks.length);
    } catch (error) {
      console.error('Failed to load tracks:', error);
      setError('Failed to load tracks. Is the API server running?');
      setTracks([]);
    } finally {
      setIsLoadingTracks(false);
    }
  };

  const handleTrackSelect = (trackId: string) => {
    // Find track metadata
    const metadata = tracks.find(t => t.track_id === trackId || t.filename === trackId);
    if (metadata) {
      setCurrentTrack(metadata.filename);
      setCurrentTrackId(metadata.track_id);
      setCurrentTrackMetadata(metadata);
    } else {
      // Fallback for tracks without metadata
      setCurrentTrack(trackId.endsWith('.wav') ? trackId : `${trackId}.wav`);
      setCurrentTrackId(trackId);
      setCurrentTrackMetadata(null);
    }
  };

  const handleRecommendationsRequest = async (trackId: string) => {
    try {
      setIsLoadingRecommendations(true);
      setError(null);
      
      // For recommendations, we need to use the filename with -SMD suffix
      // Find the metadata for this track to get the proper filename
      const metadata = tracks.find(t => t.track_id === trackId);
      const trackNameForRecommendations = metadata 
        ? metadata.filename.replace('.wav', '') 
        : trackId;
      
      setReferenceTrack(trackNameForRecommendations);
      const response = await apiClient.getRecommendations(trackNameForRecommendations, 5);
      setRecommendations(response.recommendations);
      setReferenceMetadata(response.reference_metadata);
    } catch (error) {
      console.error('Failed to load recommendations:', error);
      setError('Failed to load recommendations');
      setRecommendations([]);
      setReferenceMetadata(null);
    } finally {
      setIsLoadingRecommendations(false);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <motion.div 
              className="flex items-center gap-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="p-2 bg-primary rounded-lg">
                <Music2 className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Music Similarity Explorer</h1>
                <p className="text-sm text-muted-foreground">
                  AI-powered classical music discovery
                </p>
              </div>
            </motion.div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={loadTracks}
                className="h-10 w-10"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={toggleDarkMode}
                className="h-10 w-10"
              >
                {darkMode ? (
                  <Sun className="h-4 w-4" />
                ) : (
                  <Moon className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-destructive/10 border-b border-destructive/20 text-destructive px-4 py-3">
          <div className="container mx-auto flex items-center justify-between">
            <span className="text-sm">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
            >
              Dismiss
            </Button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Track List */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="lg:col-span-1"
          >
            <TrackList
              tracks={tracks}
              currentTrack={currentTrack}
              onTrackSelect={handleTrackSelect}
              isLoading={isLoadingTracks}
            />
          </motion.div>

          {/* Middle Column - Player and Waveform */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-1 space-y-6"
          >
            <MusicPlayer
              currentTrack={currentTrack}
              currentTrackId={currentTrackId}
              currentTrackMetadata={currentTrackMetadata}
              onRecommendationsRequest={handleRecommendationsRequest}
            />
            
            <Waveform trackName={currentTrack} />
          </motion.div>

          {/* Right Column - Recommendations */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="lg:col-span-1"
          >
            <RecommendationList
              recommendations={recommendations}
              referenceTrack={referenceTrack}
              referenceMetadata={referenceMetadata}
              onTrackSelect={handleTrackSelect}
              isLoading={isLoadingRecommendations}
            />
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 bg-card/30">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-sm text-muted-foreground">
            <p>
              Powered by MERT embeddings and FAISS similarity search
            </p>
            <p className="mt-1">
              Classical music from the Saarland Music Dataset
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;