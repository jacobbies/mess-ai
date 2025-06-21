import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, SkipBack, SkipForward, Volume2 } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { cn, formatTime } from '../lib/utils';
import { apiClient } from '../lib/api';
import { TrackMetadata } from '../lib/api';

interface MusicPlayerProps {
  currentTrack: string | null;
  currentTrackMetadata: TrackMetadata | null;
  onTrackEnd?: () => void;
  onRecommendationsRequest?: (trackName: string) => void;
}

const MusicPlayer: React.FC<MusicPlayerProps> = ({
  currentTrack,
  currentTrackMetadata,
  onTrackEnd,
  onRecommendationsRequest,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (currentTrack && audioRef.current) {
      setIsLoading(true);
      audioRef.current.src = apiClient.getAudioUrl(currentTrack);
      audioRef.current.load();
    }
  }, [currentTrack]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      onTrackEnd?.();
    };

    const handleCanPlay = () => {
      setIsLoading(false);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('canplay', handleCanPlay);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('canplay', handleCanPlay);
    };
  }, [onTrackEnd]);

  const togglePlayPause = () => {
    if (!audioRef.current || !currentTrack) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
  };

  const handleFindSimilar = () => {
    if (currentTrack) {
      onRecommendationsRequest?.(currentTrack);
    }
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <audio ref={audioRef} preload="metadata" />
        
        {/* Track Info */}
        <div className="mb-6 text-center">
          {currentTrack ? (
            <>
              <h3 className="text-lg font-semibold mb-1">
                {currentTrackMetadata?.title || currentTrack.replace(/\.wav$/, '').replace(/_/g, ' ')}
              </h3>
              {currentTrackMetadata && (
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>{currentTrackMetadata.composer_full}</p>
                  <p className="text-xs">
                    {[
                      currentTrackMetadata.era,
                      currentTrackMetadata.key_signature,
                      currentTrackMetadata.form
                    ].filter(Boolean).join(' • ')}
                  </p>
                </div>
              )}
              <motion.div
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden cursor-pointer"
                onClick={handleSeek}
              >
                <motion.div
                  className="h-full bg-blue-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </motion.div>
              <div className="flex justify-between text-sm text-muted-foreground mt-2">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
            </>
          ) : (
            <p className="text-muted-foreground">No track selected</p>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center space-x-4 mb-6">
          <Button
            variant="outline"
            size="icon"
            onClick={handleFindSimilar}
            disabled={!currentTrack}
            className="hidden sm:flex"
          >
            <SkipBack className="h-4 w-4" />
          </Button>

          <Button
            variant="default"
            size="icon"
            onClick={togglePlayPause}
            disabled={!currentTrack || isLoading}
            className="h-12 w-12"
          >
            {isLoading ? (
              <motion.div
                className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            ) : isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5" />
            )}
          </Button>

          <Button
            variant="outline"
            size="icon"
            onClick={handleFindSimilar}
            disabled={!currentTrack}
            className="hidden sm:flex"
          >
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        {/* Volume Control */}
        <div className="flex items-center justify-center space-x-2">
          <Volume2 className="h-4 w-4" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={handleVolumeChange}
            className="w-24"
          />
        </div>

        {/* Find Similar Button */}
        <div className="mt-6 text-center">
          <Button
            onClick={handleFindSimilar}
            disabled={!currentTrack}
            variant="outline"
            className="w-full sm:w-auto"
          >
            Find Similar Tracks
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default MusicPlayer;