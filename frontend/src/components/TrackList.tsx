import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Music, Clock, User, Hash } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { cn } from '../lib/utils';
import { TrackMetadata } from '../lib/api';

interface TrackListProps {
  tracks: TrackMetadata[];
  currentTrack: string | null;
  onTrackSelect: (track: string) => void;
  isLoading?: boolean;
}

const TrackList: React.FC<TrackListProps> = ({
  tracks,
  currentTrack,
  onTrackSelect,
  isLoading = false,
}) => {
  const [groupBy, setGroupBy] = useState<'composer' | 'era' | 'none'>('composer');
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  // Group tracks by composer or era
  const groupedTracks = React.useMemo(() => {
    if (groupBy === 'none') {
      return { 'All Tracks': tracks };
    }
    
    const groups: Record<string, TrackMetadata[]> = {};
    tracks.forEach(track => {
      const key = groupBy === 'composer' 
        ? track.composer_full || track.composer || 'Unknown'
        : track.era || 'Unknown';
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(track);
    });
    
    return groups;
  }, [tracks, groupBy]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Music className="h-5 w-5" />
            Loading Tracks...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="h-12 bg-muted animate-pulse rounded-md"
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Music className="h-5 w-5" />
            Classical Music Collection ({tracks.length} tracks)
          </span>
          <div className="flex gap-1">
            <Button
              variant={groupBy === 'composer' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setGroupBy('composer')}
              className="text-xs"
            >
              <User className="h-3 w-3 mr-1" />
              Composer
            </Button>
            <Button
              variant={groupBy === 'era' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setGroupBy('era')}
              className="text-xs"
            >
              <Clock className="h-3 w-3 mr-1" />
              Era
            </Button>
            <Button
              variant={groupBy === 'none' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setGroupBy('none')}
              className="text-xs"
            >
              <Hash className="h-3 w-3 mr-1" />
              All
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-4 max-h-[600px] overflow-y-auto"
        >
          {Object.entries(groupedTracks).map(([groupName, groupTracks]) => (
            <div key={groupName}>
              {groupBy !== 'none' && (
                <h3 className="text-sm font-semibold text-muted-foreground mb-2 sticky top-0 bg-background/95 backdrop-blur-sm py-1">
                  {groupName} ({groupTracks.length})
                </h3>
              )}
              <div className="space-y-2">
                <AnimatePresence>
                  {groupTracks.map((track) => (
                    <motion.div
                      key={track.track_id}
                      variants={itemVariants}
                      layout
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={cn(
                        "flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors",
                        currentTrack === track.filename
                          ? "bg-primary/10 border-primary"
                          : "hover:bg-muted/50"
                      )}
                      onClick={() => onTrackSelect(track.track_id)}
                    >
                      <div className="flex-1 min-w-0 space-y-1">
                        <p className="font-medium text-sm truncate">
                          {track.title}
                        </p>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span>{track.composer}</span>
                          {track.key_signature && (
                            <>
                              <span>•</span>
                              <span>{track.key_signature}</span>
                            </>
                          )}
                          {track.form && (
                            <>
                              <span>•</span>
                              <span>{track.form}</span>
                            </>
                          )}
                        </div>
                      </div>
                      
                      <Button
                        variant={currentTrack === track.filename ? "default" : "ghost"}
                        size="icon"
                        className="h-8 w-8 flex-shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          onTrackSelect(track.track_id);
                        }}
                      >
                        <Play className="h-3 w-3" />
                      </Button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          ))}
        </motion.div>
      </CardContent>
    </Card>
  );
};

export default TrackList;