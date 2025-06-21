import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { apiClient } from '../lib/api';
import { formatTrackName } from '../lib/utils';

interface WaveformProps {
  trackName: string | null;
}

const Waveform: React.FC<WaveformProps> = ({ trackName }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  if (!trackName) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Waveform Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 bg-muted rounded-lg flex items-center justify-center">
            <p className="text-muted-foreground text-sm">
              Select a track to view waveform
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const waveformUrl = apiClient.getWaveformUrl(trackName);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">
          Waveform - {formatTrackName(trackName)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative overflow-hidden rounded-lg bg-muted">
          {!imageLoaded && !imageError && (
            <div className="h-32 flex items-center justify-center">
              <motion.div
                className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            </div>
          )}
          
          {imageError && (
            <div className="h-32 flex items-center justify-center">
              <p className="text-muted-foreground text-sm">
                Waveform not available
              </p>
            </div>
          )}
          
          <motion.img
            src={waveformUrl}
            alt={`Waveform for ${trackName}`}
            className={`w-full h-auto ${imageLoaded ? 'block' : 'hidden'}`}
            onLoad={() => {
              setImageLoaded(true);
              setImageError(false);
            }}
            onError={() => {
              setImageError(true);
              setImageLoaded(false);
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: imageLoaded ? 1 : 0 }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default Waveform;