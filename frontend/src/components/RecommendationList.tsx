import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Play, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { cn, formatTrackName } from '../lib/utils';
import { Recommendation } from '../lib/api';

interface RecommendationListProps {
  recommendations: Recommendation[];
  referenceTrack: string;
  referenceMetadata?: any;
  onTrackSelect: (track: string) => void;
  isLoading?: boolean;
}

const RecommendationList: React.FC<RecommendationListProps> = ({
  recommendations,
  referenceTrack,
  referenceMetadata,
  onTrackSelect,
  isLoading = false,
}) => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 },
  };

  const getSimilarityColor = (score: number) => {
    if (score >= 0.9) return "text-green-600 dark:text-green-400";
    if (score >= 0.8) return "text-blue-600 dark:text-blue-400";
    if (score >= 0.7) return "text-yellow-600 dark:text-yellow-400";
    return "text-gray-600 dark:text-gray-400";
  };

  const getSimilarityLabel = (score: number) => {
    if (score >= 0.9) return "Highly Similar";
    if (score >= 0.8) return "Very Similar";
    if (score >= 0.7) return "Similar";
    return "Somewhat Similar";
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Finding Similar Tracks...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
                <div className="h-3 bg-muted animate-pulse rounded w-1/2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (recommendations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Similar Tracks
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <TrendingUp className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              Select a track to find similar music using AI
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5" />
          Similar to "{referenceMetadata?.title || formatTrackName(referenceTrack)}"
        </CardTitle>
      </CardHeader>
      <CardContent>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-3"
        >
          <AnimatePresence>
            {recommendations.map((rec, index) => (
              <motion.div
                key={rec.track_id}
                variants={itemVariants}
                layout
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center justify-between p-4 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
                onClick={() => onTrackSelect(rec.track_id)}
              >
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded">
                      #{index + 1}
                    </span>
                    <p className="font-medium text-sm truncate">
                      {rec.title}
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <div className={cn("text-xs font-semibold", getSimilarityColor(rec.similarity_score))}>
                        {(rec.similarity_score * 100).toFixed(1)}% similar
                      </div>
                      <span className="text-xs text-muted-foreground">
                        • {getSimilarityLabel(rec.similarity_score)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="text-xs text-muted-foreground">
                    <span>{rec.composer_full}</span>
                    {rec.era && rec.form && (
                      <span> • {rec.era} {rec.form}</span>
                    )}
                    {rec.key_signature && (
                      <span> • {rec.key_signature}</span>
                    )}
                  </div>
                </div>
                
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 flex-shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    onTrackSelect(rec.track_id);
                  }}
                >
                  <Play className="h-3 w-3" />
                </Button>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      </CardContent>
    </Card>
  );
};

export default RecommendationList;