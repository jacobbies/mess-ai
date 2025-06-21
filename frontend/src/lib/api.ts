import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TrackMetadata {
  track_id: string;
  title: string;
  composer: string;
  composer_full: string;
  era?: string;
  form?: string;
  key_signature?: string;
  opus?: string;
  movement?: string;
  filename: string;
  tags: string[];
  recording_date?: string;
}

export interface Recommendation extends TrackMetadata {
  similarity_score: number;
}

export interface RecommendationResponse {
  reference_track: string;
  recommendations: Recommendation[];
  total_tracks: number;
  reference_metadata?: {
    title: string;
    composer: string;
    composer_full: string;
    era?: string;
    form?: string;
    key_signature?: string;
    tags: string[];
  };
}

export interface TracksResponse {
  tracks: TrackMetadata[];
  count: number;
  filters?: {
    composer?: string;
    era?: string;
    form?: string;
    search?: string;
  };
}

export const apiClient = {
  async getTracks(): Promise<TracksResponse> {
    const response = await api.get<TracksResponse>('/tracks');
    return response.data;
  },

  async getRecommendations(trackName: string, topK: number = 5): Promise<RecommendationResponse> {
    const response = await api.get<RecommendationResponse>(`/recommend/${encodeURIComponent(trackName)}?top_k=${topK}`);
    return response.data;
  },

  getAudioUrl(filename: string): string {
    return `${API_BASE_URL}/audio/${encodeURIComponent(filename)}`;
  },

  getWaveformUrl(filename: string): string {
    return `${API_BASE_URL}/waveform/${encodeURIComponent(filename)}`;
  },
};

export default apiClient;