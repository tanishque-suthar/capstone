'use client';

import { useState } from 'react';
import { triggerPipeline } from '@/lib/api';
import { useRouter } from 'next/navigation';
import styles from './upload.module.css';

export default function UploadEvent() {
  const [videoPath, setVideoPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoPath.trim()) return;
    
    setLoading(true);
    setError(null);
    try {
      const result = await triggerPipeline(videoPath);
      router.push(`/events/${result.event_id}`);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Ingest Video Event</h1>
        <p className={styles.subtitle}>Trigger the Track 1 data engineering pipeline manually from local video assets.</p>
      </header>

      <div className={styles.card}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="videoPath" className={styles.label}>
              Absolute Path to Video File (.mp4)
            </label>
            <input
              id="videoPath"
              type="text"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              placeholder="e.g. C:/videos/intersection.mp4 or ./dataset/test.mp4"
              className={styles.input}
              disabled={loading}
            />
            <p className={styles.hint}>
              The backend will run Phase 0 (heuristic scanning) followed by Perception and Data Storage sequentially.
            </p>
          </div>

          {error && <div className={styles.errorAlert}>{error}</div>}

          <button 
            type="submit" 
            className={styles.submitBtn} 
            disabled={loading || !videoPath.trim()}
          >
            {loading ? 'Initializing Pipeline...' : 'Trigger Pipeline Analysis'}
          </button>
        </form>
      </div>
    </div>
  );
}
