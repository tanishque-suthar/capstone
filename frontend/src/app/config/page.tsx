'use client';

import { useEffect, useState } from 'react';
import { fetchConfig } from '@/lib/api';
import styles from './config.module.css';

export default function Config() {
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchConfig();
        setConfig(data);
        setLoading(false);
      } catch (err) {
        console.error(err);
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div className={styles.container}>Loading configuration...</div>;
  }
  if (!config) {
    return <div className={styles.container}>Error loading configuration.</div>;
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>System Configuration</h1>
        <p className={styles.subtitle}>Current perception and pipeline parameters</p>
      </header>
      
      <div className={styles.homographyStatus}>
          Homography Calibration: 
          {config.homography_exists ? 
             <span className={styles.statusOk}> Active</span> : 
             <span className={styles.statusWarn}> Missing (Spatial Data NaN)</span>}
      </div>

      <div className={styles.grid}>
        {['video', 'threshold', 'yolo', 'tracker', 'interpolation', 'crop'].map((category) => (
          <div key={category} className={styles.card}>
            <h2 className={styles.cardTitle}>{category.toUpperCase()}</h2>
            <div className={styles.props}>
              {config[category] && Object.entries(config[category]).map(([key, val]) => (
                <div key={key} className={styles.propRow}>
                  <span className={styles.propKey}>{key}</span>
                  <span className={styles.propVal}>{Array.isArray(val) ? val.join(', ') : String(val)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
