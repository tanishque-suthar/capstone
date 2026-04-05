'use client';

import { useEffect, useState } from 'react';
import {
  fetchGradcamFrames,
  getGradcamUrl,
  fetchShapPlotUrl,
  fetchShapValues,
} from '@/lib/api';
import styles from './XAIPanel.module.css';

interface XAIPanelProps {
  eventId: string;
}

export default function XAIPanel({ eventId }: XAIPanelProps) {
  const [gradcamFrames, setGradcamFrames] = useState<string[]>([]);
  const [activeGradcam, setActiveGradcam] = useState(0);
  const [shapPlotUrl, setShapPlotUrl] = useState<string | null>(null);
  const [shapValues, setShapValues] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      try {
        const [frames, plotUrl, values] = await Promise.all([
          fetchGradcamFrames(eventId),
          fetchShapPlotUrl(eventId),
          fetchShapValues(eventId),
        ]);
        if (!active) return;
        setGradcamFrames(frames);
        setShapPlotUrl(plotUrl);
        setShapValues(values);
      } catch (err) {
        console.error('XAI load failed:', err);
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, [eventId]);

  const hasData = gradcamFrames.length > 0 || shapPlotUrl || shapValues;

  if (loading) {
    return (
      <section className={styles.xaiSection}>
        <h2 className={styles.sectionTitle}>
          Explainability (XAI)
        </h2>
        <div className={styles.loadingState}>Loading XAI artifacts…</div>
      </section>
    );
  }

  if (!hasData) return null;

  const maxShap = shapValues ? Math.max(...Object.values(shapValues), 0.001) : 1;

  return (
    <section className={styles.xaiSection}>
      <div className={styles.xaiHeader}>
        <h2 className={styles.sectionTitle}>
          Explainability (XAI)
        </h2>
        <div className={styles.xaiBadges}>
          {gradcamFrames.length > 0 && (
            <span className={styles.badge}>GradCAM</span>
          )}
          {shapValues && (
            <span className={styles.badge}>SHAP</span>
          )}
        </div>
      </div>

      <div className={styles.xaiGrid}>
        {/* ── GradCAM Carousel ──────────────────────────────────── */}
        {gradcamFrames.length > 0 && (
          <div className={styles.xaiCard}>
            <h3 className={styles.cardTitle}>
              GradCAM — Detection Attention Maps
            </h3>
            <p className={styles.cardDesc}>
              Highlights which image regions the YOLO detector focused on when making detections.
            </p>

            <div className={styles.gradcamViewer}>
              <img
                src={getGradcamUrl(eventId, gradcamFrames[activeGradcam])}
                alt={`GradCAM frame ${activeGradcam}`}
                className={styles.gradcamImage}
              />
            </div>

            {gradcamFrames.length > 1 && (
              <div className={styles.gradcamNav}>
                <button
                  className={styles.navBtn}
                  disabled={activeGradcam <= 0}
                  onClick={() => setActiveGradcam(i => Math.max(0, i - 1))}
                >
                  ◀
                </button>
                <div className={styles.thumbStrip}>
                  {gradcamFrames.map((fname, idx) => (
                    <button
                      key={fname}
                      className={`${styles.thumb} ${idx === activeGradcam ? styles.thumbActive : ''}`}
                      onClick={() => setActiveGradcam(idx)}
                    >
                      <img
                        src={getGradcamUrl(eventId, fname)}
                        alt={`Thumb ${idx}`}
                        className={styles.thumbImg}
                      />
                    </button>
                  ))}
                </div>
                <button
                  className={styles.navBtn}
                  disabled={activeGradcam >= gradcamFrames.length - 1}
                  onClick={() => setActiveGradcam(i => Math.min(gradcamFrames.length - 1, i + 1))}
                >
                  ▶
                </button>
              </div>
            )}

            <div className={styles.frameLabel}>
              Frame {gradcamFrames[activeGradcam]?.replace('gradcam_frame_', '').replace('.jpg', '')} 
              <span className={styles.frameCount}>({activeGradcam + 1} / {gradcamFrames.length})</span>
            </div>
          </div>
        )}

        {/* ── SHAP Feature Importance ──────────────────────────── */}
        {(shapPlotUrl || shapValues) && (
          <div className={styles.xaiCard}>
            <h3 className={styles.cardTitle}>
              SHAP — Feature Importance
            </h3>
            <p className={styles.cardDesc}>
              Shows which behavioural features most influenced the anomaly detection for this event.
            </p>

            {shapValues && (
              <div className={styles.shapBars}>
                {Object.entries(shapValues).map(([feature, value]) => (
                  <div key={feature} className={styles.shapRow}>
                    <span className={styles.shapLabel}>
                      {feature.replace(/_/g, ' ')}
                    </span>
                    <div className={styles.shapBarTrack}>
                      <div
                        className={styles.shapBarFill}
                        style={{ width: `${Math.min(100, (value / maxShap) * 100)}%` }}
                      />
                    </div>
                    <span className={styles.shapValue}>
                      {value.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {shapPlotUrl && (
              <div className={styles.shapPlotWrap}>
                <img
                  src={shapPlotUrl}
                  alt="SHAP summary plot"
                  className={styles.shapPlotImg}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
