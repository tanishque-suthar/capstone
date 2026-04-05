'use client';

import { useEffect, useState, use } from 'react';
import {
  fetchEventDetail,
  fetchCrops,
  fetchReasoningReport,
  askReasoningQuestion,
  EventDetail,
  ReasoningReport,
  ReasoningAnswer,
  getCropUrl,
  getCsvUrl,
  getSourceVideoUrl,
  getVideoUrl,
} from '@/lib/api';
import VideoAnnotator from '@/components/VideoAnnotator';
import XAIPanel from '@/components/XAIPanel';
import styles from './event.module.css';
import Link from 'next/link';

const PROCESSED_STATUSES = new Set(['extracted', 'reasoned']);

export default function EventDetailView({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [crops, setCrops] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [csvData, setCsvData] = useState<any[]>([]);
  const [reasoning, setReasoning] = useState<ReasoningReport | null>(null);
  const [question, setQuestion] = useState('Why did this accident occur?');
  const [answer, setAnswer] = useState<ReasoningAnswer | null>(null);
  const [asking, setAsking] = useState(false);

  const objectClassMap = new Map<string, string>();
  for (const row of csvData) {
    const objectId = String(row.Object_ID || '');
    const classLabel = String(row.Class || '');
    if (objectId && classLabel && !objectClassMap.has(objectId)) {
      objectClassMap.set(objectId, classLabel);
    }
  }

  useEffect(() => {
    setLoading(true);
    setEvent(null);
    setCrops([]);
    setCsvData([]);
    setReasoning(null);
    setAnswer(null);

    let active = true;
    async function load() {
      try {
        const data = await fetchEventDetail(id);
        if (!active) return;
        setEvent(data);
        
        if (PROCESSED_STATUSES.has(data.Status.toLowerCase())) {
          const fetchedCrops = await fetchCrops(id);
          if (!active) return;
          setCrops(fetchedCrops);
          
          // Fetch and parse CSV
          const csvRes = await fetch(getCsvUrl(id));
          if (csvRes.ok) {
            const csvText = await csvRes.text();
            if (!active) return;
            const lines = csvText.trim().split('\n');
            const headers = lines[0].split(',');
            const parsedRows = lines.slice(1).map(line => {
              const values = line.split(',');
              const row: any = {};
              headers.forEach((h, i) => { row[h.trim()] = values[i]; });
              return row;
            });
            setCsvData(parsedRows);
          }

          try {
            const report = await fetchReasoningReport(id);
            if (!active) return;
            setReasoning(report);
          } catch (reasoningErr) {
            console.error('Failed to fetch reasoning report:', reasoningErr);
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => {
      active = false;
    };
  }, [id]);

  if (loading) return <div className={styles.container}>Loading Event Data...</div>;
  if (!event) return <div className={styles.container}>Event Not Found.</div>;

  // Calculate event clip boundaries in the source video timeline
  // The Trigger_Time is the time in the source video when the event was detected
  // Pre-buffer = 4s, so clip starts at Trigger_Time - 4
  // Duration_s tells us total clip length (typically 10s)
  const preBuffer = 4.0;
  const eventStartSec = Math.max(0, event.Trigger_Time - preBuffer);
  const eventEndSec = eventStartSec + (event.Duration_s || 10.0);

  // Use source video if available, otherwise fall back to extracted clip
  const hasSourceVideo = !!event.Source_Video_Path;
  const videoUrl = hasSourceVideo ? getSourceVideoUrl(event.Event_ID) : getVideoUrl(event.Event_ID);
  const isProcessed = PROCESSED_STATUSES.has(event.Status.toLowerCase());

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    try {
      const result = await askReasoningQuestion(id, question.trim());
      setAnswer(result);
    } catch (err) {
      console.error(err);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <Link href="/" className={styles.backBtn}>← Back</Link>
          <h1 className={styles.title}>Event Analysis: {event.Event_ID}</h1>
        </div>
        <div className={styles.metaCards}>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Status</span>
            <span className={styles.metaValue}>{event.Status}</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Trigger Time</span>
            <span className={styles.metaValue}>{event.Trigger_Time.toFixed(1)}s into source</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Event Window</span>
            <span className={styles.metaValue}>{eventStartSec.toFixed(1)}s – {eventEndSec.toFixed(1)}s</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Total Tracks</span>
            <span className={styles.metaValue}>{crops.length} entities</span>
          </div>
        </div>
      </header>

      {isProcessed && (
        <>
        <div className={styles.mediaGrid}>
          {/* Main Video & Analytics Pane */}
          <div className={styles.videoSection}>
            <div className={styles.playerCard}>
              <VideoAnnotator 
                videoUrl={videoUrl}
                csvData={csvData}
                eventStartSec={hasSourceVideo ? eventStartSec : 0}
                eventEndSec={hasSourceVideo ? eventEndSec : (event.Duration_s || 10)}
                isSourceVideo={hasSourceVideo}
              />
            </div>
          </div>
          
          {/* Entity Crops Gallery */}
          <div className={styles.cropsSection}>
            <h2 className={styles.sectionTitle}>Detected Entities</h2>
            <div className={styles.cropsGrid}>
              {crops.length > 0 ? crops.map(filename => (
                <div key={filename} className={styles.cropCard}>
                  {(() => {
                    const objectId = filename.replace(`${event.Event_ID}_`, '').replace('_crop.jpg', '');
                    const classLabel = objectClassMap.get(objectId);
                    return (
                      <>
                  <img src={getCropUrl(event.Event_ID, filename)} alt={filename} className={styles.cropImg} />
                  <div className={styles.cropLabel}>
                    {objectId}{classLabel ? ` • ${classLabel}` : ''}
                  </div>
                      </>
                    );
                  })()}
                </div>
              )) : (
                <div className={styles.emptyCrops}>No entities detected during event.</div>
              )}
            </div>
          </div>
        </div>
        <section className={styles.reasoningSection}>
          <div className={styles.reasoningHeader}>
            <h2 className={styles.sectionTitle}>Causal Reasoning</h2>
            {reasoning?.confidence_gate && (
              <span className={`${styles.gateBadge} ${reasoning.confidence_gate.sufficient ? styles.gateStrong : styles.gateWeak}`}>
                {reasoning.confidence_gate.sufficient ? 'Evidence Sufficient' : 'Insufficient Evidence'}
              </span>
            )}
          </div>

          {reasoning ? (
            <div className={styles.reasoningGrid}>
              <div className={styles.reasoningCard}>
                <h3 className={styles.cardTitle}>Summary</h3>
                <p className={styles.reasoningText}>{reasoning.summary}</p>
                {event.Reasoning_Summary && event.Reasoning_Summary !== reasoning.summary && (
                  <p className={styles.reasoningMuted}>{event.Reasoning_Summary}</p>
                )}
              </div>

              <div className={styles.reasoningCard}>
                <h3 className={styles.cardTitle}>Detected Anomalies</h3>
                {reasoning.anomalies.length > 0 ? (
                  <div className={styles.reasoningList}>
                    {reasoning.anomalies.slice(0, 4).map((item, idx) => (
                      <div key={`${item.kind}-${idx}`} className={styles.reasoningItem}>
                        <div className={styles.reasoningLabel}>{item.kind} at {Number(item.timestamp_s).toFixed(1)}s</div>
                        <div className={styles.reasoningText}>{String(item.reason)}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className={styles.reasoningMuted}>No explicit anomalies were summarized.</p>
                )}
              </div>

              <div className={styles.reasoningCard}>
                <h3 className={styles.cardTitle}>Top Hypotheses</h3>
                {reasoning.hypotheses.length > 0 ? (
                  <div className={styles.reasoningList}>
                    {reasoning.hypotheses.slice(0, 3).map((item, idx) => (
                      <div key={`${item.label}-${idx}`} className={styles.reasoningItem}>
                        <div className={styles.reasoningLabel}>
                          {String(item.label)} • {Math.round(Number(item.confidence) * 100)}%
                        </div>
                        <div className={styles.reasoningText}>{String(item.answer)}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className={styles.reasoningMuted}>No causal hypotheses available.</p>
                )}
              </div>

              <div className={styles.reasoningCard}>
                <h3 className={styles.cardTitle}>Ask About This Clip</h3>
                <form onSubmit={handleAsk} className={styles.askForm}>
                  <textarea
                    className={styles.askInput}
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question about this event..."
                  />
                  <button type="submit" className={styles.askButton} disabled={asking || !question.trim()}>
                    {asking ? 'Thinking...' : 'Ask'}
                  </button>
                </form>
                {answer && (
                  <div className={styles.answerCard}>
                    <div className={styles.reasoningLabel}>
                      Answer • {Math.round(answer.confidence * 100)}%
                    </div>
                    <div className={styles.reasoningText}>{answer.answer}</div>
                    {answer.evidence.length > 0 && (
                      <ul className={styles.evidenceList}>
                        {answer.evidence.map((item, idx) => (
                          <li key={`${idx}-${item}`} className={styles.evidenceItem}>{item}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className={styles.processingPane}>
              <h3 style={{ color: 'var(--text-accent)', marginBottom: '8px' }}>Causal Report Unavailable</h3>
              <p>This event record does not contain a generated reasoning report.</p>
              <p style={{ fontSize: '0.85rem', marginTop: '12px' }}>
                This usually occurs for "Legacy" events processed before Track 1 Reasoning was implemented, 
                or if the reasoning engine could not derive a confident causal explanation.
              </p>
            </div>
          )}
        </section>
        <XAIPanel eventId={event.Event_ID} />
        </>
      )}

      {event.Status === 'Processing' && (
        <div className={styles.processingPane}>
          <div className={styles.spinner}></div>
          <p>This event is currently being processed by the pipeline. Please refresh to check status.</p>
        </div>
      )}
    </div>
  );
}
