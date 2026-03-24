'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  fetchEvents, fetchCrops, fetchSources,
  EventDetail, VideoSource,
  getCropUrl, getCsvUrl, getSourceStreamUrl,
} from '@/lib/api';
import VideoAnnotator, { VideoAnnotatorHandle } from '@/components/VideoAnnotator';
import styles from './page.module.css';

const PRE_BUFFER = 4.0;

async function parseCsv(eventId: string): Promise<any[]> {
  try {
    const res = await fetch(getCsvUrl(eventId));
    if (!res.ok) return [];
    const text = await res.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
      const row: any = {};
      line.split(',').forEach((v, i) => { row[headers[i].trim()] = v; });
      return row;
    });
  } catch { return []; }
}

function StatusBadge({ status }: { status: string }) {
  const cls = status.toLowerCase() === 'extracted' ? styles.badgeSuccess
            : status.toLowerCase() === 'failed'    ? styles.badgeDanger
            : styles.badgeWarning;
  return <span className={`${styles.badge} ${cls}`}>{status}</span>;
}

// ── Single Feed Player Tile ──────────────────────────────────────────────────
interface FeedPlayerProps {
  source: VideoSource;
  events: EventDetail[];
  csvData: any[];
}

function FeedPlayer({ source, events, csvData }: FeedPlayerProps) {
  const extracted = events.filter(e => e.Status === 'Extracted' && e.Trigger_Time > 0);
  const firstEvt = extracted[0];
  const startSec = firstEvt ? Math.max(0, firstEvt.Trigger_Time - PRE_BUFFER) : 0;
  const endSec = firstEvt ? startSec + (firstEvt.Duration_s ?? 10) : 0;

  return (
    <div className={styles.feedTile}>
      <div className={styles.feedHeader}>
        <div className={styles.feedLabelDot} />
        <span className={styles.feedLabelText}>{source.Label}</span>
        <span className={styles.feedChip}>{extracted.length} event{extracted.length !== 1 ? 's' : ''}</span>
        {firstEvt && (
          <Link href={`/events/${firstEvt.Event_ID}`} className={styles.feedAnalyze}>
            Analyze →
          </Link>
        )}
      </div>
      <div className={styles.feedPlayerWrap}>
        {firstEvt ? (
          <VideoAnnotator
            videoUrl={getSourceStreamUrl(source.Video_ID)}
            csvData={csvData}
            eventStartSec={startSec}
            eventEndSec={endSec}
            isSourceVideo={true}
          />
        ) : (
          <div className={styles.feedPlaceholder}>
            <video
              src={getSourceStreamUrl(source.Video_ID)}
              className={styles.feedPlaceholderVideo}
              muted
              preload="metadata"
              controls
              crossOrigin="anonymous"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Dashboard ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [sources,   setSources]   = useState<VideoSource[]>([]);
  const [eventsMap, setEventsMap] = useState<Record<string, EventDetail[]>>({});
  const [csvMap,    setCsvMap]    = useState<Record<string, any[]>>({});
  const [allEvents, setAllEvents] = useState<EventDetail[]>([]);
  const [allCrops,  setAllCrops]  = useState<{ eventId: string; filename: string }[]>([]);
  const [loading,   setLoading]   = useState(true);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const [srcs, evts] = await Promise.all([fetchSources(), fetchEvents()]);
        if (!mounted) return;
        setSources(srcs);
        setAllEvents(evts);

        // Group events by Video_ID
        const map: Record<string, EventDetail[]> = {};
        evts.forEach(e => {
          const vid = e.Video_ID || '__none__';
          if (!map[vid]) map[vid] = [];
          map[vid].push(e);
        });
        setEventsMap(map);

        // Load CSV for the first extracted event of each source
        const csvEntries: Record<string, any[]> = {};
        await Promise.all(srcs.map(async (src) => {
          const srcEvents = map[src.Video_ID] || [];
          const first = srcEvents.find(e => e.Status === 'Extracted' && e.Trigger_Time > 0);
          if (first) {
            csvEntries[src.Video_ID] = await parseCsv(first.Event_ID);
          }
        }));
        if (mounted) setCsvMap(csvEntries);

        // Collect all crops
        const extracted = evts.filter(e => e.Status === 'Extracted');
        const cropEntries: { eventId: string; filename: string }[] = [];
        await Promise.all(extracted.map(async (e) => {
          const crops = await fetchCrops(e.Event_ID);
          crops.forEach(f => cropEntries.push({ eventId: e.Event_ID, filename: f }));
        }));
        if (mounted) setAllCrops(cropEntries);
      } catch (err) {
        console.error(err);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  // Adaptive grid class
  const gridClass =
    sources.length === 1 ? styles.feedGrid1
    : sources.length === 2 ? styles.feedGrid2
    : sources.length <= 4  ? styles.feedGrid4
    : styles.feedGrid6;

  return (
    <div className={styles.dashboard}>

      {/* ── 1. CAMERA FEEDS ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Camera Feeds</h2>
          <span className={styles.countBadge}>{sources.length} source{sources.length !== 1 ? 's' : ''}</span>
        </div>

        {loading ? (
          <div className={styles.emptyState}>Loading feeds...</div>
        ) : sources.length === 0 ? (
          <div className={styles.emptyState}>
            No video sources registered.{' '}
            <Link href="/upload" className={styles.inlineLink}>Upload a video to get started.</Link>
          </div>
        ) : (
          <div className={`${styles.feedGrid} ${gridClass}`}>
            {sources.map(src => (
              <FeedPlayer
                key={src.Video_ID}
                source={src}
                events={eventsMap[src.Video_ID] || []}
                csvData={csvMap[src.Video_ID] || []}
              />
            ))}
          </div>
        )}
      </section>

      {/* ── 2. DETECTED ENTITIES ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Detected Entities</h2>
          <span className={styles.countBadge}>{allCrops.length} total</span>
        </div>

        {allCrops.length === 0 ? (
          <div className={styles.emptyState}>No entities detected yet.</div>
        ) : (
          <div className={styles.entitiesGrid}>
            {allCrops.map(({ eventId, filename }) => (
              <div key={`${eventId}-${filename}`} className={styles.entityCard}>
                <img src={getCropUrl(eventId, filename)} alt={filename} className={styles.entityImg} />
                <div className={styles.entityLabel}>
                  <span className={styles.entityId}>{filename.replace(`${eventId}_`, '').replace('_crop.jpg', '')}</span>
                  <span className={styles.entityEvent}>{eventId.replace('EVT_', '')}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ── 3. EVENTS LOG ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Events Log</h2>
          <Link href="/upload" className={styles.detailLink}>+ New Run</Link>
        </div>

        {loading ? (
          <div className={styles.emptyState}>Loading events...</div>
        ) : allEvents.length === 0 ? (
          <div className={styles.emptyState}>No events registered yet.</div>
        ) : (
          <div className={styles.tableCard}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Source</th>
                  <th>Trigger Time</th>
                  <th>Duration</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {allEvents.map(event => (
                  <tr key={event.Event_ID}>
                    <td><span className={styles.cellId}>{event.Event_ID}</span></td>
                    <td>{sources.find(s => s.Video_ID === event.Video_ID)?.Label || '—'}</td>
                    <td>{event.Trigger_Time.toFixed(1)}s</td>
                    <td>{event.Duration_s ? `${event.Duration_s.toFixed(1)}s` : '—'}</td>
                    <td><StatusBadge status={event.Status} /></td>
                    <td>
                      <Link href={`/events/${event.Event_ID}`} className={styles.btnAction} onClick={e => e.stopPropagation()}>
                        Analyze
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
