'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  fetchEvents, fetchCrops, EventDetail,
  getCropUrl, getCsvUrl, getSourceVideoUrl, getVideoUrl,
} from '@/lib/api';
import VideoAnnotator, { VideoAnnotatorHandle } from '@/components/VideoAnnotator';
import styles from './page.module.css';

const PRE_BUFFER = 4.0;

function buildVideoProps(event: EventDetail) {
  const hasSource = !!event.Source_Video_Path;
  const start = Math.max(0, event.Trigger_Time - PRE_BUFFER);
  const end   = start + (event.Duration_s ?? 10);
  return {
    videoUrl: hasSource ? getSourceVideoUrl(event.Event_ID) : getVideoUrl(event.Event_ID),
    eventStartSec: hasSource ? start : 0,
    eventEndSec:   hasSource ? end   : (event.Duration_s ?? 10),
    isSourceVideo: hasSource,
  };
}

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

export default function Dashboard() {
  const [events,      setEvents]      = useState<EventDetail[]>([]);
  const [activeEvent, setActiveEvent] = useState<EventDetail | null>(null);
  const [csvData,     setCsvData]     = useState<any[]>([]);
  const [allCrops,    setAllCrops]    = useState<{ eventId: string; filename: string }[]>([]);
  const [loading,     setLoading]     = useState(true);

  // Refs for imperative control
  const playerRef      = useRef<VideoAnnotatorHandle>(null);
  const playerSectionRef = useRef<HTMLDivElement>(null);

  // ── Switch the active event playing in the player ─────────────────────────
  async function activateEvent(event: EventDetail, seekToStart = false) {
    // If it's the same source video, just seek — no reload needed
    const sameVideo = activeEvent?.Source_Video_Path === event.Source_Video_Path
                   || (!activeEvent?.Source_Video_Path && !event.Source_Video_Path);

    if (!sameVideo || activeEvent?.Event_ID !== event.Event_ID) {
      // Load CSV for the new event
      const csv = await parseCsv(event.Event_ID);
      setCsvData(csv);
      setActiveEvent(event);
    }

    // Seek to the event's start in the source video
    if (seekToStart) {
      const start = Math.max(0, event.Trigger_Time - PRE_BUFFER);
      // Give React a tick to propagate the new URL and csvData if needed
      setTimeout(() => {
        playerRef.current?.seek(start);
      }, 50);
    }

    // Scroll up to the player smoothly
    playerSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Initial load ──────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const evts = await fetchEvents();
        if (!mounted) return;
        setEvents(evts);

        const extracted = evts.filter(e => e.Status === 'Extracted');
        if (extracted.length > 0) {
          const latest = extracted[0];
          setActiveEvent(latest);
          const csv = await parseCsv(latest.Event_ID);
          if (mounted) setCsvData(csv);

          const cropEntries: { eventId: string; filename: string }[] = [];
          await Promise.all(extracted.map(async (e) => {
            const crops = await fetchCrops(e.Event_ID);
            crops.forEach(f => cropEntries.push({ eventId: e.Event_ID, filename: f }));
          }));
          if (mounted) setAllCrops(cropEntries);
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (mounted) setLoading(false);
      }
    }

    load();
    const interval = setInterval(() => {
      fetchEvents().then(evts => { if (mounted) setEvents(evts); });
    }, 5000);

    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const videoProps = activeEvent ? buildVideoProps(activeEvent) : null;

  return (
    <div className={styles.dashboard}>

      {/* ── 1. VIDEO PLAYER ── */}
      <section className={styles.section} ref={playerSectionRef}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Live Feed</h2>
          {activeEvent && (
            <div className={styles.sectionMeta}>
              <span className={styles.chip}>{activeEvent.Event_ID}</span>
              <span className={styles.chip}>
                {Math.max(0, activeEvent.Trigger_Time - PRE_BUFFER).toFixed(1)}s –{' '}
                {(Math.max(0, activeEvent.Trigger_Time - PRE_BUFFER) + (activeEvent.Duration_s ?? 10)).toFixed(1)}s
              </span>
              <Link href={`/events/${activeEvent.Event_ID}`} className={styles.detailLink}>
                Full Analysis →
              </Link>
            </div>
          )}
        </div>

        <div className={styles.playerCard}>
          {loading ? (
            <div className={styles.playerPlaceholder}>Loading video feed...</div>
          ) : videoProps ? (
            <VideoAnnotator ref={playerRef} {...videoProps} csvData={csvData} />
          ) : (
            <div className={styles.playerPlaceholder}>
              No extracted events yet.{' '}
              <Link href="/upload" className={styles.inlineLink}>Trigger a pipeline run.</Link>
            </div>
          )}
        </div>
      </section>

      {/* ── 2. ALL DETECTED ENTITIES ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Detected Entities</h2>
          <span className={styles.countBadge}>{allCrops.length} total</span>
        </div>

        {allCrops.length === 0 ? (
          <div className={styles.emptyState}>No entities detected yet.</div>
        ) : (
          <div className={styles.entitiesGrid}>
            {allCrops.map(({ eventId, filename }) => {
              const evt = events.find(e => e.Event_ID === eventId);
              return (
                <button
                  key={`${eventId}-${filename}`}
                  className={styles.entityCard}
                  onClick={() => evt && activateEvent(evt, true)}
                  title={`Jump to ${eventId}`}
                >
                  <img
                    src={getCropUrl(eventId, filename)}
                    alt={filename}
                    className={styles.entityImg}
                  />
                  <div className={styles.entityLabel}>
                    <span className={styles.entityId}>{filename.replace(`${eventId}_`, '').replace('_crop.jpg', '')}</span>
                    <span className={styles.entityEvent}>{eventId.replace('EVT_', '')}</span>
                  </div>
                </button>
              );
            })}
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
        ) : events.length === 0 ? (
          <div className={styles.emptyState}>No events registered yet.</div>
        ) : (
          <div className={styles.tableCard}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Trigger Time</th>
                  <th>Duration</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {events.map(event => (
                  <tr
                    key={event.Event_ID}
                    className={activeEvent?.Event_ID === event.Event_ID ? styles.activeRow : ''}
                    onClick={() => activateEvent(event, true)}
                    title="Click to jump to event in player"
                  >
                    <td><span className={styles.cellId}>{event.Event_ID}</span></td>
                    <td>{event.Trigger_Time.toFixed(1)}s</td>
                    <td>{event.Duration_s ? `${event.Duration_s.toFixed(1)}s` : '—'}</td>
                    <td><StatusBadge status={event.Status} /></td>
                    <td>
                      <Link
                        href={`/events/${event.Event_ID}`}
                        className={styles.btnAction}
                        onClick={e => e.stopPropagation()}
                      >
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
