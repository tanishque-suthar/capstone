'use client';

import { useState, useEffect, useRef } from 'react';
import { fetchEvents, EventDetail, ragIngest, ragSearch, RAGSearchResult, getCropUrl } from '@/lib/api';
import styles from './search.module.css';
import Link from 'next/link';

export default function SearchPage() {
  const [events, setEvents]         = useState<EventDetail[]>([]);
  const [query, setQuery]           = useState('');
  const [results, setResults]       = useState<RAGSearchResult[]>([]);
  const [searching, setSearching]   = useState(false);
  const [ingesting, setIngesting]   = useState<string | null>(null);
  const [ingestMsg, setIngestMsg]   = useState('');
  const [error, setError]           = useState('');
  const [searched, setSearched]     = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load events for the ingest panel
  useEffect(() => {
    fetchEvents().then(evts => setEvents(evts.filter(e => e.Status === 'Extracted'))).catch(console.error);
  }, []);

  // ── Ingest handler ─────────────────────────────────────────────────────────
  async function handleIngest(eventId: string) {
    setIngesting(eventId);
    setIngestMsg('');
    setError('');
    try {
      const res = await ragIngest(eventId);
      setIngestMsg(`Indexed ${res.ingested_count} entities from ${eventId}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIngesting(null);
    }
  }

  // ── Search handler ─────────────────────────────────────────────────────────
  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError('');
    setResults([]);
    setSearched(true);
    try {
      const res = await ragSearch(query.trim(), 12);
      setResults(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  // Map image_path (relative like "dataset/EVT_xxx/entity_crops/filename.jpg") to the API crop URL
  function resultToCropUrl(r: RAGSearchResult): string {
    const parts = r.image_path.split('/');
    const filename = parts[parts.length - 1];
    return getCropUrl(r.event_id, filename);
  }

  // Similarity score from distance (lower distance = higher similarity)
  function distanceToScore(distance: number): number {
    return Math.max(0, Math.round((1 - distance) * 100));
  }

  return (
    <div className={styles.container}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <h1 className={styles.title}>Entity Search</h1>
        <p className={styles.subtitle}>
          Search detected entities using natural language powered by SigLIP + LanceDB
        </p>
      </header>

      {/* ── Search Bar ── */}
      <form className={styles.searchForm} onSubmit={handleSearch}>
        <div className={styles.searchInputWrap}>
          <svg className={styles.searchIcon} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
          </svg>
          <input
            ref={inputRef}
            type="text"
            className={styles.searchInput}
            placeholder="Describe the entity... e.g. 'red car', 'white van', 'person walking'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="submit"
            className={styles.searchBtn}
            disabled={searching || !query.trim()}
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {/* ── Error / Info ── */}
      {error && <div className={styles.errorBanner}>{error}</div>}
      {ingestMsg && <div className={styles.successBanner}>{ingestMsg}</div>}

      {/* ── Results ── */}
      {searched && !searching && results.length === 0 && !error && (
        <div className={styles.emptyResults}>
          No matching entities found. Try a different query or ingest more events.
        </div>
      )}

      {results.length > 0 && (
        <section className={styles.resultsSection}>
          <div className={styles.resultsHeader}>
            <h2 className={styles.sectionTitle}>Results</h2>
            <span className={styles.resultCount}>{results.length} matches for &ldquo;{query}&rdquo;</span>
          </div>
          <div className={styles.resultsGrid}>
            {results.map((r, i) => (
              <Link
                key={`${r.event_id}-${r.object_id}-${i}`}
                href={`/events/${r.event_id}`}
                className={styles.resultCard}
              >
                <div className={styles.resultImgWrap}>
                  <img src={resultToCropUrl(r)} alt={r.object_id} className={styles.resultImg} />
                  <div className={styles.scoreBadge}>{distanceToScore(r.distance)}%</div>
                </div>
                <div className={styles.resultMeta}>
                  <span className={styles.resultObjId}>{r.object_id}</span>
                  <span className={styles.resultEventId}>{r.event_id.replace('EVT_', '')}</span>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      {/* ── Ingest Panel ── */}
      <section className={styles.ingestSection}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Index Events</h2>
          <span className={styles.sectionSub}>
            Ingest entity crops into the vector database before searching
          </span>
        </div>
        {events.length === 0 ? (
          <div className={styles.emptyResults}>No extracted events available.</div>
        ) : (
          <div className={styles.ingestGrid}>
            {events.map(event => (
              <div key={event.Event_ID} className={styles.ingestCard}>
                <div className={styles.ingestInfo}>
                  <span className={styles.ingestId}>{event.Event_ID}</span>
                  <span className={styles.ingestTime}>Trigger: {event.Trigger_Time.toFixed(1)}s</span>
                </div>
                <button
                  className={styles.ingestBtn}
                  onClick={() => handleIngest(event.Event_ID)}
                  disabled={ingesting === event.Event_ID}
                >
                  {ingesting === event.Event_ID ? 'Indexing...' : 'Index'}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
