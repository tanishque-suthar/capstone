/**
 * API client for the Capstone Video Surveillance Backend
 * Connects to FastAPI backend running on http://localhost:8000
 */

// API base URL - can be configured via environment variable
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * EventDetail interface matching the backend Pydantic model
 */
export interface EventDetail {
  Event_ID: string;
  Trigger_Time: number;
  Raw_Video_Path: string;
  Causal_CSV_Path: string;
  Crops_Dir_Path: string;
  Duration_s: number | null;
  Status: string;
  Source_Video_Path: string | null;
  Video_ID: string | null;
}

/**
 * VideoSource interface matching the backend Pydantic model
 */
export interface VideoSource {
  Video_ID: string;
  Label: string;
  File_Path: string;
  Added_At: number;
}

/**
 * Result from a RAG semantic search
 */
export interface RAGSearchResult {
  event_id: string;
  object_id: string;
  image_path: string;
  distance: number;
}

/**
 * Fetch all events from the backend
 */
export async function fetchEvents(): Promise<EventDetail[]> {
  const res = await fetch(`${API_BASE}/api/events`);
  if (!res.ok) {
    throw new Error(`Failed to fetch events: ${res.statusText}`);
  }
  const data = await res.json();
  return data.events;
}

/**
 * Fetch details for a specific event
 */
export async function fetchEventDetail(eventId: string): Promise<EventDetail> {
  const res = await fetch(`${API_BASE}/api/events/${eventId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch event ${eventId}: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch list of crop filenames for an event
 */
export async function fetchCrops(eventId: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/events/${eventId}/crops`);
  if (!res.ok) {
    throw new Error(`Failed to fetch crops for ${eventId}: ${res.statusText}`);
  }
  const data = await res.json();
  return data.crops || [];
}

/**
 * Build URL for a specific crop image
 */
export function getCropUrl(eventId: string, filename: string): string {
  return `${API_BASE}/api/events/${eventId}/crops/${filename}`;
}

/**
 * Build URL for the event's CSV file
 */
export function getCsvUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/csv`;
}

/**
 * Build URL for the event's extracted video clip
 */
export function getVideoUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/video`;
}

/**
 * Build URL for the event's source video file
 */
export function getSourceVideoUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/source-video`;
}

/**
 * Fetch all registered video sources
 */
export async function fetchSources(): Promise<VideoSource[]> {
  const res = await fetch(`${API_BASE}/api/sources`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sources: ${res.statusText}`);
  }
  const data = await res.json();
  return data.sources;
}

/**
 * Build URL for streaming a source video
 */
export function getSourceStreamUrl(videoId: string): string {
  return `${API_BASE}/api/sources/${videoId}/stream`;
}

/**
 * Trigger the pipeline on a video file
 */
export async function triggerPipeline(videoPath: string): Promise<{ event_id: string | null; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: videoPath }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Failed to trigger pipeline: ${detail}`);
  }
  return res.json();
}

/**
 * Index an event's crops into LanceDB RAG pipeline
 */
export async function ragIngest(eventId: string): Promise<{ status: string; event_id: string; ingested_count: number }> {
  const res = await fetch(`${API_BASE}/api/rag/ingest/${eventId}`, {
    method: 'POST',
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Failed to ingest event: ${detail}`);
  }
  return res.json();
}

/**
 * Perform semantic search for entities
 */
export async function ragSearch(query: string, limit: number = 5): Promise<RAGSearchResult[]> {
  const res = await fetch(`${API_BASE}/api/rag/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Search failed: ${detail}`);
  }
  const data = await res.json();
  return data.results || [];
}

/**
 * Fetch system configuration
 */
export async function fetchConfig(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) {
    throw new Error(`Failed to fetch config: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch system logs (last 100 lines)
 */
export async function fetchLogs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/logs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch logs: ${res.statusText}`);
  }
  const data = await res.json();
  return data.logs || [];
}


// ── Privacy & Audit API ──────────────────────────────────────────────────────

/**
 * Privacy status interface
 */
export interface PrivacyStatus {
  face_blur_enabled: boolean;
  plate_redaction_enabled: boolean;
  edge_processing: boolean;
  encryption_enabled: boolean;
  encryption_key_configured: boolean;
  faces_blurred: number;
  plates_redacted: number;
  frames_processed: number;
  crops_processed: number;
}

/**
 * Audit log entry interface
 */
export interface AuditEntry {
  ID: number;
  Timestamp: number;
  Action: string;
  Actor: string;
  Resource: string | null;
  Details: string | null;
  Checksum: string | null;
}

/**
 * Audit log response with pagination
 */
export interface AuditLogResponse {
  entries: AuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Fetch current privacy configuration and statistics
 */
export async function fetchPrivacyStatus(): Promise<PrivacyStatus> {
  const res = await fetch(`${API_BASE}/api/privacy/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch privacy status: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch audit log entries with optional filters
 */
export async function fetchAuditLog(
  limit: number = 50,
  offset: number = 0,
  action?: string,
  actor?: string,
): Promise<AuditLogResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (action) params.set('action', action);
  if (actor) params.set('actor', actor);

  const res = await fetch(`${API_BASE}/api/privacy/audit?${params}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch audit log: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Build URL for exporting audit log as CSV
 */
export function getAuditExportUrl(): string {
  return `${API_BASE}/api/privacy/audit/export`;
}
