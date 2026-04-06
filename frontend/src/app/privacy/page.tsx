'use client';

import { useEffect, useState } from 'react';
import {
  fetchPrivacyStatus,
  fetchAuditLog,
  getAuditExportUrl,
  PrivacyStatus,
  AuditEntry,
} from '@/lib/api';
import styles from './page.module.css';

// ── Action type filter options ──────────────────────────────────────────────
const ACTION_FILTERS = [
  { label: 'All', value: '' },
  { label: 'Created', value: 'DATA_CREATED' },
  { label: 'Accessed', value: 'DATA_ACCESSED' },
  { label: 'Privacy', value: 'PRIVACY_APPLIED' },
  { label: 'Pipeline', value: 'PIPELINE_STARTED' },
  { label: 'Errors', value: 'PIPELINE_FAILED' },
];

// ── Helper: action badge styling ────────────────────────────────────────────
function getActionStyle(action: string): string {
  if (action.includes('CREATED')) return styles.actionCreated;
  if (action.includes('ACCESSED')) return styles.actionAccessed;
  if (action.includes('PRIVACY')) return styles.actionPrivacy;
  if (action.includes('FAILED')) return styles.actionError;
  if (action.includes('PIPELINE') || action.includes('ENCRYPTION') || action.includes('DECRYPTION'))
    return styles.actionPipeline;
  return styles.actionAccessed;
}

// ── Helper: format timestamp ────────────────────────────────────────────────
function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString('en-US', {
    month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

// ── Helper: truncate resource path for display ──────────────────────────────
function truncateResource(resource: string | null): string {
  if (!resource) return '—';
  // Show last two path segments or the full string if short
  const parts = resource.replace(/\\/g, '/').split('/');
  if (parts.length > 2) {
    return '…/' + parts.slice(-2).join('/');
  }
  return resource;
}

// ── Compliance Card Component ───────────────────────────────────────────────
interface ComplianceCardProps {
  icon: string;
  title: string;
  description: string;
  enabled: boolean;
  iconStyle: string;
}

function ComplianceCard({ icon, title, description, enabled, iconStyle }: ComplianceCardProps) {
  return (
    <div className={`${styles.complianceCard} ${enabled ? styles.cardEnabled : styles.cardDisabled}`}>
      <div className={styles.cardHeader}>
        <div className={`${styles.cardIcon} ${iconStyle}`}>{icon}</div>
        <span className={`${styles.statusBadge} ${enabled ? styles.badgeActive : styles.badgeInactive}`}>
          {enabled ? 'Active' : 'Inactive'}
        </span>
      </div>
      <div className={styles.cardTitle}>{title}</div>
      <div className={styles.cardDescription}>{description}</div>
    </div>
  );
}

// ── Main Privacy Dashboard ──────────────────────────────────────────────────
export default function PrivacyDashboard() {
  const [privacyStatus, setPrivacyStatus] = useState<PrivacyStatus | null>(null);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditOffset, setAuditOffset] = useState(0);
  const [actionFilter, setActionFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [auditLoading, setAuditLoading] = useState(false);

  const PAGE_SIZE = 25;

  // Fetch privacy status on mount
  useEffect(() => {
    async function loadStatus() {
      try {
        const status = await fetchPrivacyStatus();
        setPrivacyStatus(status);
      } catch (err) {
        console.error('Failed to fetch privacy status:', err);
      } finally {
        setLoading(false);
      }
    }
    loadStatus();
  }, []);

  // Fetch audit log (re-fetch when filter or offset changes)
  useEffect(() => {
    async function loadAudit() {
      setAuditLoading(true);
      try {
        const res = await fetchAuditLog(PAGE_SIZE, auditOffset, actionFilter || undefined);
        setAuditEntries(res.entries);
        setAuditTotal(res.total);
      } catch (err) {
        console.error('Failed to fetch audit log:', err);
      } finally {
        setAuditLoading(false);
      }
    }
    loadAudit();
  }, [auditOffset, actionFilter]);

  // Reset offset when filter changes
  const handleFilterChange = (value: string) => {
    setActionFilter(value);
    setAuditOffset(0);
  };

  if (loading) return <div className={styles.loadingState}>Loading privacy status…</div>;

  const ps = privacyStatus;

  return (
    <div className={styles.privacyPage}>

      {/* ── Page Header ── */}
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.pageTitle}>Privacy & Compliance</h1>
          <p className={styles.pageSubtitle}>
            Edge-level AI privacy controls, encrypted storage, and audit trail
          </p>
        </div>
      </div>

      {/* ── Compliance Status Cards ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Compliance Status</h2>
        </div>
        <div className={styles.complianceGrid}>
          <ComplianceCard
            icon="👤"
            title="Automatic Face Blurring"
            description="MediaPipe face detection with Gaussian blur applied to all stored video frames and entity crops."
            enabled={ps?.face_blur_enabled ?? false}
            iconStyle={ps?.face_blur_enabled ? styles.iconGreen : styles.iconRed}
          />
          <ComplianceCard
            icon="🚗"
            title="License Plate Redaction"
            description="OpenCV contour-based plate detection with solid-fill redaction on all stored outputs."
            enabled={ps?.plate_redaction_enabled ?? false}
            iconStyle={ps?.plate_redaction_enabled ? styles.iconGreen : styles.iconRed}
          />
          <ComplianceCard
            icon="⚡"
            title="Edge-Level Processing"
            description="All privacy operations run locally on-device. No frames or data leave the edge node."
            enabled={ps?.edge_processing ?? true}
            iconStyle={styles.iconBlue}
          />
          <ComplianceCard
            icon="🔒"
            title="Encrypted Storage (AES-256)"
            description="AES-256-GCM encryption for video clips, entity crops, and CSV data at rest."
            enabled={ps?.encryption_enabled ?? false}
            iconStyle={ps?.encryption_enabled ? styles.iconGreen : styles.iconAmber}
          />
          <ComplianceCard
            icon="📋"
            title="Audit Logging"
            description="Tamper-evident audit trail with SHA-256 checksums for every data access and pipeline operation."
            enabled={true}
            iconStyle={styles.iconGreen}
          />
        </div>
      </section>

      {/* ── Runtime Statistics ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Privacy Statistics</h2>
          <span className={styles.pulseGreen}>● Live</span>
        </div>
        <div className={styles.statsGrid}>
          <div className={styles.statCard}>
            <span className={styles.statValue}>{ps?.faces_blurred.toLocaleString() ?? 0}</span>
            <span className={styles.statLabel}>Faces Blurred</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statValue}>{ps?.plates_redacted.toLocaleString() ?? 0}</span>
            <span className={styles.statLabel}>Plates Redacted</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statValue}>{ps?.frames_processed.toLocaleString() ?? 0}</span>
            <span className={styles.statLabel}>Frames Processed</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statValue}>{ps?.crops_processed.toLocaleString() ?? 0}</span>
            <span className={styles.statLabel}>Crops Processed</span>
          </div>
        </div>
      </section>

      {/* ── Audit Log ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Audit Trail</h2>
          <a href={getAuditExportUrl()} className={styles.exportBtn} download>
            ↓ Export CSV
          </a>
        </div>

        {/* Filter bar */}
        <div className={styles.filterBar}>
          {ACTION_FILTERS.map(f => (
            <button
              key={f.value}
              className={`${styles.filterBtn} ${actionFilter === f.value ? styles.filterBtnActive : ''}`}
              onClick={() => handleFilterChange(f.value)}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Audit Table */}
        {auditLoading ? (
          <div className={styles.loadingState}>Loading audit entries…</div>
        ) : auditEntries.length === 0 ? (
          <div className={styles.emptyState}>No audit entries found.</div>
        ) : (
          <div className={styles.tableCard}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Actor</th>
                  <th>Resource</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {auditEntries.map(entry => (
                  <tr key={entry.ID}>
                    <td className={styles.cellTimestamp}>{formatTimestamp(entry.Timestamp)}</td>
                    <td>
                      <span className={`${styles.actionBadge} ${getActionStyle(entry.Action)}`}>
                        {entry.Action}
                      </span>
                    </td>
                    <td className={styles.cellMono}>{entry.Actor}</td>
                    <td className={styles.cellResource} title={entry.Resource ?? ''}>
                      {truncateResource(entry.Resource)}
                    </td>
                    <td className={styles.cellMono} style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {entry.Details && entry.Details !== '{}' ? entry.Details : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div className={styles.pagination}>
              <button
                className={styles.paginationBtn}
                onClick={() => setAuditOffset(Math.max(0, auditOffset - PAGE_SIZE))}
                disabled={auditOffset === 0}
              >
                ← Previous
              </button>
              <span className={styles.paginationInfo}>
                {auditOffset + 1}–{Math.min(auditOffset + PAGE_SIZE, auditTotal)} of {auditTotal}
              </span>
              <button
                className={styles.paginationBtn}
                onClick={() => setAuditOffset(auditOffset + PAGE_SIZE)}
                disabled={auditOffset + PAGE_SIZE >= auditTotal}
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
