'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './annotator.module.css';

interface VideoAnnotatorProps {
  videoUrl: string;
  csvData: any[];
  eventStartSec: number;
  eventEndSec: number;
  isSourceVideo: boolean;
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function VideoAnnotator({ videoUrl, csvData, eventStartSec, eventEndSec, isSourceVideo }: VideoAnnotatorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const animationRef = useRef<number | null>(null);

  // Parse CSV into frame-indexed structure
  const framesData = useRef<Map<number, any[]>>(new Map());

  useEffect(() => {
    const map = new Map<number, any[]>();
    csvData.forEach(row => {
      const frameId = parseInt(row.Frame_ID, 10);
      if (!isNaN(frameId)) {
        if (!map.has(frameId)) map.set(frameId, []);
        map.get(frameId)!.push(row);
      }
    });
    framesData.current = map;
  }, [csvData]);

  const isInEventWindow = useCallback((time: number) => {
    return time >= eventStartSec && time <= eventEndSec;
  }, [eventStartSec, eventEndSec]);

  const drawAnnotations = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas resolution to video intrinsic size
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 1920;
      canvas.height = video.videoHeight || 1080;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    setCurrentTime(video.currentTime);

    if (showAnnotations && isInEventWindow(video.currentTime)) {
      // For source video: compute which frame of the 10s clip we're at
      // For extracted clip: currentTime maps directly
      const relativeTime = isSourceVideo
        ? video.currentTime - eventStartSec
        : video.currentTime;
      
      const currentFrame = Math.max(0, Math.floor(relativeTime * 10)); // 10 FPS
      const rows = framesData.current.get(currentFrame) || [];

      rows.forEach(row => {
        const x1 = parseFloat(row.BBox_X1);
        const y1 = parseFloat(row.BBox_Y1);
        const x2 = parseFloat(row.BBox_X2);
        const y2 = parseFloat(row.BBox_Y2);
        
        if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) return;

        const width = x2 - x1;
        const height = y2 - y1;

        // Draw bounding box
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        ctx.fillStyle = 'rgba(59, 130, 246, 0.85)';
        const velocity = parseFloat(row.Velocity_mps || '0');
        const label = `${row.Class} ${row.Object_ID} | ${velocity.toFixed(1)}m/s`;
        ctx.font = 'bold 14px monospace';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 24, textWidth + 10, 24);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + 5, y1 - 7);
      });
    }

    if (!video.paused && !video.ended) {
      animationRef.current = requestAnimationFrame(drawAnnotations);
    }
  }, [showAnnotations, isSourceVideo, eventStartSec, isInEventWindow]);

  const handlePlay = () => { setIsPlaying(true); drawAnnotations(); };
  const handlePause = () => { setIsPlaying(false); if (animationRef.current) cancelAnimationFrame(animationRef.current); drawAnnotations(); };
  const handleSeek = () => { drawAnnotations(); };
  const handleTimeUpdate = () => { if (videoRef.current) setCurrentTime(videoRef.current.currentTime); };
  const handleLoadedMetadata = () => { if (videoRef.current) setDuration(videoRef.current.duration); };

  useEffect(() => {
    drawAnnotations();
  }, [showAnnotations, csvData, drawAnnotations]);

  // Jump to event start
  const jumpToEvent = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = eventStartSec;
      drawAnnotations();
    }
  };

  // Custom timeline click handler
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || !videoRef.current || duration === 0) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    videoRef.current.currentTime = fraction * duration;
    drawAnnotations();
  };

  // Calculate percentages for the timeline highlight
  const eventStartPct = duration > 0 ? (eventStartSec / duration) * 100 : 0;
  const eventWidthPct = duration > 0 ? ((eventEndSec - eventStartSec) / duration) * 100 : 0;
  const playheadPct = duration > 0 ? (currentTime / duration) * 100 : 0;
  const inEvent = isInEventWindow(currentTime);

  return (
    <div className={styles.wrapper}>
      {/* Video + Canvas Overlay */}
      <div className={styles.videoContainer}>
        <video
          ref={videoRef}
          src={videoUrl}
          className={styles.video}
          onPlay={handlePlay}
          onPause={handlePause}
          onSeeked={handleSeek}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          crossOrigin="anonymous"
        />
        <canvas
          ref={canvasRef}
          className={styles.canvas}
        />
      </div>

      {/* Custom Controls Bar */}
      <div className={styles.controlsBar}>
        {/* Play/Pause */}
        <button
          className={styles.playBtn}
          onClick={() => {
            if (videoRef.current) {
              if (videoRef.current.paused) videoRef.current.play();
              else videoRef.current.pause();
            }
          }}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        {/* Time Display */}
        <span className={styles.timeDisplay}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {/* Custom Timeline */}
        <div className={styles.timeline} ref={timelineRef} onClick={handleTimelineClick}>
          {/* Base track */}
          <div className={styles.timelineTrack}></div>

          {/* Event highlight zone */}
          {isSourceVideo && (
            <div
              className={styles.eventHighlight}
              style={{
                left: `${eventStartPct}%`,
                width: `${eventWidthPct}%`,
              }}
            >
              <div className={styles.eventHighlightLabel}>EVENT</div>
            </div>
          )}

          {/* Playhead */}
          <div
            className={styles.playhead}
            style={{ left: `${playheadPct}%` }}
          />
        </div>

        {/* Right Controls */}
        <div className={styles.rightControls}>
          {isSourceVideo && (
            <button
              className={styles.jumpBtn}
              onClick={jumpToEvent}
              title="Jump to extracted event"
            >
              Go to Event
            </button>
          )}

          <button
            className={`${styles.toggleBtn} ${showAnnotations ? styles.active : ''}`}
            onClick={() => setShowAnnotations(!showAnnotations)}
          >
            {showAnnotations ? '◉ Overlay ON' : '○ Overlay OFF'}
          </button>
        </div>
      </div>

      {/* Event in-window indicator */}
      {isSourceVideo && (
        <div className={`${styles.eventIndicator} ${inEvent ? styles.eventActive : ''}`}>
          {inEvent 
            ? '● Currently viewing extracted event window'
            : '○ Outside event window — annotations inactive'}
        </div>
      )}
    </div>
  );
}
