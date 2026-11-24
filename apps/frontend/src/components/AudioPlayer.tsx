import type { AudioInfo } from '@foshol-lens/contracts';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api, mediaUrl } from '../lib/api-client';
import { toBengaliDigits } from '../lib/format';

const AUDIO_POLL_MS = 1200;
const AUDIO_POLL_TIMEOUT_MS = 90_000;

function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${toBengaliDigits(minutes)}:${toBengaliDigits(String(seconds).padStart(2, '0'))}`;
}

/**
 * The spoken narrative.
 *
 * `preload="none"` is deliberate: a compact MP3 file is cheap, but on a rural
 * mobile connection even that should be opt-in, and audio is advice rather than
 * an alarm, so it never plays on its own.
 *
 * Synthesis finishes after the text is served, so a `pending` clip is polled
 * rather than written off - the reader is already reading while the voice is
 * being made.
 */
export function AudioPlayer({ audio: initial, analysisId }: { audio: AudioInfo; analysisId: string }) {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [audio, setAudio] = useState<AudioInfo>(initial);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);
  const [retryingStatus, setRetryingStatus] = useState(false);

  useEffect(() => {
    setAudio(initial);
    setFailed(false);
  }, [initial]);

  // Poll only while there is something to wait for.
  useEffect(() => {
    if (audio.status !== 'pending') return;

    let cancelled = false;
    const startedAt = Date.now();

    async function poll() {
      if (cancelled || Date.now() - startedAt > AUDIO_POLL_TIMEOUT_MS) return;

      try {
        const result = await api.getResult(analysisId);
        if (cancelled) return;
        if (result.audio) setAudio(result.audio);
        if (result.audio && result.audio.status !== 'pending') return;
      } catch {
        // Transient: keep waiting until the deadline.
      }

      if (!cancelled) {
        window.setTimeout(poll, AUDIO_POLL_MS);
      }
    }

    const handle = window.setTimeout(poll, AUDIO_POLL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [analysisId, audio.status]);

  const src = mediaUrl(audio.url);

  useEffect(() => {
    return () => {
      ref.current?.pause();
    };
  }, []);

  async function retryStatus() {
    if (retryingStatus) return;
    setRetryingStatus(true);
    try {
      const result = await api.getResult(analysisId);
      if (result.audio) {
        setAudio(result.audio);
        setFailed(false);
      }
    } catch {
      // Keep the inline error notice visible if status refresh fails.
    } finally {
      setRetryingStatus(false);
    }
  }

  if (audio.status === 'pending') {
    return (
      <div className="audio audio--pending" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        <p className="audio__title">কণ্ঠে রূপ দেওয়া হচ্ছে...</p>
      </div>
    );
  }

  if (audio.status === 'failed') {
    return (
      <div className="audio audio--error" role="alert">
        <div className="audio__body">
          <p className="audio__title">অডিও তৈরি করা যায়নি</p>
          <p className="audio__meta">পরামর্শটি উপরে লিখে দেওয়া হয়েছে।</p>
        </div>
        <button
          type="button"
          className="audio__retry"
          disabled={retryingStatus}
          onClick={retryStatus}
        >
          <RotateCcw size={15} strokeWidth={2.2} aria-hidden="true" />
          {retryingStatus ? 'দেখা হচ্ছে...' : 'আবার চেষ্টা করুন'}
        </button>
      </div>
    );
  }

  if (audio.status !== 'ready' || !src) return null;

  async function toggle() {
    const element = ref.current;
    if (!element) return;
    if (element.paused) {
      try {
        setFailed(false);
        await element.play();
      } catch {
        setPlaying(false);
        setFailed(true);
      }
    } else {
      element.pause();
    }
  }

  async function retryPlay() {
    const element = ref.current;
    if (!element) return;
    setFailed(false);
    try {
      element.load();
      await element.play();
    } catch {
      setPlaying(false);
      setFailed(true);
    }
  }

  const audioElement = (
    <audio
      ref={ref}
      src={src}
      preload="none"
      onPlay={() => {
        setFailed(false);
        setPlaying(true);
      }}
      onPause={() => setPlaying(false)}
      onEnded={() => setPlaying(false)}
      onError={() => {
        setPlaying(false);
        setFailed(true);
      }}
    />
  );

  if (failed) {
    return (
      <div className="audio audio--error" role="alert">
        <div className="audio__body">
          <p className="audio__title">অডিও চালু করা যায়নি</p>
          <p className="audio__meta">সংযোগ বা ব্রাউজারের সমস্যায় শোনা যায়নি।</p>
        </div>
        <button type="button" className="audio__retry" onClick={retryPlay}>
          <RotateCcw size={15} strokeWidth={2.2} aria-hidden="true" />
          আবার চেষ্টা করুন
        </button>
        {audioElement}
      </div>
    );
  }

  return (
    <div className="audio">
      <button
        type="button"
        className="audio__button"
        onClick={toggle}
        aria-label={playing ? 'পড়া বন্ধ করুন' : 'পরামর্শটি শুনুন'}
      >
        {playing ? (
          <Pause size={20} strokeWidth={2.2} aria-hidden="true" />
        ) : (
          <Play size={20} strokeWidth={2.2} aria-hidden="true" />
        )}
      </button>

      <div className="audio__body">
        <p className="audio__title">{playing ? 'শোনা হচ্ছে...' : 'পরামর্শটি শুনুন'}</p>
        {audio.durationMs ? <p className="audio__meta">{formatDuration(audio.durationMs)}</p> : null}
      </div>

      {audioElement}
    </div>
  );
}
