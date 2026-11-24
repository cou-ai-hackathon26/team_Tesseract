import { Pause, RotateCcw, Volume2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../lib/api-client';

interface SpeakCardButtonProps {
  text: string;
  label?: string;
  ariaLabel?: string;
}

type SpeakStatus = 'idle' | 'loading' | 'playing' | 'error';

/**
 * Compact per-card Bangla voice button backed by POST /api/v1/tts.
 *
 * Keeps the <audio> element mounted and retains the fetched Object URL so an
 * async-fetch autoplay rejection (such as on iOS Safari) can be retried
 * synchronously on the next tap.
 */
export function SpeakCardButton({ text, label = 'শুনুন', ariaLabel }: SpeakCardButtonProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);
  const cachedTextRef = useRef<string>('');
  const requestIdRef = useRef(0);

  const [status, setStatus] = useState<SpeakStatus>('idle');
  const [errorText, setErrorText] = useState<string | null>(null);

  useEffect(() => {
    setStatus('idle');
    setErrorText(null);
    return () => {
      requestIdRef.current += 1;
      const element = audioRef.current;
      if (element) {
        element.pause();
        element.removeAttribute('src');
      }
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
      cachedTextRef.current = '';
    };
  }, [text]);

  const trimmed = text.trim();
  if (!trimmed) return null;

  async function playCachedUrl(): Promise<void> {
    const element = audioRef.current;
    const url = urlRef.current;
    if (!element || !url) return;

    setErrorText(null);
    if (element.src !== url) {
      element.src = url;
    }
    try {
      await element.play();
    } catch {
      setStatus('error');
      setErrorText('অডিও চালু করা যায়নি');
    }
  }

  async function handleClick() {
    if (status === 'loading') return;

    if (status === 'playing') {
      audioRef.current?.pause();
      setStatus('idle');
      return;
    }

    if (urlRef.current && cachedTextRef.current === trimmed) {
      await playCachedUrl();
      return;
    }

    const requestId = ++requestIdRef.current;
    setStatus('loading');
    setErrorText(null);

    let blob: Blob;
    try {
      blob = await api.synthesizeSpeech(trimmed);
    } catch (error) {
      if (requestId !== requestIdRef.current) return;
      setStatus('error');
      if (error instanceof ApiError && error.status === 503) {
        setErrorText('ভয়েস সেবা এখন চালু নেই');
      } else if (error instanceof ApiError && error.code === 'network_error') {
        setErrorText('নেটওয়ার্ক সংযোগ নেই');
      } else {
        setErrorText('ভয়েস তৈরি করা যায়নি');
      }
      return;
    }

    if (requestId !== requestIdRef.current) return;

    const url = URL.createObjectURL(blob);
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
    }
    urlRef.current = url;
    cachedTextRef.current = trimmed;

    const element = audioRef.current;
    if (!element) return;
    element.src = url;

    try {
      await element.play();
    } catch {
      if (requestId !== requestIdRef.current) return;
      setStatus('error');
      setErrorText('অডিও প্রস্তুত, শুনতে আবার চাপুন');
    }
  }

  const buttonText =
    status === 'loading' ? 'লোড হচ্ছে...' : status === 'playing' ? 'থামান' : label;

  return (
    <div className="speak-btn-wrap">
      <button
        type="button"
        className={`speak-btn ${status === 'playing' ? 'speak-btn--playing' : ''}`}
        disabled={status === 'loading'}
        aria-label={ariaLabel ?? label}
        onClick={handleClick}
      >
        {status === 'loading' ? (
          <span className="spinner speak-btn__spinner" aria-hidden="true" />
        ) : status === 'playing' ? (
          <Pause size={14} strokeWidth={2.2} aria-hidden="true" />
        ) : (
          <Volume2 size={14} strokeWidth={2.2} aria-hidden="true" />
        )}
        <span>{buttonText}</span>
      </button>

      {status === 'error' && errorText && (
        <div className="speak-btn__error" role="alert">
          <span>{errorText}</span>
          <button type="button" className="speak-btn__retry" onClick={handleClick}>
            <RotateCcw size={12} strokeWidth={2.2} aria-hidden="true" />
            আবার চেষ্টা করুন
          </button>
        </div>
      )}

      <audio
        ref={audioRef}
        preload="none"
        onPlay={() => {
          setErrorText(null);
          setStatus('playing');
        }}
        onPause={() => {
          setStatus((prev) => (prev === 'playing' ? 'idle' : prev));
        }}
        onEnded={() => setStatus('idle')}
        onError={() => {
          if (audioRef.current?.getAttribute('src')) {
            setStatus('error');
            setErrorText('অডিও চালু করা যায়নি');
          }
        }}
      />
    </div>
  );
}
