import type { AnalysisStage } from '@foshol-lens/contracts';
import { LocateFixed, Mic, ScanSearch, Square, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../lib/api-client';
import { toBengaliDigits } from '../lib/format';
import { prepareImage } from '../lib/image';
import { findNearestDistrict, useDistrict } from '../lib/use-district';
import { AnalysisLoadingState, type UploadPhase } from './AnalysisLoadingState';
import { ImageDropzone } from './ImageDropzone';
import { ImagePreview } from './ImagePreview';

const MAX_BYTES = 8 * 1024 * 1024;
const MAX_RECORDING_SECONDS = 12;
const MAX_QUESTION_CHARS = 300;
const RECORDING_MIMES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
] as const;

function pickRecorderMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return undefined;
  }
  return RECORDING_MIMES.find((mime) => MediaRecorder.isTypeSupported(mime));
}

const CROPS = [
  { id: 'rice', name: 'ধান' },
  { id: 'potato', name: 'আলু' },
] as const;

/**
 * Turn a failure into something a farmer can act on.
 *
 * "আবার চেষ্টা করুন" is only honest if retrying could work. A stopped API is
 * not a bad photo, and telling someone to retake their picture when the server
 * is simply down wastes their time and data.
 */
function describeFailure(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === 'network_error') {
      return 'সার্ভারে পৌঁছানো যাচ্ছে না। ইন্টারনেট সংযোগ দেখুন, তারপর আবার চেষ্টা করুন';
    }
    if (error.code === 'timeout') {
      return 'বিশ্লেষণ শেষ হয়নি। একটু পরে আবার চেষ্টা করুন';
    }
    if (error.code === 'analysis_failed') {
      return 'ছবিটি বিশ্লেষণ করা যায়নি। আরও পরিষ্কার ছবি দিয়ে আবার চেষ্টা করুন';
    }
    if (error.code === 'audio_too_large') {
      return 'ভয়েস রেকর্ডিং ৫ মেগাবাইটের কম হতে হবে';
    }
    if (error.code === 'unsupported_audio_type') {
      return 'এই অডিও ফরম্যাটটি সমর্থিত নয়';
    }
    if (error.status === 413) {
      return 'ছবির আকার বেশি বড়। ছোট ছবি দিয়ে আবার চেষ্টা করুন';
    }
    if (error.status >= 500) {
      return 'সার্ভারে সমস্যা হচ্ছে। কিছুক্ষণ পরে আবার চেষ্টা করুন';
    }
  }
  return 'বিশ্লেষণ করা যায়নি। আবার চেষ্টা করুন';
}

function validateFile(file: File): string | null {
  if (!file.type.startsWith('image/')) {
    return 'শুধুমাত্র ছবি আপলোড করুন';
  }
  if (file.size > MAX_BYTES) {
    return 'ছবির আকার ৮ মেগাবাইটের কম হতে হবে';
  }
  return null;
}

export function CropUploadCard() {
  const navigate = useNavigate();
  const { district, districts, choose } = useDistrict();
  const districtSelectId = useId();

  const [file, setFile] = useState<File | null>(null);
  const [crop, setCrop] = useState<string>('rice');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [stage, setStage] = useState<AnalysisStage | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [questionText, setQuestionText] = useState('');
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [micNote, setMicNote] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const [gpsNote, setGpsNote] = useState<string | null>(null);
  const [gpsCoords, setGpsCoords] = useState<{ latitude: number; longitude: number } | null>(null);

  // Object URL for the on-screen preview, revoked on replace/unmount.
  const previewRef = useRef<string | null>(null);
  const audioPreviewRef = useRef<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  // Guards against double-submits from rapid taps/keyboard repeat.
  const submittingRef = useRef(false);

  const stopStream = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaStreamRef.current) {
      for (const track of mediaStreamRef.current.getTracks()) {
        track.stop();
      }
      mediaStreamRef.current = null;
    }
  }, []);

  const clearAudio = useCallback(() => {
    if (audioPreviewRef.current) {
      URL.revokeObjectURL(audioPreviewRef.current);
      audioPreviewRef.current = null;
    }
    setAudioPreviewUrl(null);
    setAudioBlob(null);
    setRecordingSeconds(0);
  }, []);

  const clearPreview = useCallback(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => clearPreview, [clearPreview]);
  useEffect(() => {
    return () => {
      stopStream();
      clearAudio();
    };
  }, [stopStream, clearAudio]);

  function handleFile(nextFile: File) {
    if (submittingRef.current) return;

    const problem = validateFile(nextFile);
    if (problem) {
      setErrorMessage(problem);
      setPhase('error');
      return;
    }

    clearPreview();
    previewRef.current = URL.createObjectURL(nextFile);
    setPreviewUrl(previewRef.current);

    setFile(nextFile);
    setErrorMessage(undefined);
    setStage(undefined);
    setPhase('idle');
  }

  function handleReplace() {
    if (submittingRef.current) return;
    clearPreview();
    setFile(null);
    setPhase('idle');
    setStage(undefined);
    setErrorMessage(undefined);
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }
    stopStream();
    setRecording(false);
  }

  async function toggleRecording() {
    if (busy) return;
    if (recording) {
      stopRecording();
      return;
    }

    if (!window.isSecureContext) {
      setMicNote('ভয়েস রেকর্ডিংয়ের জন্য নিরাপদ (HTTPS) সংযোগ প্রয়োজন; নিচে লিখে প্রশ্ন করুন।');
      return;
    }

    if (
      typeof MediaRecorder === 'undefined' ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== 'function'
    ) {
      setMicNote('এই ব্রাউজারে ভয়েস রেকর্ডিং সমর্থিত নয়; নিচে লিখে প্রশ্ন করুন।');
      return;
    }

    setMicNote(null);
    clearAudio();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const mimeType = pickRecorderMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      const chunks: BlobPart[] = [];
      let hadRecorderError = false;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onerror = () => {
        hadRecorderError = true;
        stopStream();
        setRecording(false);
        setMicNote('ভয়েস রেকর্ডিংয়ে সমস্যা হয়েছে; আবার চেষ্টা করুন অথবা নিচে লিখে প্রশ্ন করুন।');
      };
      recorder.onstop = () => {
        stopStream();
        setRecording(false);
        if (hadRecorderError) return;
        const finalType = recorder.mimeType || mimeType || 'audio/webm';
        const recorded = new Blob(chunks, { type: finalType });
        if (recorded.size === 0) {
          setMicNote('কোনো অডিও রেকর্ড হয়নি; আবার চেষ্টা করুন অথবা নিচে লিখে প্রশ্ন করুন।');
          return;
        }
        const url = URL.createObjectURL(recorded);
        audioPreviewRef.current = url;
        setAudioBlob(recorded);
        setAudioPreviewUrl(url);
      };

      recorder.start();
      setRecording(true);
      setRecordingSeconds(0);

      const startedAt = Date.now();
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        setRecordingSeconds(Math.min(elapsed, MAX_RECORDING_SECONDS));
        if (elapsed >= MAX_RECORDING_SECONDS && mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }, 250);
    } catch (error) {
      stopStream();
      setRecording(false);
      const errorName =
        error instanceof DOMException || error instanceof Error ? error.name : '';
      if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError') {
        setMicNote(
          'মাইক্রোফোন ব্যবহারের অনুমতি দেওয়া হয়নি; ব্রাউজার সেটিংস থেকে অনুমতি দিন অথবা নিচে লিখে প্রশ্ন করুন।',
        );
      } else if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError') {
        setMicNote('কোনো মাইক্রোফোন পাওয়া যায়নি; নিচে লিখে প্রশ্ন করুন।');
      } else {
        setMicNote('মাইক্রোফোন চালু করা যায়নি; আবার চেষ্টা করুন অথবা নিচে লিখে প্রশ্ন করুন।');
      }
    }
  }

  function handleLocateDistrict() {
    if (busy || locating) return;
    if (districts.length === 0) {
      setGpsNote('জেলার তালিকা লোড হচ্ছে, অনুগ্রহ করে হাতে নির্বাচন করুন।');
      return;
    }
    if (!navigator.geolocation || typeof navigator.geolocation.getCurrentPosition !== 'function') {
      setGpsNote('এই ব্রাউজারে অবস্থান নির্ণয় সমর্থিত নয়; হাতে জেলা বেছে নিন।');
      return;
    }

    setLocating(true);
    setGpsNote(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocating(false);
        const { latitude, longitude } = position.coords;
        setGpsCoords({ latitude, longitude });
        const nearest = findNearestDistrict(latitude, longitude, districts);
        if (nearest) {
          choose(nearest.id);
          setGpsNote(`অবস্থান অনুযায়ী নির্বাচিত: ${nearest.name}`);
        } else {
          setGpsNote('নিকটস্থ জেলা নির্ণয় করা যায়নি; হাতে নির্বাচন করুন।');
        }
      },
      () => {
        setLocating(false);
        setGpsCoords(null);
        setGpsNote('অবস্থান জানার অনুমতি পাওয়া যায়নি; তালিকা থেকে জেলা বেছে নিন।');
      },
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300_000 },
    );
  }

  async function handleAnalyze() {
    if (!file || submittingRef.current || recording) return;

    submittingRef.current = true;
    setPhase('uploading');
    setStage(undefined);
    setErrorMessage(undefined);

    let preparedPreview: string | null = null;

    try {
      const prepared = await prepareImage(file);
      preparedPreview = prepared.previewUrl;

      setPhase('analyzing');
      setStage('queued');

      const created = await api.createAnalysis({
        imageBlob: prepared.blob,
        crop,
        district: district || null,
        latitude: gpsCoords?.latitude ?? null,
        longitude: gpsCoords?.longitude ?? null,
        questionText,
        audioBlob,
      });

      const result = await api.waitForResult(created.id, (nextStage) => setStage(nextStage));

      setPhase('success');
      navigate(`/analysis/${result.id}`);
    } catch (error) {
      setPhase('error');
      setErrorMessage(describeFailure(error));
    } finally {
      // The preview we created for upload is not the one on screen; release it.
      if (preparedPreview) URL.revokeObjectURL(preparedPreview);
      submittingRef.current = false;
    }
  }

  const busy = phase === 'uploading' || phase === 'analyzing';
  const hasImage = Boolean(file && previewUrl);

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <h2 id="upload-title" className="upload-card__title">
        ফসলের ছবি আপলোড করুন
      </h2>
      <p className="upload-card__hint">আপনার ফসলের পরিষ্কার ছবি দিন</p>

      {hasImage && previewUrl ? (
        <ImagePreview
          src={previewUrl}
          alt="নির্বাচিত ফসলের ছবি"
          disabled={busy}
          onReplace={handleReplace}
        />
      ) : (
        <ImageDropzone onFile={handleFile} disabled={busy} />
      )}

      <div className="field-row">
        <label className="field">
          <span className="field__label">ফসল</span>
          <select
            className="field__input"
            value={crop}
            disabled={busy}
            onChange={(event) => setCrop(event.target.value)}
          >
            {CROPS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </label>

        <div className="field">
          <div className="field__label-row">
            <label htmlFor={districtSelectId} className="field__label">
              জেলা
            </label>
            <button
              type="button"
              className="gps-btn"
              disabled={busy || locating}
              onClick={handleLocateDistrict}
              title="বর্তমান অবস্থান থেকে নিকটতম জেলা বেছে নিন"
            >
              <LocateFixed size={13} strokeWidth={2.2} aria-hidden="true" />
              {locating ? 'খোঁজা হচ্ছে...' : 'আমার অবস্থান (GPS)'}
            </button>
          </div>
          <select
            id={districtSelectId}
            className="field__input"
            value={district}
            disabled={busy}
            onChange={(event) => {
              setGpsCoords(null);
              setGpsNote(null);
              choose(event.target.value);
            }}
          >
            <option value="">নির্বাচন করুন</option>
            {districts.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {gpsNote && <p className="gps-note">{gpsNote}</p>}
      {gpsCoords && (
        <p className="gps-badge" role="status">
          সঠিক GPS অবস্থান ব্যবহৃত হচ্ছে
        </p>
      )}

      <div className="question-box">
        <div className="question-box__header">
          <span className="field__label">ঐচ্ছিক প্রশ্ন (ভয়েস বা টেক্সট)</span>
          {recording && (
            <span className="question-box__timer" aria-live="polite">
              <span className="record-dot" aria-hidden="true" />
              {toBengaliDigits(recordingSeconds)} / {toBengaliDigits(MAX_RECORDING_SECONDS)} সে.
            </span>
          )}
        </div>

        <div className="question-box__controls">
          <button
            type="button"
            className={`record-btn ${recording ? 'record-btn--active' : ''}`}
            disabled={busy}
            onClick={toggleRecording}
          >
            {recording ? (
              <Square size={15} strokeWidth={2.2} aria-hidden="true" />
            ) : (
              <Mic size={16} strokeWidth={2.1} aria-hidden="true" />
            )}
            {recording ? 'রেকর্ডিং থামান' : audioBlob ? 'নতুন করে রেকর্ড করুন' : 'ভয়েসে প্রশ্ন করুন'}
          </button>

          <input
            type="text"
            className="field__input question-box__text"
            value={questionText}
            disabled={busy}
            maxLength={MAX_QUESTION_CHARS}
            onChange={(event) => setQuestionText(event.target.value)}
            placeholder="যেমন: আজ বৃষ্টি হলে কি স্প্রে করব?"
            aria-label="ঐচ্ছিক প্রশ্ন লিখুন"
          />
        </div>

        {audioPreviewUrl && !recording && (
          <div className="question-audio">
            <audio controls src={audioPreviewUrl} className="question-audio__player" />
            <button
              type="button"
              className="question-audio__clear"
              disabled={busy}
              onClick={clearAudio}
              aria-label="রেকর্ডিং মুছুন"
            >
              <Trash2 size={15} strokeWidth={2.1} aria-hidden="true" />
              মুছুন
            </button>
          </div>
        )}

        {micNote && <p className="question-box__note">{micNote}</p>}
      </div>

      <button
        type="button"
        className="button-primary"
        disabled={!file || busy || recording}
        onClick={handleAnalyze}
      >
        {busy ? (
          <span className="spinner" aria-hidden="true" />
        ) : (
          <ScanSearch size={19} strokeWidth={2.1} aria-hidden="true" />
        )}
        {busy ? 'অপেক্ষা করুন...' : 'ছবি বিশ্লেষণ করুন'}
      </button>

      <AnalysisLoadingState phase={phase} stage={stage} errorMessage={errorMessage} />
    </section>
  );
}
