import type { AnalysisStage } from '@foshol-lens/contracts';
import { AlertCircle, CheckCircle } from 'lucide-react';

export type UploadPhase = 'idle' | 'uploading' | 'analyzing' | 'success' | 'error';

interface AnalysisLoadingStateProps {
  phase: UploadPhase;
  stage?: AnalysisStage;
  errorMessage?: string;
}

/** Stage-specific copy keeps a long wait legible. */
const STAGE_LABELS: Record<AnalysisStage, string> = {
  queued: 'অপেক্ষায় আছে...',
  reading: 'ছবি দেখে রোগ শনাক্ত করা হচ্ছে...',
  grounding: 'ফসলের তথ্য খোঁজা হচ্ছে...',
  writing: 'আপনার জন্য পরামর্শ লেখা হচ্ছে...',
  speaking: 'কণ্ঠে রূপ দেওয়া হচ্ছে...',
  done: 'কাজ শেষ হয়েছে',
};

export function AnalysisLoadingState({ phase, stage, errorMessage }: AnalysisLoadingStateProps) {
  if (phase === 'idle') return null;

  if (phase === 'error') {
    return (
      <p className="status status--error" role="alert">
        <AlertCircle size={18} strokeWidth={2.1} aria-hidden="true" />
        {errorMessage ?? 'ছবি আপলোড করা যায়নি'}
      </p>
    );
  }

  if (phase === 'success') {
    return (
      <p className="status status--success" role="status">
        <CheckCircle size={18} strokeWidth={2.1} aria-hidden="true" />
        বিশ্লেষণ সম্পন্ন হয়েছে
      </p>
    );
  }

  const label =
    phase === 'uploading' ? 'ছবি আপলোড হচ্ছে...' : (stage ? STAGE_LABELS[stage] : 'বিশ্লেষণ করা হচ্ছে...');

  return (
    <p className="status status--info" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label}
    </p>
  );
}
