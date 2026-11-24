import type { AnalysisStatus } from '@foshol-lens/contracts';

export const STATUS_LABELS: Record<AnalysisStatus, string> = {
  pending: 'অপেক্ষমাণ',
  processing: 'বিশ্লেষণ চলছে',
  completed: 'সম্পন্ন',
  failed: 'ব্যর্থ',
};
