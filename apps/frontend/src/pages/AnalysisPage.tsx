import type { AnalysisResult as AnalysisResultType } from '@foshol-lens/contracts';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { AnalysisResultView } from '../components/AnalysisResult';
import { api } from '../lib/api-client';

type State =
  | { kind: 'loading' }
  | { kind: 'ready'; result: AnalysisResultType }
  | { kind: 'error'; message: string };

export function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const [state, setState] = useState<State>({ kind: 'loading' });

  useEffect(() => {
    let cancelled = false;

    if (!id) {
      setState({ kind: 'error', message: 'এই বিশ্লেষণটি খুঁজে পাওয়া যায়নি।' });
      return;
    }

    api
      .getResult(id)
      .then((result) => {
        if (!cancelled) setState({ kind: 'ready', result });
      })
      .catch(() => {
        if (!cancelled) {
          setState({ kind: 'error', message: 'এই বিশ্লেষণটি খুঁজে পাওয়া যায়নি। আবার চেষ্টা করুন।' });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  if (state.kind === 'loading') {
    return (
      <div className="page">
        <h1 className="page-heading">বিশ্লেষণের ফলাফল</h1>
        <p className="status status--info" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          ফলাফল লোড হচ্ছে...
        </p>
      </div>
    );
  }

  if (state.kind === 'error') {
    return (
      <div className="page">
        <h1 className="page-heading">বিশ্লেষণের ফলাফল</h1>
        <div className="notice notice--error" style={{ marginTop: 20 }}>
          <div>
            <p className="notice__title">ফলাফল পাওয়া যায়নি</p>
            <p className="notice__text">{state.message}</p>
          </div>
        </div>
      </div>
    );
  }

  return <AnalysisResultView result={state.result} />;
}
