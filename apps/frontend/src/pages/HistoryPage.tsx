import type { AnalysisSummary } from '@foshol-lens/contracts';
import { useEffect, useState } from 'react';
import { EmptyHistoryState } from '../components/EmptyHistoryState';
import { HistoryItem } from '../components/HistoryItem';
import { api } from '../lib/api-client';

export function HistoryPage() {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .listAnalyses()
      .then((next) => {
        if (!cancelled) setItems(next);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="page">
      <h1 className="page-heading">বিশ্লেষণের ইতিহাস</h1>
      <p className="page-subtitle">আপনার আগের ফসল বিশ্লেষণ.</p>

      {failed && (
        <div className="notice notice--error" style={{ marginTop: 20 }}>
          <div>
            <p className="notice__title">ইতিহাস লোড করা যায়নি</p>
            <p className="notice__text">আবার চেষ্টা করুন।</p>
          </div>
        </div>
      )}

      {items === null && !failed && (
        <p className="status status--info" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          ইতিহাস লোড হচ্ছে...
        </p>
      )}

      {items !== null && items.length === 0 && <EmptyHistoryState />}

      {items !== null && items.length > 0 && (
        <ul className="history-list">
          {items.map((item) => (
            <HistoryItem key={item.id} item={item} />
          ))}
        </ul>
      )}
    </div>
  );
}
