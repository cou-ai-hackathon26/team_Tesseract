import type { AnalysisSummary } from '@foshol-lens/contracts';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { mediaUrl } from '../lib/api-client';
import { formatBengaliDate } from '../lib/format';
import { STATUS_LABELS } from '../lib/status';

export function HistoryItem({ item }: { item: AnalysisSummary }) {
  const thumb = mediaUrl(item.thumbnailUrl);

  return (
    <li>
      <Link to={`/analysis/${item.id}`} className="history-item">
        <span className="history-item__thumb">
          {thumb ? <img src={thumb} alt="" loading="lazy" /> : null}
        </span>
        <span className="history-item__body">
          <span className="history-item__crop">{item.cropName}</span>
          <span className="history-item__meta">
            <span>{formatBengaliDate(item.createdAt)}</span>
            <span className={`badge ${item.status === 'completed' ? 'badge--ok' : 'badge--warn'}`}>
              {STATUS_LABELS[item.status]}
            </span>
          </span>
          {item.diseaseLabel && <span className="history-item__disease">{item.diseaseLabel}</span>}
        </span>
        <ChevronRight className="history-item__chevron" size={19} aria-hidden="true" />
      </Link>
    </li>
  );
}
