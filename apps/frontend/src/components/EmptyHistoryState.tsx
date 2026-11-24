import { Leaf } from 'lucide-react';
import { Link } from 'react-router-dom';

export function EmptyHistoryState() {
  return (
    <div className="empty-state">
      <span className="empty-state__icon" aria-hidden="true">
        <Leaf size={30} strokeWidth={1.8} />
      </span>
      <h2 className="empty-state__title">এখনও কোনো বিশ্লেষণ নেই</h2>
      <p className="empty-state__text">
        আপনার ফসলের ছবি দিয়ে প্রথম বিশ্লেষণ শুরু করুন। এখানে আপনার আগের সব বিশ্লেষণ সংরক্ষিত থাকবে।
      </p>
      <Link to="/" className="button-secondary empty-state__action">
        হোমে ফিরে যান
      </Link>
    </div>
  );
}
