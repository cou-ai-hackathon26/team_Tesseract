import { Sprout } from 'lucide-react';
import { Link } from 'react-router-dom';

export function Brand() {
  return (
    <Link to="/" className="brand" aria-label="ফসল লেন্স — হোম">
      <span className="brand__mark" aria-hidden="true">
        <Sprout size={22} strokeWidth={2.2} />
      </span>
      <span className="brand__text">
        <span className="brand__name">ফসল লেন্স</span>
        <span className="brand__tagline">আপনার কৃষি সহায়ক</span>
      </span>
    </Link>
  );
}
