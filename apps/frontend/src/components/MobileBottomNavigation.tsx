import { History, House } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: 'হোম', icon: House, end: true },
  { to: '/history', label: 'ইতিহাস', icon: History, end: false },
] as const;

export function MobileBottomNavigation() {
  return (
    <nav className="bottom-nav" aria-label="প্রধান নেভিগেশন">
      <ul className="bottom-nav__list">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to}>
              <NavLink to={item.to} end={item.end} className="bottom-nav__link">
                <Icon size={22} strokeWidth={2} aria-hidden="true" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
