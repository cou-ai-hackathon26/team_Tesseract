import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: 'হোম', end: true },
  { to: '/history', label: 'ইতিহাস', end: false },
] as const;

export function DesktopNavigation() {
  return (
    <nav className="desktop-nav" aria-label="প্রধান নেভিগেশন">
      {NAV_ITEMS.map((item) => (
        <NavLink key={item.to} to={item.to} end={item.end} className="desktop-nav__link">
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}
