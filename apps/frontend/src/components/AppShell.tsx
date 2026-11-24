import type { ReactNode } from 'react';
import { Brand } from './Brand';
import { DesktopNavigation } from './DesktopNavigation';
import { MobileBottomNavigation } from './MobileBottomNavigation';

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="container app-header__inner">
          <Brand />
          <DesktopNavigation />
        </div>
      </header>

      <main className="app-main">
        <div className="container">{children}</div>
      </main>

      <MobileBottomNavigation />
    </div>
  );
}
