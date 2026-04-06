'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Sidebar.module.css';

const navItems = [
  { name: 'Dashboard', path: '/' },
  { name: 'Upload Event', path: '/upload' },
  { name: 'Entity Search', path: '/search' },
  { name: 'System Logs', path: '/logs' },
  { name: 'Configuration', path: '/config' },
];

export default function Sidebar({ className }: { className?: string }) {
  const pathname = usePathname();

  return (
    <aside className={`${styles.sidebar} ${className || ''}`}>
      <div className={styles.header}>
        <div className={styles.logo}>
          <div className={styles.logoIcon}></div>
          <span className={styles.logoText}>SOC Analytics</span>
        </div>
      </div>
      
      <nav className={styles.nav}>
        <ul>
          {navItems.map((item) => {
            const isActive = pathname === item.path || (pathname.startsWith('/events') && item.path === '/');
            return (
              <li key={item.path} className={styles.navItem}>
                <Link
                  href={item.path}
                  className={`${styles.navLink} ${isActive ? styles.active : ''}`}
                >
                  {item.name}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      
      <div className={styles.footer}>
        <div className={styles.statusIndicator}>
          <div className={styles.statusDot}></div>
          <span>System Online</span>
        </div>
      </div>
    </aside>
  );
}
