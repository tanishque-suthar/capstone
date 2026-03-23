import type { Metadata } from 'next';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import styles from './layout.module.css';

export const metadata: Metadata = {
  title: 'Operations Center - Track 1',
  description: 'Video Surveillance & Analytics Pipeline Dashboard',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={styles.container}>
        <Sidebar className={styles.sidebar} />
        <main className={styles.mainContent}>
          {children}
        </main>
      </body>
    </html>
  );
}
