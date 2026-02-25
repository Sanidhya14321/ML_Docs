import { useEffect } from 'react';
import { getNextTopic, getPrevTopic } from '../lib/contentHelpers';

export const useKeyboardNavigation = (currentPath: string, onNavigate: (path: string) => void) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input, textarea, or contenteditable
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName) || (e.target as HTMLElement).isContentEditable) return;
      
      // Ignore if modifier keys are pressed (except Shift)
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === 'j') {
        const next = getNextTopic(currentPath);
        if (next) onNavigate(next);
      } else if (e.key === 'k') {
        const prev = getPrevTopic(currentPath);
        if (prev) onNavigate(prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPath, onNavigate]);
};
