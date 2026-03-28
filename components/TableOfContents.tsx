
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { List, ChevronRight } from 'lucide-react';

interface ToCItem {
  id: string;
  label: string;
  level: number;
}

export const TableOfContents: React.FC = () => {
  const [items, setItems] = useState<ToCItem[]>([]);
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    // Scan the DOM for headings within the main content area
    const updateToC = () => {
      const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
        .filter(h => h.id)
        .map(h => ({
          id: h.id,
          label: h.textContent || '',
          level: parseInt(h.tagName[1])
        }));
      setItems(headings);
    };

    // Use a small timeout to allow view transition to complete
    const timeoutId = setTimeout(updateToC, 500);

    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;
      // Get all headings from the DOM to find the one currently in view
      const docHeadings = Array.from(document.querySelectorAll('h1, h2, h3'))
        .filter(h => !!h.id);

      for (let i = docHeadings.length - 1; i >= 0; i--) {
        const h = docHeadings[i] as HTMLElement;
        if (h && h.offsetTop <= scrollPosition) {
          setActiveId(h.id);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  if (items.length === 0) return null;

  return (
    <div className="hidden xl:block fixed top-32 right-8 w-64 p-6 border-l border-border-strong space-y-4">
      <div className="flex items-center gap-2 text-[10px] font-mono font-black text-text-muted uppercase tracking-[0.2em] mb-4">
        <List size={12} /> On This Page
      </div>
      <nav className="space-y-3">
        {items.map((item) => (
          <motion.a
            key={item.id}
            href={`#${item.id}`}
            className={`
              block text-[11px] font-mono font-medium transition-all hover:text-brand uppercase tracking-wider
              ${item.level === 3 ? 'pl-4 text-text-muted' : 'text-text-secondary'}
              ${activeId === item.id ? 'text-brand font-black translate-x-1' : ''}
            `}
            whileHover={{ x: 2 }}
          >
            <div className="flex items-center gap-2">
               {activeId === item.id && <motion.div layoutId="active-dot" className="w-1.5 h-1.5 rounded-none bg-brand shadow-[0_0_8px_var(--color-brand)]" />}
               {item.label}
            </div>
          </motion.a>
        ))}
      </nav>
      
      <div className="mt-8 pt-8 border-t border-border-strong">
        <div className="p-4 rounded-none bg-brand/5 border border-brand/10">
          <p className="text-[10px] text-brand italic leading-relaxed font-serif">
            "Everything is theoretically impossible until it is done."
          </p>
          <p className="text-[9px] text-text-muted mt-2 font-mono uppercase">— Robert Heinlein</p>
        </div>
      </div>
    </div>
  );
};
