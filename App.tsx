
import React, { useState, useEffect, Suspense } from 'react';
import { ViewSection, NavigationItem } from './types';
import { CONTENT_REGISTRY } from './content/registry';
import { NAV_ITEMS } from './lib/navigation-data';
import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { LoadingOverlay } from './components/LoadingOverlay';
import { SearchModal } from './components/SearchModal';
import { TableOfContents } from './components/TableOfContents';
import { Sidebar } from './components/Sidebar';
import { Breadcrumbs } from './components/Breadcrumbs';
import { BackToTop } from './components/BackToTop';
import { DocViewer } from './components/DocViewer'; // Import DocViewer
import { SitemapView } from './views/SitemapView';
import { 
  Menu, X, Search, Command, BrainCircuit
} from 'lucide-react';

const App: React.FC = () => {
  const [currentPath, setCurrentPath] = useState<string>(ViewSection.FOUNDATIONS);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [activeLabel, setActiveLabel] = useState<string>('');

  // Scroll Progress Logic
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 100, damping: 30, restDelta: 0.001 });

  // Handle Command+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Helper to find label recursively
  const findLabel = (items: NavigationItem[], id: string): string => {
    for (const item of items) {
        if (item.id === id) return item.label;
        if (item.items) {
            const found = findLabel(item.items, id);
            if (found) return found;
        }
    }
    return '';
  };

  // Routing Logic
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#/', '');
      const path = hash || ViewSection.FOUNDATIONS;
      setCurrentPath(path);
      const label = findLabel(NAV_ITEMS, path);
      setActiveLabel(label || 'Documentation');
    };
    
    window.addEventListener('hashchange', handleHashChange);
    if (window.location.hash) handleHashChange(); 
    else {
        // Init label
        const initLabel = findLabel(NAV_ITEMS, ViewSection.FOUNDATIONS);
        setActiveLabel(initLabel);
    }
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigateTo = (path: string) => {
    window.location.hash = `#/${path}`;
    setCurrentPath(path);
    setIsMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Determine which component to render
  let contentElement: React.ReactNode;
  
  // 1. Check Registry (Interactive Modules)
  if (CONTENT_REGISTRY[currentPath]) {
      const InteractiveComponent = CONTENT_REGISTRY[currentPath];
      contentElement = <InteractiveComponent />;
  } 
  // 2. Check Sitemap Route
  else if (currentPath === ViewSection.SITEMAP) {
      contentElement = <SitemapView navItems={NAV_ITEMS} onNavigate={navigateTo} />;
  }
  // 3. Fallback to Generic Doc Viewer (The Engine)
  else {
      contentElement = <DocViewer topicId={currentPath} title={activeLabel} />;
  }

  return (
    <div className="h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans selection:bg-indigo-500/30">
      
      {/* 1. Global Reading Progress Bar */}
      <motion.div 
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 z-[110] origin-left shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
        style={{ scaleX }} 
      />

      {/* 2. Mobile Header */}
      <div className="md:hidden fixed top-0 w-full bg-slate-900/90 backdrop-blur-md border-b border-slate-800 z-50 flex items-center justify-between p-4">
        <span className="font-serif font-bold text-white tracking-tight">AI<span className="text-indigo-500">.</span>Codex</span>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 rounded-lg bg-slate-800 text-slate-300">
          {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* 3. Main Layout Grid */}
      <div className="flex h-full pt-16 md:pt-0">
        
        {/* LEFT COLUMN: Sidebar Navigation */}
        <aside className={`
          fixed md:relative z-40 h-full w-72 bg-slate-950/50 backdrop-blur-xl border-r border-slate-800/50 flex flex-col transition-transform duration-300
          ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}>
          {/* Sidebar Header */}
          <div className="p-6 border-b border-slate-800/50">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/20">
                <BrainCircuit size={18} className="text-white" />
              </div>
              <div>
                <h1 className="font-serif font-black text-lg text-white tracking-tighter">AI Codex</h1>
                <p className="text-[9px] text-slate-500 font-mono uppercase tracking-[0.3em]">v2.5.0</p>
              </div>
            </div>
            
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="w-full bg-slate-900 border border-slate-800 hover:border-indigo-500/50 p-2.5 rounded-lg flex items-center justify-between text-slate-500 transition-all group shadow-sm"
            >
              <div className="flex items-center gap-2">
                <Search size={14} className="group-hover:text-indigo-400" />
                <span className="text-[11px] font-bold">Quick find...</span>
              </div>
              <div className="flex items-center gap-1 opacity-50 group-hover:opacity-100">
                <Command size={10} />
                <span className="text-[10px] font-mono">K</span>
              </div>
            </button>
          </div>

          {/* Sidebar Content */}
          <div className="flex-1 overflow-y-auto custom-scrollbar pt-6">
            <Sidebar items={NAV_ITEMS} currentPath={currentPath} onNavigate={navigateTo} />
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-slate-800/50 text-[10px] text-slate-600 flex justify-between">
             <button onClick={() => navigateTo(ViewSection.SITEMAP)} className="hover:text-indigo-400 transition-colors">Site Map</button>
             <span>© 2024 AI Codex</span>
          </div>
        </aside>

        {/* CENTER COLUMN: Main Content */}
        <main className="flex-1 h-full overflow-y-auto scroll-smooth bg-[#020617] relative custom-scrollbar">
          {/* Background Ambience */}
          <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/5 rounded-full blur-[120px] pointer-events-none" />
          <div className="fixed bottom-[-5%] right-[-5%] w-[30%] h-[30%] bg-purple-600/5 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-[1000px] mx-auto p-8 md:p-12 lg:p-16 relative z-10 min-h-screen">
             {/* Breadcrumbs */}
             <Breadcrumbs currentPath={currentPath} navItems={NAV_ITEMS} onNavigate={navigateTo} />
             
             {/* Content Area with Fade Up Transition */}
             <Suspense fallback={<LoadingOverlay />}>
              <AnimatePresence mode="wait">
                <motion.div 
                  key={currentPath}
                  initial={{ opacity: 0, y: 20, filter: 'blur(10px)' }}
                  animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                  exit={{ opacity: 0, y: -20, filter: 'blur(10px)' }}
                  transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }} 
                >
                  {contentElement}
                </motion.div>
              </AnimatePresence>
            </Suspense>
          </div>
        </main>

        {/* RIGHT COLUMN: Table of Contents (Desktop Only) */}
        <aside className="hidden xl:block w-64 border-l border-slate-800/50 bg-[#020617]/50 backdrop-blur-sm p-8 h-full overflow-y-auto">
           {CONTENT_REGISTRY[currentPath] && <TableOfContents />}
        </aside>

      </div>

      <BackToTop />
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} onNavigate={(slug) => navigateTo(slug as string)} />
    </div>
  );
};

export default App;
