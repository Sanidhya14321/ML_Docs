
import React, { useState, useEffect, Suspense } from 'react';
import { ViewSection } from './types';
import { CONTENT_REGISTRY } from './content/registry';
import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { SearchModal } from './components/SearchModal';
import { TableOfContents } from './components/TableOfContents';
import { Sidebar } from './components/Sidebar';
import { BackToTop } from './components/BackToTop';
import { DocViewer } from './components/DocViewer';
import { LabWorkspace } from './components/LabWorkspace';
import { Dashboard } from './components/Dashboard';
import { QuizView } from './components/QuizView';
import { SitemapView } from './views/SitemapView';
import { CertificateView } from './views/CertificateView';
import { getTopicById } from './lib/contentHelpers';
import { NAV_ITEMS } from './lib/navigation-data';
import { useCourseProgress } from './hooks/useCourseProgress';
import { useTheme } from './hooks/useTheme';
import { 
  BrainCircuit, Search, Command, Sparkles
} from 'lucide-react';

// New Components
import { Header } from './components/Header';
import { MobileNav } from './components/MobileNav';
import { SettingsModal } from './components/SettingsModal';

// Sophisticated Loading State for Route Transitions
const PageLoader = () => (
  <motion.div 
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.5 }}
    className="min-h-[60vh] flex flex-col items-center justify-center relative"
  >
    <div className="relative w-24 h-24 mb-8">
      <motion.div 
        className="absolute inset-0 border-4 border-slate-200/20 dark:border-slate-800/50 rounded-full"
      />
      <motion.div 
        className="absolute inset-0 border-4 border-t-indigo-500 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
      />
      <motion.div 
        className="absolute inset-3 border-4 border-b-emerald-500 rounded-full"
        animate={{ rotate: -360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      />
      <motion.div 
        className="absolute inset-0 flex items-center justify-center"
        animate={{ scale: [0.9, 1.1, 0.9], opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
         <BrainCircuit size={28} className="text-slate-400 dark:text-slate-600" />
      </motion.div>
    </div>

    <div className="space-y-3 text-center z-10">
      <motion.h3 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-lg font-serif font-bold text-slate-700 dark:text-slate-200 flex items-center justify-center gap-2"
      >
        <Sparkles size={16} className="text-indigo-500" />
        Synthesizing Knowledge
      </motion.h3>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="flex flex-col items-center gap-2"
      >
        <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em]">
          Loading Neural Weights...
        </p>
        <div className="w-32 h-1 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden mt-1">
          <motion.div 
            className="h-full bg-indigo-500"
            animate={{ x: [-130, 130] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </motion.div>
    </div>
    
    {/* Background Glow Effect */}
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />
  </motion.div>
);

const App: React.FC = () => {
  const [currentPath, setCurrentPath] = useState<string>(ViewSection.DASHBOARD);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeLabel, setActiveLabel] = useState<string>('');
  
  // Initialize Global Hooks
  const { setLastActiveTopic } = useCourseProgress();
  const { theme } = useTheme(); // Initialize theme effect

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

  // Routing Logic
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#/', '');
      const path = hash || ViewSection.DASHBOARD;
      setCurrentPath(path);
      
      // Update Active Label and Progress Tracking
      if (path === ViewSection.DASHBOARD) {
        setActiveLabel('Dashboard');
      } else {
        const topicId = path.startsWith('lab/') ? path.replace('lab/', '') : path;
        const topic = getTopicById(topicId);
        setActiveLabel(topic?.title || 'Documentation');
        
        // Track last active for resume functionality
        if (topic) {
           setLastActiveTopic(topic.id);
        }
      }
    };
    
    window.addEventListener('hashchange', handleHashChange);
    // Initial load
    handleHashChange();
    
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setLastActiveTopic]);

  const navigateTo = (path: string) => {
    if (path === currentPath) return;
    window.location.hash = `#/${path}`;
    setCurrentPath(path);
    setIsMobileMenuOpen(false);
    
    if (!path.startsWith('lab/')) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  // Determine State
  const isLabMode = currentPath.startsWith('lab/');
  const labTopicId = isLabMode ? currentPath.replace('lab/', '') : '';
  const currentTopic = getTopicById(currentPath);
  const isQuizMode = currentTopic?.type === 'quiz';

  // Determine Content
  let contentElement: React.ReactNode;
  
  if (isLabMode) {
     contentElement = <LabWorkspace topicId={labTopicId} onBack={() => navigateTo(labTopicId)} />;
  }
  else if (isQuizMode) {
     contentElement = <QuizView topicId={currentPath} onBack={() => navigateTo(ViewSection.DASHBOARD)} onComplete={() => navigateTo(ViewSection.DASHBOARD)} />;
  }
  else if (currentPath === ViewSection.DASHBOARD) {
     contentElement = <Dashboard onNavigate={navigateTo} />;
  }
  else if (currentPath === ViewSection.SITEMAP) {
      contentElement = <SitemapView navItems={NAV_ITEMS} onNavigate={navigateTo} />;
  }
  else if (currentPath === ViewSection.CERTIFICATE) {
      contentElement = <CertificateView />;
  }
  else if (CONTENT_REGISTRY[currentPath]) {
      const InteractiveComponent = CONTENT_REGISTRY[currentPath];
      contentElement = <InteractiveComponent />;
  } 
  else {
      contentElement = <DocViewer topicId={currentPath} title={activeLabel} />;
  }

  // Lab Mode Layout
  if (isLabMode) {
      return (
        <div className="h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
             {contentElement}
        </div>
      );
  }

  // Standard Layout
  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-200 overflow-hidden font-sans selection:bg-indigo-500/30 transition-colors duration-300">
      
      {/* Scroll Progress Bar (Updated to Teal/Emerald) */}
      <motion.div 
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-teal-500 via-emerald-500 to-indigo-500 z-[110] origin-left shadow-[0_0_10px_rgba(20,184,166,0.5)]" 
        style={{ scaleX }} 
      />

      {/* Desktop Sidebar */}
      <aside className="hidden md:flex w-72 flex-col border-r border-slate-200 dark:border-slate-800/50 bg-white dark:bg-slate-950/50 backdrop-blur-xl relative z-40">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800/50">
           <button 
              onClick={() => navigateTo(ViewSection.DASHBOARD)}
              className="flex items-center gap-3 mb-6 w-full text-left group"
            >
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/20 group-hover:scale-105 transition-transform">
                <BrainCircuit size={18} className="text-white" />
              </div>
              <div>
                <h1 className="font-serif font-black text-lg text-slate-900 dark:text-white tracking-tighter">AI Codex</h1>
                <p className="text-[9px] text-slate-500 font-mono uppercase tracking-[0.3em]">v3.2.0</p>
              </div>
            </button>
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 p-2.5 rounded-lg flex items-center justify-between text-slate-500 transition-all group shadow-sm"
            >
              <div className="flex items-center gap-2">
                <Search size={14} className="group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
                <span className="text-[11px] font-bold">Quick find...</span>
              </div>
              <div className="flex items-center gap-1 opacity-50 group-hover:opacity-100">
                <Command size={10} />
                <span className="text-[10px] font-mono">K</span>
              </div>
            </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar pt-6">
          <Sidebar currentPath={currentPath} onNavigate={navigateTo} />
        </div>

        <div className="p-4 border-t border-slate-200 dark:border-slate-800/50 text-[10px] text-slate-600 flex justify-between">
            <button onClick={() => navigateTo(ViewSection.SITEMAP)} className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Site Map</button>
            <span>© 2024 AI Codex</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-white dark:bg-slate-950 relative">
         <Header 
            onMenuClick={() => setIsMobileMenuOpen(true)}
            onSearchClick={() => setIsSearchOpen(true)}
            onSettingsClick={() => setIsSettingsOpen(true)}
            currentPath={currentPath}
            navItems={NAV_ITEMS}
            onNavigate={navigateTo}
         />

         <main className="flex-1 overflow-y-auto scroll-smooth custom-scrollbar relative bg-slate-50/50 dark:bg-slate-950">
             {/* Gradients updated to Teal/Emerald for a cleaner look */}
             <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-teal-500/5 rounded-full blur-[120px] pointer-events-none" />
             <div className="fixed bottom-[-5%] right-[-5%] w-[30%] h-[30%] bg-indigo-500/5 rounded-full blur-[100px] pointer-events-none" />
             
             <div className="max-w-[1000px] mx-auto p-4 md:p-12 lg:p-16 relative z-10 min-h-screen">
                <Suspense fallback={<PageLoader />}>
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
      </div>

      {/* Desktop TOC Sidebar */}
      <aside className="hidden xl:block w-64 border-l border-slate-200 dark:border-slate-800/50 bg-white dark:bg-slate-950/50 backdrop-blur-sm p-8 h-full overflow-y-auto z-40">
          {CONTENT_REGISTRY[currentPath] && <TableOfContents />}
      </aside>

      {/* Global Overlays */}
      <MobileNav 
        isOpen={isMobileMenuOpen} 
        onClose={() => setIsMobileMenuOpen(false)} 
        currentPath={currentPath} 
        onNavigate={navigateTo} 
      />
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
      <BackToTop />
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} onNavigate={(slug) => navigateTo(slug as string)} />
    </div>
  );
};

export default App;
