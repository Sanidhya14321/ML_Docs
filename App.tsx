
import React, { useState, useEffect, Suspense } from 'react';
import { ViewSection } from './types';
import { CONTENT_REGISTRY } from './content/registry';
import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { SearchModal } from './src/components/shared/SearchModal';
import { SettingsModal } from './src/components/shared/SettingsModal';
import { BackToTop } from './components/BackToTop';
import { DocViewer } from './components/DocViewer';
import { LabWorkspace } from './components/LabWorkspace';
import { Dashboard } from './components/Dashboard';
import { QuizView } from './components/QuizView';
import { SitemapView } from './views/SitemapView';
import { CertificateView } from './views/CertificateView';
import { NewsFeedView } from './views/NewsFeedView';
import { getTopicById } from './lib/contentHelpers';
import { NAV_ITEMS } from './lib/navigation-data';
import { useCourseProgress } from './hooks/useCourseProgress';
import { useTheme } from './hooks/useTheme';
import { 
  BrainCircuit, Search, Command
} from 'lucide-react';

import { useKeyboardNavigation } from './hooks/useKeyboardNavigation';

// New Components
import { Header } from './components/Header';
import { MobileNav } from './components/MobileNav';
import { LoadingOverlay } from './components/LoadingOverlay';

import { AppShell } from './src/components/layout/AppShell';
import { useUIStore } from './src/stores/useUIStore';

const App: React.FC = () => {
  const [currentPath, setCurrentPath] = useState<string>(ViewSection.DASHBOARD);
  const { isSearchOpen, isSettingsOpen, toggleSearch, toggleSettings } = useUIStore();
  
  // Initialize Global Hooks
  const { setLastActiveTopic } = useCourseProgress();

  // Navigation Helper
  const navigateTo = (path: string) => {
    if (path === currentPath) return;
    window.location.hash = `#/${path}`;
    setCurrentPath(path);
    
    if (!path.startsWith('lab/')) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  // Keyboard Navigation
  useKeyboardNavigation(currentPath, navigateTo);

  // Handle Command+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        toggleSearch();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleSearch]);

  // Routing Logic
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#/', '');
      const path = hash || ViewSection.DASHBOARD;
      setCurrentPath(path);
      
      if (path !== ViewSection.DASHBOARD) {
        const topicId = path.startsWith('lab/') ? path.replace('lab/', '') : path;
        const topic = getTopicById(topicId);
        if (topic) {
           setLastActiveTopic(topic.id);
        }
      }
    };
    
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setLastActiveTopic]);

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
  else if (currentPath === ViewSection.NEWS_FEED) {
       contentElement = <NewsFeedView />;
  }
  else if (currentPath === ViewSection.CERTIFICATE) {
       contentElement = <CertificateView />;
  }
  else if (CONTENT_REGISTRY[currentPath]) {
       const InteractiveComponent = CONTENT_REGISTRY[currentPath];
       contentElement = <InteractiveComponent />;
  } 
  else {
       contentElement = <DocViewer topicId={currentPath} title={currentTopic?.title || 'Documentation'} />;
  }

  // Lab Mode Layout
  if (isLabMode) {
      return (
        <div className="h-screen bg-zinc-950 text-zinc-200 overflow-hidden font-sans">
             <Suspense fallback={<div className="h-full flex items-center justify-center bg-zinc-950"><LoadingOverlay message="Initializing Lab" subMessage="Provisioning Container..." /></div>}>
                {contentElement}
             </Suspense>
        </div>
      );
  }

  return (
    <AppShell currentPath={currentPath} onNavigate={navigateTo}>
      <Suspense fallback={<LoadingOverlay />}>
        {contentElement}
      </Suspense>

      {/* Global Overlays */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={toggleSettings} 
      />
      <SearchModal isOpen={isSearchOpen} onClose={toggleSearch} onNavigate={(slug) => navigateTo(slug as string)} />
      <BackToTop />
    </AppShell>
  );
};


export default App;
