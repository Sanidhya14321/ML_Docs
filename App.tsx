
import React, { useState, useEffect, useMemo, Suspense, lazy } from 'react';
import { ViewSection, NavigationItem } from './types';
import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { LoadingOverlay } from './components/LoadingOverlay';
import { SearchModal } from './components/SearchModal';
import { TableOfContents } from './components/TableOfContents';
import { 
  BookOpen, 
  TrendingUp, 
  GitBranch, 
  Network, 
  BrainCircuit, 
  FlaskConical, 
  Menu,
  X,
  Layers,
  Gamepad2,
  ChevronDown,
  Swords,
  Zap,
  Search,
  Command,
  Layout
} from 'lucide-react';

// Specialized Lazy Load with View Registry
const Views = {
  [ViewSection.FOUNDATIONS]: lazy(() => import('./views/FoundationsView').then(m => ({ default: m.FoundationsView }))),
  [ViewSection.OPTIMIZATION]: lazy(() => import('./views/OptimizationView').then(m => ({ default: m.OptimizationView }))),
  [ViewSection.REGRESSION]: lazy(() => import('./views/RegressionView').then(m => ({ default: m.RegressionView }))),
  [ViewSection.CLASSIFICATION]: lazy(() => import('./views/ClassificationView').then(m => ({ default: m.ClassificationView }))),
  [ViewSection.ENSEMBLE]: lazy(() => import('./views/EnsembleView').then(m => ({ default: m.EnsembleView }))),
  [ViewSection.UNSUPERVISED]: lazy(() => import('./views/UnsupervisedView').then(m => ({ default: m.UnsupervisedView }))),
  [ViewSection.DEEP_LEARNING]: lazy(() => import('./views/DeepLearningView').then(m => ({ default: m.DeepLearningView }))),
  [ViewSection.REINFORCEMENT]: lazy(() => import('./views/ReinforcementView').then(m => ({ default: m.ReinforcementView }))),
  [ViewSection.MODEL_COMPARISON]: lazy(() => import('./views/ModelComparisonView').then(m => ({ default: m.ModelComparisonView }))),
  [ViewSection.PROJECT_LAB]: lazy(() => import('./views/ProjectLabView').then(m => ({ default: m.ProjectLabView }))),
};

const NAV_ITEMS: NavigationItem[] = [
  { id: ViewSection.FOUNDATIONS, label: 'Math Foundations', icon: <BookOpen size={18} />, category: 'Core' },
  { id: ViewSection.OPTIMIZATION, label: 'The Optimization Engine', icon: <Zap size={18} />, category: 'Core' },
  { id: ViewSection.REGRESSION, label: 'Supervised: Regression', icon: <TrendingUp size={18} />, category: 'Supervised' },
  { id: ViewSection.CLASSIFICATION, label: 'Supervised: Classification', icon: <GitBranch size={18} />, category: 'Supervised' },
  { id: ViewSection.ENSEMBLE, label: 'Ensemble Learning', icon: <Layers size={18} />, category: 'Supervised' },
  { id: ViewSection.UNSUPERVISED, label: 'Unsupervised Logic', icon: <Network size={18} />, category: 'Advanced' },
  { id: ViewSection.DEEP_LEARNING, label: 'Deep Neural Networks', icon: <BrainCircuit size={18} />, category: 'Advanced' },
  { id: ViewSection.REINFORCEMENT, label: 'Reinforcement Learning', icon: <Gamepad2 size={18} />, category: 'Advanced' },
  { id: ViewSection.MODEL_COMPARISON, label: 'Algorithm Battleground', icon: <Swords size={18} />, category: 'Lab' },
  { id: ViewSection.PROJECT_LAB, label: 'Clinical Case Lab', icon: <FlaskConical size={18} />, category: 'Lab' },
];

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewSection>(ViewSection.FOUNDATIONS);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  // Sync with Hash Routing for Deep Linking
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#/', '');
      const validSection = Object.values(ViewSection).find(v => v === hash);
      if (validSection) setCurrentView(validSection as ViewSection);
    };
    
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange(); // Initial load
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Scroll Progress
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 100, damping: 30, restDelta: 0.001 });

  const navigateTo = (section: ViewSection) => {
    window.location.hash = `#/${section}`;
    setCurrentView(section);
    setIsMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const ActiveView = Views[currentView];

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans">
      
      {/* Top Reading Progress Bar */}
      <motion.div 
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 z-[110] origin-left" 
        style={{ scaleX }} 
      />

      {/* Mobile Bar */}
      <div className="md:hidden fixed top-0 w-full bg-slate-900/90 backdrop-blur-md border-b border-slate-800 z-50 flex items-center justify-between p-4">
        <span className="font-serif font-bold text-white tracking-tight">AI<span className="text-indigo-500">.</span>Codex</span>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 rounded-lg bg-slate-800 text-slate-300">
          {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Sidebar Architecture */}
      <aside className={`
        fixed md:relative z-40 h-full w-80 bg-slate-900/50 backdrop-blur-2xl border-r border-slate-800/50 flex-shrink-0 transition-transform duration-300
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <div className="p-8 border-b border-slate-800/50">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/20">
               <BrainCircuit size={20} className="text-white" />
            </div>
            <div>
              <h1 className="font-serif font-black text-xl text-white tracking-tighter">AI Codex</h1>
              <p className="text-[8px] text-slate-500 font-mono uppercase tracking-[0.4em]">Advanced Mastery</p>
            </div>
          </div>
          
          <button 
            onClick={() => setIsSearchOpen(true)}
            className="w-full bg-slate-950 border border-slate-800 hover:border-indigo-500/50 p-3 rounded-xl flex items-center justify-between text-slate-500 transition-all group"
          >
            <div className="flex items-center gap-3">
              <Search size={14} className="group-hover:text-indigo-400" />
              <span className="text-[11px] font-bold">Quick find...</span>
            </div>
            <div className="flex items-center gap-1 opacity-50 group-hover:opacity-100 transition-opacity">
              <Command size={10} />
              <span className="text-[10px] font-mono">K</span>
            </div>
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto h-[calc(100%-160px)] custom-scrollbar">
          {['Core', 'Supervised', 'Advanced', 'Lab'].map(category => (
            <div key={category} className="mb-8">
              <h3 className="px-4 text-[9px] font-black text-slate-600 uppercase tracking-[0.3em] mb-3">{category}</h3>
              <nav className="space-y-1">
                {NAV_ITEMS.filter(item => item.category === category).map((item) => (
                  <button
                    key={item.id}
                    onClick={() => navigateTo(item.id)}
                    className={`
                      w-full flex items-center gap-3 px-4 py-3 rounded-xl text-[11px] font-bold uppercase tracking-widest transition-all relative group
                      ${currentView === item.id ? 'text-white' : 'text-slate-400 hover:text-slate-200'}
                    `}
                  >
                    <span className={currentView === item.id ? "text-indigo-400" : "text-slate-600 group-hover:text-slate-400"}>
                      {item.icon}
                    </span>
                    {item.label}
                    {currentView === item.id && (
                      <motion.div layoutId="nav-bg" className="absolute inset-0 bg-indigo-600/10 border border-indigo-500/20 rounded-xl z-[-1]" />
                    )}
                  </button>
                ))}
              </nav>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Documentation Viewer */}
      <main className="flex-1 h-full overflow-y-auto scroll-smooth bg-[#020617] relative custom-scrollbar selection:bg-indigo-500/30">
        {/* Background Gradients */}
        <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/5 rounded-full blur-[120px] pointer-events-none" />
        <div className="fixed bottom-[-5%] right-[-5%] w-[30%] h-[30%] bg-purple-600/5 rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-4xl mx-auto p-8 md:p-16 lg:p-24 relative z-10 min-h-screen">
          <Suspense fallback={<LoadingOverlay />}>
            <AnimatePresence mode="wait">
              <motion.div 
                key={currentView}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <ActiveView />
              </motion.div>
            </AnimatePresence>
          </Suspense>
        </div>

        {/* Intelligence Helpers */}
        <TableOfContents />
        <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} onNavigate={navigateTo} />
      </main>
    </div>
  );
};

export default App;
