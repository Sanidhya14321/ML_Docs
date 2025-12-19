import React, { useState, useEffect } from 'react';
import { ViewSection, NavigationItem } from './types';
import { FoundationsView } from './views/FoundationsView';
import { RegressionView } from './views/RegressionView';
import { ClassificationView } from './views/ClassificationView';
import { UnsupervisedView } from './views/UnsupervisedView';
import { DeepLearningView } from './views/DeepLearningView';
import { ProjectLabView } from './views/ProjectLabView';
import { EnsembleView } from './views/EnsembleView';
import { ReinforcementView } from './views/ReinforcementView';
import { ModelComparisonView } from './views/ModelComparisonView';
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
  ChevronRight,
  Swords,
  Target
} from 'lucide-react';

const NAV_ITEMS: NavigationItem[] = [
  { 
    id: ViewSection.FOUNDATIONS, label: 'Foundations', icon: <BookOpen size={18} />,
    subItems: [
        { id: 'math-primer', label: 'Math Primer' },
        { id: 'python-core', label: 'Python Stack' },
        { id: 'learning-definition', label: 'What is Learning?' },
        { id: 'bias-variance', label: 'Bias-Variance' },
        { id: 'feature-engineering', label: 'Preprocessing' }
    ]
  },
  { id: ViewSection.REGRESSION, label: 'Supervised: Reg', icon: <TrendingUp size={18} /> },
  { id: ViewSection.CLASSIFICATION, label: 'Supervised: Class', icon: <GitBranch size={18} /> },
  { id: ViewSection.ENSEMBLE, label: 'Ensemble Models', icon: <Layers size={18} /> },
  { id: ViewSection.UNSUPERVISED, label: 'Unsupervised', icon: <Network size={18} /> },
  { id: ViewSection.DEEP_LEARNING, label: 'Deep Learning', icon: <BrainCircuit size={18} /> },
  { id: ViewSection.REINFORCEMENT, label: 'Reinforcement', icon: <Gamepad2 size={18} /> },
  { id: ViewSection.MODEL_COMPARISON, label: 'Battleground', icon: <Swords size={18} /> },
  { id: ViewSection.PROJECT_LAB, label: 'Project Lab', icon: <FlaskConical size={18} /> },
];

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewSection>(ViewSection.FOUNDATIONS);
  const [expandedSection, setExpandedSection] = useState<ViewSection | null>(ViewSection.FOUNDATIONS);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const main = document.querySelector('main');
      if (main) {
        const scrollPercent = (main.scrollTop / (main.scrollHeight - main.clientHeight)) * 100;
        setProgress(scrollPercent);
      }
    };
    const main = document.querySelector('main');
    main?.addEventListener('scroll', handleScroll);
    return () => main?.removeEventListener('scroll', handleScroll);
  }, [currentView]);

  const renderContent = () => {
    switch (currentView) {
      case ViewSection.FOUNDATIONS: return <FoundationsView />;
      case ViewSection.REGRESSION: return <RegressionView />;
      case ViewSection.CLASSIFICATION: return <ClassificationView />;
      case ViewSection.ENSEMBLE: return <EnsembleView />;
      case ViewSection.UNSUPERVISED: return <UnsupervisedView />;
      case ViewSection.DEEP_LEARNING: return <DeepLearningView />;
      case ViewSection.REINFORCEMENT: return <ReinforcementView />;
      case ViewSection.MODEL_COMPARISON: return <ModelComparisonView />;
      case ViewSection.PROJECT_LAB: return <ProjectLabView />;
      default: return <FoundationsView />;
    }
  };

  const handleNavClick = (section: ViewSection) => {
     setExpandedSection(expandedSection === section ? null : section);
     setCurrentView(section);
     setIsMobileMenuOpen(false);
     document.querySelector('main')?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSubItemClick = (section: ViewSection, id: string) => {
      setCurrentView(section);
      setIsMobileMenuOpen(false);
      setTimeout(() => {
          document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
      
      {/* Mobile Top Bar */}
      <div className="md:hidden fixed top-0 w-full bg-slate-900/90 backdrop-blur-md border-b border-slate-800 z-50 flex items-center justify-between p-4">
        <span className="font-serif font-bold text-white tracking-tight">Encyclopedia<span className="text-indigo-500">.</span>Algo</span>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 rounded-lg bg-slate-800">
          {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Sidebar Navigation */}
      <aside className={`
        fixed md:relative z-40 h-full w-72 bg-slate-900 border-r border-slate-800 flex-shrink-0 transition-transform duration-300 overflow-y-auto
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <div className="p-8 border-b border-slate-800 hidden md:block sticky top-0 bg-slate-900 z-10">
          <h1 className="font-serif font-black text-2xl text-white tracking-tighter">
            Encyclopedia<span className="text-indigo-500">.</span>ML
          </h1>
          <p className="text-[10px] text-slate-500 font-mono uppercase tracking-[0.3em] mt-2">v2.5 Master Hub</p>
          
          {/* Progress Bar */}
          <div className="mt-6 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
             <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${progress}%` }}></div>
          </div>
          <div className="flex justify-between mt-1 text-[8px] font-mono text-slate-600 uppercase tracking-widest">
             <span>Reading Progress</span>
             <span>{Math.round(progress)}%</span>
          </div>
        </div>
        
        <nav className="p-6 space-y-2 mt-16 md:mt-0 pb-12">
          {NAV_ITEMS.map((item) => (
            <div key={item.id} className="space-y-1">
                <button
                onClick={() => handleNavClick(item.id)}
                className={`
                    w-full flex items-center justify-between px-4 py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all duration-300
                    ${currentView === item.id 
                    ? 'bg-indigo-600 text-white shadow-2xl shadow-indigo-900/40 translate-x-1' 
                    : 'text-slate-500 hover:bg-slate-800 hover:text-slate-300'
                    }
                `}
                >
                <div className="flex items-center gap-3">
                    {item.icon}
                    {item.label}
                </div>
                {item.subItems && (
                    <span>{expandedSection === item.id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span>
                )}
                </button>
                
                {item.subItems && expandedSection === item.id && (
                    <div className="pl-10 space-y-1 animate-fade-in py-2">
                        {item.subItems.map((sub) => (
                            <button
                                key={sub.id}
                                onClick={() => handleSubItemClick(item.id, sub.id)}
                                className="block w-full text-left py-2 px-2 text-[10px] text-slate-500 hover:text-indigo-400 font-mono uppercase tracking-tighter border-l border-slate-800 hover:border-indigo-500 transition-all"
                            >
                                {sub.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>
          ))}
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 h-full overflow-y-auto scroll-smooth pt-20 md:pt-0">
        <div className="max-w-5xl mx-auto p-8 md:p-16 lg:p-20">
          <div className="animate-fade-in-up">
            {renderContent()}
          </div>
        </div>
      </main>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 md:hidden" onClick={() => setIsMobileMenuOpen(false)} />
      )}
    </div>
  );
};

export default App;