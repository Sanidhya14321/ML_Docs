import React, { useState } from 'react';
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
  Swords
} from 'lucide-react';

// Navigation Structure
const NAV_ITEMS: NavigationItem[] = [
  { 
    id: ViewSection.FOUNDATIONS, 
    label: 'Foundations', 
    icon: <BookOpen size={20} />,
    subItems: [
        { id: 'math-primer', label: '1. Math Primer' },
        { id: 'python-core', label: '2. Python Core' },
        { id: 'ml-libraries', label: '3. Key Libraries' },
        { id: 'learning-definition', label: '4. What is Learning?' },
        { id: 'types-of-learning', label: '5. Types of Learning' },
        { id: 'data-preprocessing', label: '6. Data & Scaling' },
        { id: 'optimization', label: '7. Optimization' },
        { id: 'bias-variance', label: '8. Bias-Variance' }
    ]
  },
  { 
    id: ViewSection.REGRESSION, 
    label: 'Supervised (Reg)', 
    icon: <TrendingUp size={20} />,
    subItems: [
      { id: 'linear-regression', label: 'Linear Regression' },
      { id: 'ridge-lasso', label: 'Ridge / Lasso' },
      { id: 'polynomial-regression', label: 'Polynomial' }
    ]
  },
  { 
    id: ViewSection.CLASSIFICATION, 
    label: 'Supervised (Class)', 
    icon: <GitBranch size={20} />,
    subItems: [
      { id: 'logistic-regression', label: 'Logistic Regression' },
      { id: 'knn', label: 'K-Nearest Neighbors' },
      { id: 'svm', label: 'SVM' },
      { id: 'naive-bayes', label: 'Naive Bayes' },
      { id: 'decision-trees', label: 'Decision Trees' }
    ]
  },
  {
    id: ViewSection.ENSEMBLE,
    label: 'Ensemble Methods',
    icon: <Layers size={20} />,
    subItems: [
        { id: 'random-forest', label: 'Random Forest' },
        { id: 'adaboost', label: 'AdaBoost' },
        { id: 'gradient-boosting', label: 'Gradient Boosting' }
    ]
  },
  { 
    id: ViewSection.UNSUPERVISED, 
    label: 'Unsupervised', 
    icon: <Network size={20} />,
    subItems: [
        { id: 'k-means', label: 'K-Means' },
        { id: 'hierarchical', label: 'Hierarchical' },
        { id: 'dbscan', label: 'DBSCAN' },
        { id: 'pca', label: 'PCA' },
        { id: 'tsne', label: 't-SNE' }
    ]
  },
  { 
    id: ViewSection.DEEP_LEARNING, 
    label: 'Deep Learning', 
    icon: <BrainCircuit size={20} />,
    subItems: [
        { id: 'mlp', label: 'MLP' },
        { id: 'cnn', label: 'CNN' },
        { id: 'rnn', label: 'RNN / LSTM' },
        { id: 'embeddings', label: 'Embeddings' },
        { id: 'transformers', label: 'Transformers' }
    ]
  },
  {
      id: ViewSection.REINFORCEMENT,
      label: 'Reinforcement',
      icon: <Gamepad2 size={20} />,
      subItems: [
          { id: 'rl-foundations', label: 'Foundations (MDP)' },
          { id: 'exploration', label: 'Exploration (Bandits)' },
          { id: 'value-based', label: 'Value-Based (DQN)' },
          { id: 'policy-based', label: 'Policy-Based' },
          { id: 'actor-critic', label: 'Actor-Critic' }
      ]
  },
  {
      id: ViewSection.MODEL_COMPARISON,
      label: 'Model Battleground',
      icon: <Swords size={20} />
  },
  { 
    id: ViewSection.PROJECT_LAB, 
    label: 'Project Lab', 
    icon: <FlaskConical size={20} /> 
  },
];

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewSection>(ViewSection.FOUNDATIONS);
  const [expandedSection, setExpandedSection] = useState<ViewSection | null>(ViewSection.FOUNDATIONS);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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
     if (section === ViewSection.PROJECT_LAB || section === ViewSection.MODEL_COMPARISON) {
         setCurrentView(section);
         setExpandedSection(null);
     } else {
         // Toggle accordion
         setExpandedSection(expandedSection === section ? null : section);
         // Also navigate to the main category page
         setCurrentView(section);
     }
  };

  const handleSubItemClick = (section: ViewSection, id: string) => {
      setCurrentView(section);
      setIsMobileMenuOpen(false);
      
      // Small delay to allow view to render before scrolling
      setTimeout(() => {
          const element = document.getElementById(id);
          if (element) {
              element.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
      }, 100);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden">
      
      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 w-full bg-slate-900 border-b border-slate-800 z-50 flex items-center justify-between p-4">
        <span className="font-serif font-bold text-indigo-400">Encyclopedia of Algorithms</span>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
          {isMobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Sidebar Navigation */}
      <aside className={`
        fixed md:relative z-40 h-full w-72 bg-slate-900 border-r border-slate-800 flex-shrink-0 transition-transform duration-300 overflow-y-auto
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <div className="p-6 border-b border-slate-800 hidden md:block sticky top-0 bg-slate-900 z-10">
          <h1 className="font-serif font-bold text-xl text-white tracking-wide">
            Encyclopedia<span className="text-indigo-500">.</span>Algo
          </h1>
          <p className="text-xs text-slate-500 mt-1">Machine Learning Mastery Hub</p>
        </div>
        
        <nav className="p-4 space-y-2 mt-16 md:mt-0">
          {NAV_ITEMS.map((item) => (
            <div key={item.id} className="space-y-1">
                <button
                onClick={() => handleNavClick(item.id)}
                className={`
                    w-full flex items-center justify-between px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200
                    ${currentView === item.id 
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                    }
                `}
                >
                <div className="flex items-center gap-3">
                    {item.icon}
                    {item.label}
                </div>
                {item.subItems && (
                    <span>{expandedSection === item.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</span>
                )}
                </button>
                
                {/* Submenu Accordion */}
                {item.subItems && expandedSection === item.id && (
                    <div className="pl-11 space-y-1 animate-fade-in-down">
                        {item.subItems.map((sub) => (
                            <button
                                key={sub.id}
                                onClick={() => handleSubItemClick(item.id, sub.id)}
                                className="block w-full text-left py-2 px-2 text-xs text-slate-400 hover:text-indigo-400 border-l border-slate-700 hover:border-indigo-400 transition-colors"
                            >
                                {sub.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800 mt-auto">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500"></div>
            <div>
              <div className="text-[10px] text-slate-500">Encyclopedia Mode</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 h-full overflow-y-auto pt-20 md:pt-0 scroll-smooth relative">
        <div className="max-w-6xl mx-auto p-6 md:p-12">
          {renderContent()}
        </div>
      </main>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
    </div>
  );
};

export default App;