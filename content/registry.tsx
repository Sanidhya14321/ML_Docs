
import React, { useState, lazy } from 'react';
import { ViewSection, ContentModule } from '../types';
import { Calendar, Clock, BarChart, Share2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { SEOHead } from '../components/SEOHead';
import { Feedback } from '../components/Feedback';
import { ShareDialog } from '../components/ShareDialog';

// Helper Component to Read Metadata + Content
const DocReader: React.FC<{ module: ContentModule; path: string }> = ({ module, path }) => {
  const { metadata, Content } = module;
  const [isShareOpen, setIsShareOpen] = useState(false);
  
  const difficultyColor = {
    'Beginner': 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    'Intermediate': 'text-indigo-400 bg-indigo-500/10 border-indigo-500/20',
    'Advanced': 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    'Expert': 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  };

  return (
    <>
      <SEOHead metadata={metadata} path={path} />
      <ShareDialog isOpen={isShareOpen} onClose={() => setIsShareOpen(false)} metadata={metadata} url={path} />

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="pb-24">
        <header className="mb-12 border-b border-slate-800 pb-8 relative">
          <div className="flex justify-between items-start">
             <div className="flex gap-2 mb-4">
                {metadata.tags.map(tag => (
                  <span key={tag} className="text-[10px] font-mono font-bold uppercase px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-500">
                    #{tag}
                  </span>
                ))}
             </div>
             
             <button 
               onClick={() => setIsShareOpen(true)}
               className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
               title="Share Article"
             >
                <Share2 size={16} />
             </button>
          </div>

          <h1 className="text-4xl md:text-5xl font-serif font-bold text-white mb-6 leading-tight">{metadata.title}</h1>
          <p className="text-lg md:text-xl text-slate-400 font-light leading-relaxed mb-8">{metadata.description}</p>
          
          <div className="flex flex-wrap items-center gap-6 text-xs font-mono text-slate-500 uppercase tracking-widest">
             <div className="flex items-center gap-2">
                <Calendar size={14} /> {metadata.date}
             </div>
             <div className="flex items-center gap-2">
                <Clock size={14} /> {metadata.readTimeMinutes} min read
             </div>
             <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${difficultyColor[metadata.difficulty]}`}>
                <BarChart size={14} /> {metadata.difficulty}
             </div>
          </div>
        </header>
        
        {/* Prose Class for Automatic Markdown Styling */}
        <div className="prose prose-invert prose-lg max-w-none">
           <Content />
        </div>

        {/* User Feedback Loop */}
        <Feedback />
      </motion.div>
    </>
  );
};

// We map generic IDs to Lazy Loaded Content Modules
export const CONTENT_REGISTRY: Record<string, any> = {
  // Legacy Views
  [ViewSection.FOUNDATIONS]: lazy(() => import('../views/FoundationsView').then(m => ({ default: m.FoundationsView }))),
  [ViewSection.OPTIMIZATION]: lazy(() => import('../views/OptimizationView').then(m => ({ default: m.OptimizationView }))),
  [ViewSection.REGRESSION]: lazy(() => import('../views/RegressionView').then(m => ({ default: m.RegressionView }))),
  [ViewSection.CLASSIFICATION]: lazy(() => import('../views/ClassificationView').then(m => ({ default: m.ClassificationView }))),
  [ViewSection.ENSEMBLE]: lazy(() => import('../views/EnsembleView').then(m => ({ default: m.EnsembleView }))),
  [ViewSection.UNSUPERVISED]: lazy(() => import('../views/UnsupervisedView').then(m => ({ default: m.UnsupervisedView }))),
  [ViewSection.DEEP_LEARNING]: lazy(() => import('../views/DeepLearningView').then(m => ({ default: m.DeepLearningView }))),
  [ViewSection.REINFORCEMENT]: lazy(() => import('../views/ReinforcementView').then(m => ({ default: m.ReinforcementView }))),
  [ViewSection.MODEL_COMPARISON]: lazy(() => import('../views/ModelComparisonView').then(m => ({ default: m.ModelComparisonView }))),
  [ViewSection.PROJECT_LAB]: lazy(() => import('../views/ProjectLabView').then(m => ({ default: m.ProjectLabView }))),
  
  // New Content-First Modules
  // Note: 'm as any' cast is required to satisfy the ContentModule interface during strict TS builds
  'deep-learning/attention-mechanism': lazy(() => import('./deep-learning/AttentionMechanism').then(m => ({ default: () => <DocReader module={m as any} path="deep-learning/attention-mechanism" /> }))),
};
