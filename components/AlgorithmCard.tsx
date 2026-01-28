
import React, { useState } from 'react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Check, X, Info, Sparkles, ListChecks } from 'lucide-react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import { AlgorithmSkeleton } from './Skeletons';

export interface Hyperparameter {
  name: string;
  description: string;
  default?: string;
  range?: string;
}

interface AlgorithmCardProps {
  id: string;
  title: string;
  theory: string;
  math: React.ReactNode;
  mathLabel?: string;
  code: string;
  pros: string[];
  cons: string[];
  complexity?: 'Fundamental' | 'Intermediate' | 'Advanced';
  hyperparameters?: Hyperparameter[];
  steps?: string[];
  children?: React.ReactNode;
  isLoading?: boolean;
}

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  show: { 
    opacity: 1, 
    y: 0, 
    transition: { type: "spring", stiffness: 200, damping: 20 } 
  }
};

export const AlgorithmCard: React.FC<AlgorithmCardProps> = ({
  id,
  title,
  theory,
  math,
  mathLabel,
  code,
  pros,
  cons,
  complexity = 'Intermediate',
  hyperparameters,
  steps,
  children,
  isLoading = false
}) => {
  const [showAI, setShowAI] = useState(false);

  if (isLoading) {
    return <AlgorithmSkeleton />;
  }

  const complexityColors = {
    'Fundamental': 'bg-emerald-100 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20',
    'Intermediate': 'bg-indigo-100 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border-indigo-200 dark:border-indigo-500/20',
    'Advanced': 'bg-rose-100 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/20'
  };

  return (
    <motion.div 
      id={id} 
      variants={itemVariants}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className="bg-white dark:bg-slate-900/50 backdrop-blur-xl border border-slate-200 dark:border-slate-800 rounded-3xl overflow-hidden mb-16 shadow-xl dark:shadow-2xl scroll-mt-24 transition-all hover:border-indigo-500/30 group/card"
    >
      <div className="p-8 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-indigo-100 dark:bg-indigo-500/10 flex items-center justify-center text-indigo-600 dark:text-indigo-400 group-hover/card:scale-110 transition-transform duration-500">
             <Info size={24} />
          </div>
          <div>
            <h2 className="text-3xl font-serif font-bold text-slate-900 dark:text-white tracking-wide group-hover/card:text-indigo-600 dark:group-hover/card:text-indigo-300 transition-colors">{title}</h2>
            <div className="flex items-center gap-2 mt-1">
               <div className={`px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase tracking-widest border ${complexityColors[complexity]}`}>
                {complexity}
              </div>
            </div>
          </div>
        </div>
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowAI(!showAI)}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${showAI ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-700'}`}
        >
          <Sparkles size={14} className={showAI ? "animate-pulse" : ""} />
          AI Explanation
        </motion.button>
      </div>
      
      <div className="p-8 md:p-10 space-y-10">
        <div className="relative">
          <h3 className="text-[10px] font-black text-indigo-600 dark:text-indigo-400 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
            <Info size={14} /> Theoretical Foundations
          </h3>
          <p className="text-slate-600 dark:text-slate-400 leading-relaxed text-lg font-light">{theory}</p>
        </div>

        <AnimatePresence>
          {showAI && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="p-6 rounded-2xl bg-indigo-50 dark:bg-indigo-500/5 border border-indigo-100 dark:border-indigo-500/20 mb-8 relative">
                 <div className="absolute -top-3 left-6 px-3 py-1 bg-indigo-600 text-[10px] font-black rounded-lg uppercase tracking-widest text-white">Intuitive Insight</div>
                 <p className="text-indigo-800 dark:text-indigo-300 text-sm italic leading-relaxed">
                   Think of {title} not just as a mathematical operation, but as a way to find patterns in chaos. 
                   {complexity === 'Fundamental' ? ' It is like looking for the most basic building blocks of a puzzle.' : ' It is like building a complex network of logic where each step refines the overall understanding.'}
                 </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {children && (
          <div className="my-10 bg-slate-50 dark:bg-slate-950/80 rounded-3xl border border-slate-200 dark:border-slate-800 p-8 shadow-inner relative group/viz overflow-hidden">
             <div className="absolute top-4 right-6 text-[9px] font-mono text-slate-400 dark:text-slate-600 uppercase tracking-[0.3em]">Live Interaction Layer</div>
             <div className="relative z-10">
              {children}
             </div>
             <div className="absolute -bottom-24 -right-24 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none"></div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          <div className="lg:col-span-5 space-y-8">
            <MathBlock label={mathLabel || "Core Equation"}>
              {math}
            </MathBlock>
            
            <div className="space-y-6">
              <h3 className="text-[10px] font-black text-indigo-600 dark:text-indigo-400 uppercase tracking-[0.2em]">The Scorecard</h3>
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-emerald-50 dark:bg-emerald-500/5 p-5 rounded-2xl border border-emerald-100 dark:border-emerald-500/10">
                  <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3">
                    <Check size={14} /> Strengths
                  </h4>
                  <ul className="space-y-2">
                    {pros.map((pro, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400">
                        <span className="w-1 h-1 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="bg-rose-50 dark:bg-rose-500/5 p-5 rounded-2xl border border-rose-100 dark:border-rose-500/10">
                  <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-rose-600 dark:text-rose-400 mb-3">
                    <X size={14} /> Weaknesses
                  </h4>
                  <ul className="space-y-2">
                    {cons.map((con, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400">
                        <span className="w-1 h-1 rounded-full bg-rose-500 mt-1.5 flex-shrink-0" />
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-7">
            <h3 className="text-[10px] font-black text-indigo-600 dark:text-indigo-400 uppercase tracking-[0.2em] mb-4">Implementation Reference</h3>
            <CodeBlock code={code} />
            
            {hyperparameters && hyperparameters.length > 0 && (
              <div className="mt-8">
                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4">Key Parameters</h4>
                <div className="space-y-3">
                  {hyperparameters.map((param, idx) => (
                    <div key={idx} className="bg-white dark:bg-slate-800/30 p-4 rounded-xl border border-slate-200 dark:border-slate-800 flex justify-between items-center group/param hover:border-slate-300 dark:hover:border-slate-700 transition-colors">
                      <div className="flex-1">
                        <code className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">{param.name}</code>
                        <p className="text-[10px] text-slate-500 mt-1">{param.description}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-slate-400 dark:text-slate-600 uppercase font-mono tracking-tighter">Default</div>
                        <div className="text-xs text-slate-600 dark:text-slate-300 font-mono">{param.default || 'N/A'}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {steps && steps.length > 0 && (
              <div className="mt-8 pt-8 border-t border-slate-200 dark:border-slate-800/50">
                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                  <ListChecks size={14} className="text-indigo-500" /> Implementation Protocol
                </h4>
                <div className="space-y-3">
                  {steps.map((step, idx) => (
                    <div key={idx} className="flex gap-4">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-[10px] font-bold text-slate-500 flex items-center justify-center">
                        {idx + 1}
                      </span>
                      <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed pt-0.5">
                        {step}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};
