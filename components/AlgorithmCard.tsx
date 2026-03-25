
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
    'Fundamental': 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
    'Intermediate': 'bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20',
    'Advanced': 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20'
  };

  return (
    <motion.div 
      id={id} 
      variants={itemVariants}
      whileHover={{ y: -5, boxShadow: "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)" }}
      className="bg-surface border border-border-strong rounded-none overflow-hidden mb-16 shadow-sm scroll-mt-24 transition-all duration-300 group/card"
    >
      <div className="p-6 md:p-8 border-b border-border-strong bg-app/30 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded bg-text-primary flex items-center justify-center text-app shrink-0">
             <Info size={20} />
          </div>
          <div>
            <h2 className="text-xl md:text-2xl font-heading font-black text-text-primary uppercase tracking-tight">{title}</h2>
            <div className="flex items-center gap-2 mt-1">
               <div className={`px-2 py-0.5 rounded-none text-[8px] font-mono uppercase tracking-widest border ${complexityColors[complexity]}`}>
                {complexity}
              </div>
            </div>
          </div>
        </div>
        <motion.button 
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setShowAI(!showAI)}
          className={`self-start md:self-auto flex items-center gap-2 px-4 py-2 rounded-none text-[10px] font-mono uppercase tracking-widest transition-all duration-300 ${showAI ? 'bg-text-primary text-app' : 'bg-surface border border-border-strong text-text-secondary hover:text-text-primary hover:border-text-primary'}`}
        >
          <Sparkles size={12} className={showAI ? "animate-pulse" : ""} />
          AI_ANALYSIS
        </motion.button>
      </div>
      
      <div className="p-8 md:p-10 space-y-12">
        <div className="relative">
          <h3 className="text-[9px] font-mono font-black text-brand uppercase tracking-[0.3em] mb-6 flex items-center gap-2">
            <div className="w-1 h-1 bg-brand rounded-full" /> THEORETICAL_MODEL
          </h3>
          <p className="text-text-secondary leading-relaxed text-lg font-light max-w-3xl">{theory}</p>
        </div>

        <AnimatePresence>
          {showAI && (
            <motion.div 
              id={`ai-explanation-${id}`}
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="p-6 bg-brand/5 border-l-4 border-brand mb-8 relative">
                 <div className="text-[9px] font-mono font-black text-brand uppercase tracking-widest mb-2">INTUITIVE_INSIGHT</div>
                 <p className="text-text-primary text-sm italic leading-relaxed font-serif">
                   "{title} is effectively a mapping function that minimizes entropy within the target domain. 
                   {complexity === 'Fundamental' ? ' It serves as a foundational primitive for more complex architectures.' : ' It introduces non-linear decision boundaries that are essential for high-dimensional feature spaces.'}"
                 </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {children && (
          <div className="my-10 bg-app border border-border-strong p-8 relative group/viz overflow-hidden">
             <div className="absolute top-4 right-6 text-[8px] font-mono text-text-muted uppercase tracking-[0.4em]">DYNAMIC_SIMULATION_LAYER</div>
             <div className="relative z-10">
              {children}
             </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 border-t border-border-subtle pt-12">
          <div className="lg:col-span-5 space-y-10">
            <MathBlock label={mathLabel || "FORMAL_DEFINITION"}>
              {math}
            </MathBlock>
            
            <div className="space-y-6">
              <h3 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em]">PERFORMANCE_METRICS</h3>
              <div className="grid grid-cols-1 gap-0 border border-border-subtle">
                <div className="p-5 border-b border-border-subtle bg-emerald-500/5">
                  <h4 className="flex items-center gap-2 font-mono font-black text-[9px] uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-4">
                    <Check size={12} /> ADVANTAGES
                  </h4>
                  <ul className="space-y-3">
                    {pros.map((pro, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-[11px] text-text-secondary font-mono uppercase tracking-tight">
                        <span className="text-emerald-500">+</span>
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="p-5 bg-rose-500/5">
                  <h4 className="flex items-center gap-2 font-mono font-black text-[9px] uppercase tracking-widest text-rose-600 dark:text-rose-400 mb-4">
                    <X size={12} /> LIMITATIONS
                  </h4>
                  <ul className="space-y-3">
                    {cons.map((con, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-[11px] text-text-secondary font-mono uppercase tracking-tight">
                        <span className="text-rose-500">-</span>
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-7">
            <h3 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-6">IMPLEMENTATION_REFERENCE</h3>
            <div className="border border-border-strong">
              <CodeBlock code={code} />
            </div>
            
            {hyperparameters && hyperparameters.length > 0 && (
              <div className="mt-10">
                <h4 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-6">HYPERPARAMETER_SPACE</h4>
                <div className="grid grid-cols-1 gap-px bg-border-strong border border-border-strong">
                  {hyperparameters.map((param, idx) => (
                    <div key={idx} className="bg-surface p-4 flex justify-between items-center group/param">
                      <div className="flex-1">
                        <code className="text-[11px] font-mono font-bold text-brand">{param.name}</code>
                        <p className="text-[10px] text-text-muted mt-1 uppercase tracking-tighter">{param.description}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-[8px] text-text-muted uppercase font-mono tracking-widest mb-1">DEFAULT</div>
                        <div className="text-[10px] text-text-primary font-mono">{param.default || 'NULL'}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {steps && steps.length > 0 && (
              <div className="mt-10 pt-10 border-t border-border-subtle">
                <h4 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-6 flex items-center gap-2">
                  <ListChecks size={12} className="text-brand" /> EXECUTION_PROTOCOL
                </h4>
                <div className="space-y-4">
                  {steps.map((step, idx) => (
                    <div key={idx} className="flex gap-4 group/step">
                      <span className="flex-shrink-0 w-6 h-6 border border-border-strong text-[10px] font-mono font-bold text-text-muted flex items-center justify-center group-hover/step:border-brand group-hover/step:text-brand transition-colors">
                        {String(idx + 1).padStart(2, '0')}
                      </span>
                      <p className="text-[11px] text-text-secondary leading-relaxed pt-1 uppercase tracking-tight">
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
