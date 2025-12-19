import React from 'react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Check, X, Sliders, Info } from 'lucide-react';

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
  children?: React.ReactNode;
}

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
  children
}) => {
  const complexityColors = {
    'Fundamental': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    'Intermediate': 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    'Advanced': 'bg-rose-500/10 text-rose-400 border-rose-500/20'
  };

  return (
    <div id={id} className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden mb-12 shadow-2xl scroll-mt-24 transition-all hover:border-slate-700/50 group">
      <div className="p-6 border-b border-slate-800 bg-slate-850 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-serif font-bold text-white tracking-wide">{title}</h2>
        </div>
        <div className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest border ${complexityColors[complexity]}`}>
          {complexity}
        </div>
      </div>
      
      <div className="p-6 md:p-8 space-y-8">
        {/* Theory Section */}
        <div className="relative">
          <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
            <Info size={14} /> Theoretical Foundations
          </h3>
          <p className="text-slate-400 leading-relaxed text-lg font-light">{theory}</p>
        </div>

        {/* Visualization Area */}
        {children && (
          <div className="my-6 bg-slate-950 rounded-2xl border border-slate-800/50 p-6 shadow-inner relative group/viz">
             <div className="absolute top-3 right-4 text-[8px] font-mono text-slate-700 uppercase tracking-widest">Interactive Component</div>
             {children}
          </div>
        )}

        {/* Math Section */}
        <div className="transform transition-transform hover:scale-[1.01]">
           <MathBlock label={mathLabel || "Core Formula"}>
             {math}
           </MathBlock>
        </div>

        {/* Code Section */}
        <div className="group/code">
          <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-3">Implementation</h3>
          <CodeBlock code={code} />
        </div>

        {/* Hyperparameters Section */}
        {hyperparameters && hyperparameters.length > 0 && (
          <div>
            <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
              <Sliders size={14} /> Control Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {hyperparameters.map((param, idx) => (
                <div key={idx} className="bg-slate-950/40 p-4 rounded-xl border border-slate-800/50 hover:border-indigo-500/30 transition-all hover:bg-slate-950/60">
                   <div className="flex justify-between items-center mb-2">
                     <code className="text-xs font-mono font-bold text-indigo-300 bg-indigo-900/20 px-2 py-1 rounded">{param.name}</code>
                     {param.default && <span className="text-[9px] text-slate-500 font-mono bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800/50 uppercase">Default: {param.default}</span>}
                   </div>
                   <p className="text-xs text-slate-500 leading-relaxed mb-2">{param.description}</p>
                   {param.range && (
                     <div className="text-[9px] text-slate-600 font-mono flex items-center gap-1">
                       <span className="w-1 h-1 rounded-full bg-indigo-500/40"></span>
                       Domain: <span className="text-slate-500 italic">{param.range}</span>
                     </div>
                   )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pros & Cons */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-emerald-500/5 p-6 rounded-2xl border border-emerald-500/10 transition-colors hover:border-emerald-500/20">
            <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-emerald-400 mb-4">
              <Check size={14} /> Advantages
            </h4>
            <ul className="space-y-3">
              {pros.map((pro, idx) => (
                <li key={idx} className="flex items-start gap-3 text-sm text-slate-400 font-light">
                  <span className="w-1 h-1 rounded-full bg-emerald-500 mt-2 flex-shrink-0" />
                  {pro}
                </li>
              ))}
            </ul>
          </div>
          
          <div className="bg-rose-500/5 p-6 rounded-2xl border border-rose-500/10 transition-colors hover:border-rose-500/20">
            <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-rose-400 mb-4">
              <X size={14} /> Limitations
            </h4>
            <ul className="space-y-3">
              {cons.map((con, idx) => (
                <li key={idx} className="flex items-start gap-3 text-sm text-slate-400 font-light">
                  <span className="w-1 h-1 rounded-full bg-rose-500 mt-2 flex-shrink-0" />
                  {con}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};