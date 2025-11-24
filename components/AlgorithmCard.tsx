import React from 'react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Check, X, Sliders } from 'lucide-react';

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
  hyperparameters,
  children
}) => {
  return (
    <div id={id} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden mb-12 shadow-lg scroll-mt-24">
      <div className="p-6 border-b border-slate-800 bg-slate-850">
        <h2 className="text-2xl font-serif font-bold text-white tracking-wide">{title}</h2>
      </div>
      
      <div className="p-6 md:p-8 space-y-8">
        {/* Theory Section */}
        <div>
          <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-3">Theory</h3>
          <p className="text-slate-300 leading-relaxed text-lg">{theory}</p>
        </div>

        {/* Visualization Area (Optional) */}
        {children && (
          <div className="my-6 bg-slate-950 rounded-lg border border-slate-800 p-4 shadow-inner">
             {children}
          </div>
        )}

        {/* Math Section */}
        <div>
           <MathBlock label={mathLabel || "Core Equation"}>
             {math}
           </MathBlock>
        </div>

        {/* Code Section */}
        <div>
          <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-3">Python Implementation</h3>
          <CodeBlock code={code} />
        </div>

        {/* Hyperparameters Section */}
        {hyperparameters && hyperparameters.length > 0 && (
          <div>
            <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-3 flex items-center gap-2">
              <Sliders size={16} /> Key Hyperparameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {hyperparameters.map((param, idx) => (
                <div key={idx} className="bg-slate-950/50 p-4 rounded border border-slate-800/50 hover:border-slate-700 transition-colors">
                   <div className="flex justify-between items-center mb-2">
                     <code className="text-xs font-mono font-bold text-indigo-300 bg-indigo-900/20 px-2 py-1 rounded">{param.name}</code>
                     {param.default && <span className="text-[10px] text-slate-500 font-mono bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">Def: {param.default}</span>}
                   </div>
                   <p className="text-sm text-slate-400 leading-snug mb-2">{param.description}</p>
                   {param.range && (
                     <div className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                       <span className="w-1 h-1 rounded-full bg-slate-600"></span>
                       Range: <span className="text-slate-400">{param.range}</span>
                     </div>
                   )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pros & Cons */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-950/50 p-5 rounded-lg border border-slate-800/50">
            <h4 className="flex items-center gap-2 font-bold text-emerald-400 mb-4">
              <Check size={18} /> Advantages
            </h4>
            <ul className="space-y-2">
              {pros.map((pro, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                  {pro}
                </li>
              ))}
            </ul>
          </div>
          
          <div className="bg-slate-950/50 p-5 rounded-lg border border-slate-800/50">
            <h4 className="flex items-center gap-2 font-bold text-rose-400 mb-4">
              <X size={18} /> Limitations
            </h4>
            <ul className="space-y-2">
              {cons.map((con, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-1.5 flex-shrink-0" />
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