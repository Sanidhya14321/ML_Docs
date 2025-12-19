
import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Calendar, Clock, BarChart, Tag, Share2, ArrowRight } from 'lucide-react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Callout } from './Callout';
import { LatexRenderer } from './LatexRenderer';

interface DocViewerProps {
  topicId: string;
  title: string;
}

// Helper to generate context-aware mock content
const getMockData = (id: string, title: string) => {
  const isMath = id.startsWith('math');
  const isCode = id.startsWith('de') || id.startsWith('mlops');
  
  return {
    description: `A comprehensive deep dive into ${title}, exploring its theoretical underpinnings, algorithmic implementation, and real-world application in modern ${isCode ? 'data infrastructure' : 'machine learning systems'}.`,
    math: isMath 
      ? "\\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}"
      : "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}",
    code: isCode 
      ? `import apache_beam as beam\n\nwith beam.Pipeline() as p:\n    (p | 'Read' >> beam.ReadFromText('gs://data/input.txt')\n       | 'Transform' >> beam.Map(process_record)\n       | 'Write' >> beam.WriteToText('output.txt'))`
      : `import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer = nn.Linear(768, 10)\n\n    def forward(self, x):\n        return self.layer(x)`
  };
};

export const DocViewer: React.FC<DocViewerProps> = ({ topicId, title }) => {
  const content = useMemo(() => getMockData(topicId, title), [topicId, title]);
  const tags = topicId.split('/').filter(t => t !== 'cat-math' && t !== 'cat-ml');

  return (
    <div className="pb-24 max-w-4xl mx-auto">
      {/* 1. Header Section */}
      <header className="mb-12 border-b border-slate-800 pb-8">
        <div className="flex justify-between items-start mb-6">
           <div className="flex gap-2">
              {tags.map((tag, i) => (
                <span key={i} className="text-[10px] font-mono font-bold uppercase px-2 py-1 rounded bg-slate-900 border border-slate-800 text-indigo-400">
                  #{tag}
                </span>
              ))}
           </div>
           <button className="p-2 rounded-lg bg-slate-900 text-slate-400 hover:text-white transition-colors">
              <Share2 size={16} />
           </button>
        </div>

        <h1 className="text-4xl md:text-5xl font-serif font-bold text-white mb-6 leading-tight">
          {title}
        </h1>
        <p className="text-xl text-slate-400 font-light leading-relaxed mb-8">
          {content.description}
        </p>
        
        <div className="flex flex-wrap items-center gap-6 text-xs font-mono text-slate-500 uppercase tracking-widest">
           <div className="flex items-center gap-2">
              <Calendar size={14} /> Updated Today
           </div>
           <div className="flex items-center gap-2">
              <Clock size={14} /> 12 min read
           </div>
           <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-indigo-500/20 text-indigo-400 bg-indigo-500/10">
              <BarChart size={14} /> Intermediate
           </div>
        </div>
      </header>
      
      {/* 2. Main Content Area */}
      <div className="prose prose-invert prose-lg max-w-none">
         <Callout type="note" title="Learning Objectives">
            In this module, we will deconstruct the core mechanics of <strong>{title}</strong>. 
            By the end, you will understand the mathematical derivation and how to implement a production-ready version from scratch.
         </Callout>

         <h2 className="flex items-center gap-3 text-2xl font-bold text-slate-200 mt-12 mb-6">
            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 text-sm">01</span>
            Theoretical Foundation
         </h2>
         <p className="text-slate-400 leading-relaxed">
            At its core, {title} relies on minimizing a specific objective function. 
            Understanding the gradients involved is crucial for debugging convergence issues in practice.
         </p>

         <MathBlock label="Core Equation">
            <LatexRenderer formula={content.math} displayMode={true} />
         </MathBlock>

         <p className="text-slate-400 leading-relaxed">
            Notice how the interaction between the parameters allows the system to generalize beyond the training set.
            We can verify this property by examining the asymptotic behavior as <LatexRenderer formula="n \to \infty" />.
         </p>

         <h2 className="flex items-center gap-3 text-2xl font-bold text-slate-200 mt-12 mb-6">
            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 text-sm">02</span>
            Implementation Strategy
         </h2>
         <p className="text-slate-400 leading-relaxed">
            While libraries like Scikit-Learn or PyTorch abstract away the complexity, implementing {title} from scratch is the best way to intuit the hyperparameters.
         </p>

         <CodeBlock 
            language="python" 
            filename="implementation.py" 
            code={content.code} 
         />

         <Callout type="tip" title="Production Note">
            When deploying this to production, ensure you handle edge cases where input vectors might be sparse or non-normalized, as this can lead to numerical instability.
         </Callout>

         <h2 className="flex items-center gap-3 text-2xl font-bold text-slate-200 mt-12 mb-6">
            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-rose-500/10 text-rose-400 text-sm">03</span>
            Real-World Application
         </h2>
         <p className="text-slate-400 leading-relaxed">
            In industry, {title} is frequently used in high-throughput environments. 
            The trade-off between latency and accuracy usually favors this approach when resources are constrained.
         </p>
         
         <div className="mt-12 p-1 rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
            <div className="bg-slate-950 rounded-xl p-8 text-center">
                <h3 className="text-xl font-bold text-white mb-4">Ready to test your knowledge?</h3>
                <p className="text-slate-400 mb-6">Explore the interactive lab to see {title} in action against a real dataset.</p>
                <button className="px-6 py-3 bg-white text-slate-950 font-bold rounded-lg hover:bg-slate-200 transition-colors flex items-center gap-2 mx-auto">
                   Go to Lab <ArrowRight size={16} />
                </button>
            </div>
         </div>
      </div>
    </div>
  );
};
