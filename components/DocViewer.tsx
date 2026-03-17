
import React, { useMemo, useState, useEffect } from 'react';
import { Calendar, Clock, BarChart, Share2, ArrowRight, FlaskConical } from 'lucide-react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Callout } from './Callout';
import { LatexRenderer } from './LatexRenderer';
import { DocPagination } from './DocPagination';
import { getTopicById } from '../lib/contentHelpers';
import { DocSkeleton } from './Skeletons';
import ReactMarkdown from 'react-markdown';

interface DocViewerProps {
  topicId: string;
  title: string;
  isCompact?: boolean;
}

// Fallback mock content generator for topics that don't have custom content yet
const getMockData = (id: string, title: string) => {
  return {
    theory: `A comprehensive deep dive into ${title}. This section explores the theoretical underpinnings, explaining why this concept is critical in the broader context of Artificial Intelligence engineering.`,
    math: "\\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}",
    mathLabel: "Core Equation",
    code: `import torch\nimport torch.nn as nn\n\n# Example implementation\nmodel = nn.Linear(10, 1)`,
    codeLanguage: 'python'
  };
};

import { motion } from 'framer-motion';

export const DocViewer: React.FC<DocViewerProps> = ({ topicId, title, isCompact = false }) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
    const timer = setTimeout(() => setIsLoading(false), 600);
    return () => clearTimeout(timer);
  }, [topicId]);

  const topic = getTopicById(topicId);
  const displayTitle = topic?.title || title;
  const displayDesc = topic?.description || "Technical documentation and analysis.";
  
  const content = topic?.details || getMockData(topicId, displayTitle);
  
  const tags = topicId.split('/');
  const showStartLab = topic?.type === 'lab' || !!topic?.labConfig;

  if (isLoading && !isCompact) {
      return <DocSkeleton />;
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={`mx-auto ${isCompact ? 'max-w-none px-2' : 'max-w-4xl pb-24'}`}
    >
      {/* 1. Header Section - Hidden in Compact Mode */}
      {!isCompact && (
        <header className="mb-16 border-b border-border-strong pb-12 transition-all duration-300">
          <div className="flex justify-between items-start mb-8 gap-4">
            <div className="flex flex-wrap gap-2">
                {tags.map((tag, i) => (
                  <span key={i} className="text-[9px] font-mono font-black uppercase px-3 py-1 bg-surface border border-border-strong text-brand transition-colors">
                    #{tag}
                  </span>
                ))}
            </div>
            <button className="p-2 bg-surface border border-border-strong text-text-muted hover:text-text-primary hover:border-text-primary transition-all duration-300 shrink-0">
                <Share2 size={16} />
            </button>
          </div>

          <h1 className="text-4xl md:text-6xl font-heading font-black text-text-primary mb-8 leading-none uppercase tracking-tight transition-all duration-300">
            {displayTitle}
          </h1>
          
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
            <div className="flex flex-wrap items-center gap-6 text-[10px] font-mono font-black text-text-muted uppercase tracking-widest transition-colors duration-300">
              <div className="flex items-center gap-2">
                  <Calendar size={14} className="text-brand" /> UPDATED_TODAY
              </div>
              <div className="flex items-center gap-2">
                  <Clock size={14} className="text-brand" /> 15_MIN_READ
              </div>
              <div className="flex items-center gap-2 px-3 py-1 border border-border-strong bg-surface text-text-primary">
                  <BarChart size={14} className="text-brand" /> INTERMEDIATE
              </div>
            </div>

            {showStartLab && (
              <a 
                href={`#lab/${topicId}`}
                className="inline-flex items-center gap-3 px-6 py-3 bg-text-primary text-app hover:bg-brand text-[11px] font-mono font-black uppercase tracking-[0.2em] transition-all group"
              >
                <FlaskConical size={16} className="group-hover:rotate-12 transition-transform" />
                INITIALIZE_LAB
              </a>
            )}
          </div>
          
          <p className="text-xl text-text-secondary font-light leading-relaxed mt-10 max-w-3xl transition-colors duration-300">
            {displayDesc}
          </p>
        </header>
      )}

      {/* Compact Header for Lab Mode */}
      {isCompact && (
        <div className="mb-8 pb-6 border-b border-border-strong">
           <h2 className="text-2xl font-heading font-black text-text-primary mb-3 uppercase tracking-tight">{displayTitle}</h2>
           <p className="text-sm text-text-secondary leading-relaxed font-light">{displayDesc}</p>
        </div>
      )}
      
      {/* 2. Main Content Area */}
      <div className={`prose dark:prose-invert ${isCompact ? 'prose-sm max-w-none' : 'prose-lg max-w-none'}`}>
         {topic?.type === 'lab' && (
            <div className="mb-12">
              <Callout type="tip" title="INTERACTIVE_ENVIRONMENT_ACTIVE">
                 This is a hands-on lab. Read the theory below, then use the Code Editor on the right to implement the solution.
              </Callout>
            </div>
         )}

         {/* Theory Section */}
         {content.theory && (
           <div className="mb-16">
             <h2 className="flex items-center gap-4 text-[11px] font-mono font-black text-text-muted uppercase tracking-[0.4em] mt-12 mb-8">
                <div className="w-8 h-px bg-brand" />
                01 // THEORETICAL_FOUNDATION
             </h2>
             <div className="text-text-secondary leading-relaxed font-light">
                <ReactMarkdown>{content.theory}</ReactMarkdown>
             </div>
           </div>
         )}

         {/* Math Section */}
         {content.math && (
           <div className="mb-16">
             <MathBlock label={content.mathLabel || "FORMAL_DEFINITION"} type={isCompact ? "inline" : "block"}>
                <LatexRenderer formula={content.math} displayMode={true} />
             </MathBlock>
           </div>
         )}

         {/* Code Section */}
         {content.code && (
           <div className="mb-16">
             <h2 className="flex items-center gap-4 text-[11px] font-mono font-black text-text-muted uppercase tracking-[0.4em] mt-12 mb-8">
                <div className="w-8 h-px bg-brand" />
                02 // IMPLEMENTATION_STRATEGY
             </h2>
             <p className="text-text-secondary leading-relaxed font-light mb-6">
                The following implementation demonstrates the core logic in <code className="text-brand font-bold">{content.codeLanguage || 'python'}</code>:
             </p>
             <div className="border border-border-strong">
               <CodeBlock 
                  language={content.codeLanguage || 'python'}
                  code={content.code} 
               />
             </div>
           </div>
         )}

         {/* Extra content only shown in full view */}
         {!isCompact && showStartLab && (
             <div className="mt-24 p-12 border border-border-strong bg-surface relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-1 h-full bg-brand" />
                <div className="relative z-10 text-center max-w-xl mx-auto">
                    <h3 className="text-2xl font-heading font-black text-text-primary mb-4 uppercase tracking-tight">READY_FOR_DEPLOYMENT?</h3>
                    <p className="text-text-secondary font-light mb-8">Launch the interactive workspace to apply these concepts against real-world data structures.</p>
                    <a href={`#lab/${topicId}`} className="inline-flex items-center gap-3 px-8 py-4 bg-text-primary text-app hover:bg-brand font-mono font-black text-[12px] uppercase tracking-[0.2em] transition-all group/btn shadow-xl">
                       LAUNCH_LAB_ENVIRONMENT <ArrowRight size={18} className="group-hover/btn:translate-x-2 transition-transform" />
                    </a>
                </div>
                <div className="absolute inset-0 opacity-[0.02] pointer-events-none" 
                     style={{ backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)', backgroundSize: '16px 16px' }} />
             </div>
         )}
      </div>

      {/* 3. Footer Navigation - Hidden in Compact Mode */}
      {!isCompact && (
        <div className="mt-24 pt-12 border-t border-border-strong">
          <DocPagination currentPath={topicId} />
        </div>
      )}
    </motion.div>
  );
};
