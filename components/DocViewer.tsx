
import React, { useMemo, useState, useEffect } from 'react';
import { Calendar, Clock, BarChart, Share2, ArrowRight, FlaskConical, BookOpen } from 'lucide-react';
import { MathBlock } from './MathBlock';
import { CodeBlock } from './CodeBlock';
import { Callout } from './Callout';
import { LatexRenderer } from './LatexRenderer';
import { DocPagination } from './DocPagination';
import { getTopicById } from '../lib/contentHelpers';
import { DocSkeleton } from './Skeletons';

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

export const DocViewer: React.FC<DocViewerProps> = ({ topicId, title, isCompact = false }) => {
  const [isLoading, setIsLoading] = useState(true);

  // Simulate granular loading for docs (e.g., fetching MDX/Remote content)
  useEffect(() => {
    setIsLoading(true);
    const timer = setTimeout(() => setIsLoading(false), 600);
    return () => clearTimeout(timer);
  }, [topicId]);

  // Get real topic data
  const topic = getTopicById(topicId);
  const displayTitle = topic?.title || title;
  const displayDesc = topic?.description || "Technical documentation and analysis.";
  
  // Use topic.details if available, otherwise fallback to mock data
  const content = topic?.details || getMockData(topicId, displayTitle);
  
  const tags = topicId.split('/');
  const showStartLab = topic?.type === 'lab' || !!topic?.labConfig;

  if (isLoading && !isCompact) {
      return <DocSkeleton />;
  }

  return (
    <div className={`mx-auto ${isCompact ? 'max-w-none px-2' : 'max-w-4xl pb-24'}`}>
      {/* 1. Header Section - Hidden in Compact Mode */}
      {!isCompact && (
        <header className="mb-12 border-b border-slate-200 dark:border-slate-800 pb-8">
          <div className="flex justify-between items-start mb-6">
            <div className="flex gap-2">
                {tags.map((tag, i) => (
                  <span key={i} className="text-[10px] font-mono font-bold uppercase px-2 py-1 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-indigo-600 dark:text-indigo-400">
                    #{tag}
                  </span>
                ))}
            </div>
            <button className="p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors">
                <Share2 size={16} />
            </button>
          </div>

          <h1 className="text-4xl md:text-5xl font-serif font-bold text-slate-900 dark:text-white mb-6 leading-tight">
            {displayTitle}
          </h1>
          
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="flex flex-wrap items-center gap-6 text-xs font-mono text-slate-500 uppercase tracking-widest">
              <div className="flex items-center gap-2">
                  <Calendar size={14} /> Updated Today
              </div>
              <div className="flex items-center gap-2">
                  <Clock size={14} /> 15 min read
              </div>
              <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-indigo-200 dark:border-indigo-500/20 text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/10">
                  <BarChart size={14} /> Intermediate
              </div>
            </div>

            {/* Start Lab Button - Only if lab config exists */}
            {showStartLab && (
              <a 
                href={`#lab/${topicId}`}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg shadow-indigo-600/20 dark:shadow-indigo-900/40 transition-all hover:scale-105"
              >
                <FlaskConical size={16} /> Start Lab
              </a>
            )}
          </div>
          
          <p className="text-xl text-slate-600 dark:text-slate-400 font-light leading-relaxed mt-8">
            {displayDesc}
          </p>
        </header>
      )}

      {/* Compact Header for Lab Mode */}
      {isCompact && (
        <div className="mb-8 pb-4 border-b border-slate-800/50">
           <h2 className="text-2xl font-serif font-bold text-white mb-2">{displayTitle}</h2>
           <p className="text-sm text-slate-400 leading-relaxed">{displayDesc}</p>
        </div>
      )}
      
      {/* 2. Main Content Area */}
      <div className={`prose dark:prose-invert ${isCompact ? 'prose-sm max-w-none' : 'prose-lg max-w-none'}`}>
         {topic?.type === 'lab' && (
            <Callout type="tip" title="Interactive Environment">
               This is a hands-on lab. Read the theory below, then use the Code Editor on the right to implement the solution.
            </Callout>
         )}

         {/* Theory Section */}
         {content.theory && (
           <>
             <h2 className="flex items-center gap-3 text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-200 mt-12 mb-6">
                <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 text-sm">01</span>
                Theoretical Foundation
             </h2>
             <p className="text-slate-600 dark:text-slate-400 leading-relaxed whitespace-pre-line">
                {content.theory}
             </p>
           </>
         )}

         {/* Math Section */}
         {content.math && (
           <MathBlock label={content.mathLabel || "Key Formula"} type={isCompact ? "inline" : "block"}>
              <LatexRenderer formula={content.math} displayMode={true} />
           </MathBlock>
         )}

         {/* Code Section */}
         {content.code && (
           <>
             <h2 className="flex items-center gap-3 text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-200 mt-12 mb-6">
                <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-sm">02</span>
                Implementation Strategy
             </h2>
             <p className="text-slate-600 dark:text-slate-400 leading-relaxed mb-4">
                The following implementation demonstrates the core logic in {content.codeLanguage || 'python'}:
             </p>
             <CodeBlock 
                language={content.codeLanguage || 'python'}
                code={content.code} 
             />
           </>
         )}

         {/* Extra content only shown in full view */}
         {!isCompact && showStartLab && (
             <div className="mt-16 p-1 rounded-2xl bg-gradient-to-r from-teal-500 via-emerald-500 to-indigo-500">
                <div className="bg-white dark:bg-slate-950 rounded-xl p-8 text-center">
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">Ready to test your knowledge?</h3>
                    <p className="text-slate-600 dark:text-slate-400 mb-6">Launch the interactive workspace to apply these concepts against real data.</p>
                    <a href={`#lab/${topicId}`} className="px-6 py-3 bg-slate-900 dark:bg-white text-white dark:text-slate-950 font-bold rounded-lg hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors flex items-center gap-2 mx-auto w-fit shadow-xl">
                       Launch Lab Environment <ArrowRight size={16} />
                    </a>
                </div>
             </div>
         )}
      </div>

      {/* 3. Footer Navigation - Hidden in Compact Mode */}
      {!isCompact && <DocPagination currentPath={topicId} />}
    </div>
  );
};
