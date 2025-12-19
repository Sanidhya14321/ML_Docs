import React from 'react';
import { motion } from 'framer-motion';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { MOTION_VARIANTS } from '../constants';

const GeometricDotProduct = () => {
    return (
        <div className="flex flex-col md:flex-row items-center justify-center gap-12 py-8 bg-slate-950 rounded-2xl border border-slate-900 shadow-inner">
            <div className="relative w-64 h-64 border-l-2 border-b-2 border-slate-800">
                <svg width="100%" height="100%" viewBox="0 0 200 200">
                    <defs>
                        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                            <path d="M0,0 L0,6 L9,3 z" fill="#6366f1" />
                        </marker>
                    </defs>
                    <line x1="0" y1="200" x2="200" y2="200" stroke="#1e293b" />
                    <line x1="0" y1="0" x2="0" y2="200" stroke="#1e293b" />
                    <line x1="0" y1="200" x2="160" y2="200" stroke="#10b981" strokeWidth="4" markerEnd="url(#arrow)" />
                    <text x="140" y="190" fill="#10b981" fontSize="10" fontWeight="bold">Vector B</text>
                    <line x1="0" y1="200" x2="100" y2="80" stroke="#6366f1" strokeWidth="4" markerEnd="url(#arrow)" />
                    <text x="80" y="70" fill="#6366f1" fontSize="10" fontWeight="bold">Vector A</text>
                    <line x1="100" y1="80" x2="100" y2="200" stroke="#475569" strokeDasharray="4" />
                    <line x1="0" y1="200" x2="100" y2="200" stroke="#f43f5e" strokeWidth="6" strokeOpacity="0.4" />
                    <path d="M 30 200 A 30 30 0 0 0 25 170" fill="none" stroke="#fbbf24" strokeWidth="2" />
                    <text x="35" y="180" fill="#fbbf24" fontSize="10" fontStyle="italic">θ</text>
                </svg>
                <div className="absolute top-0 right-0 p-4 text-[9px] font-mono text-slate-600 bg-slate-900 border border-slate-800 rounded">
                    Projection = |A| cos(θ)
                </div>
            </div>

            <div className="space-y-4 max-w-xs">
                <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
                    <h4 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-2">Geometric Meaning</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                        The dot product measures the <strong className="text-slate-200">aligned magnitude</strong> of two vectors. In ML, this translates to <strong className="text-emerald-400">Similarity</strong>.
                    </p>
                </div>
            </div>
        </div>
    );
};

export const FoundationsView: React.FC = () => {
  return (
    <motion.div 
      variants={MOTION_VARIANTS.container}
      initial="hidden"
      animate="show"
      className="space-y-24 pb-20"
    >
      <motion.header variants={MOTION_VARIANTS.item} className="border-b border-slate-800 pb-12">
        <h1 className="text-6xl font-serif font-bold text-white mb-6">Theoretical Foundations</h1>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed font-light">
          The rigorous mathematical architecture and core logic that enables machines to extract meaningful intelligence from raw datasets.
        </p>
      </motion.header>

      <motion.section variants={MOTION_VARIANTS.item} id="math-primer" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">01. Mathematical Engine</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <div className="space-y-12">
          <AlgorithmCard
              id="linear-algebra" title="Linear Algebra" complexity="Fundamental"
              theory="Data is structured as vectors and matrices. Linear Algebra provides the operations to transform these structures."
              math={<span>a &middot; b = &Sigma; a<sub>i</sub>b<sub>i</sub> = ||a|| ||b|| cos(&theta;)</span>} mathLabel="Vector Dot Product"
              code={`import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nsimilarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`}
              pros={['Foundational for Deep Learning', 'Mathematically elegant', 'Highly parallelizable']}
              cons={['Computationally expensive at scale', 'Susceptible to sparsity']}
          >
              <GeometricDotProduct />
          </AlgorithmCard>
        </div>
      </motion.section>
    </motion.div>
  );
};