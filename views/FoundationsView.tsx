import React, { useState, useEffect, useRef, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, ComposedChart, Scatter, ScatterChart, BarChart, Bar, Legend, ReferenceLine, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ReferenceDot } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { CodeBlock } from '../components/CodeBlock';
import { Play, RotateCcw, FastForward, Activity, Triangle } from 'lucide-react';

// --- DATA FOR CHARTS ---

const learningDefData = Array.from({ length: 40 }, (_, i) => {
  const x = (i / 40) * 10; 
  const trueFx = Math.sin(x);
  const noise = (Math.random() - 0.5) * 0.5;
  const y = trueFx + noise;
  const learnedFx = Math.sin(x) * 0.95 + 0.05;
  return { x, y, trueFx, learnedFx };
});

const biasVarianceData = Array.from({ length: 50 }, (_, i) => {
  const complexity = (i / 2.5) + 1;
  const biasSq = 60 * Math.exp(-0.45 * complexity); 
  const variance = 0.5 * Math.pow(complexity, 1.6);
  const noise = 5;
  const totalError = biasSq + variance + noise;
  return { complexity, biasSq, variance, totalError };
});

const learningRateData = Array.from({ length: 30 }, (_, i) => {
  const epoch = i;
  const goodLR = 2 * Math.exp(-0.2 * i);
  const badLR = 2 * Math.exp(-0.05 * i) + 0.3 * Math.sin(i);
  return { epoch, goodLR, badLR };
});

const splitData = [{ name: 'Dataset', Train: 70, Validation: 15, Test: 15 }];

const scalingData = [
  ...Array.from({ length: 15 }, () => ({ x: Math.random() * 1000 + 1000, y: Math.random() * 5000, type: 'Raw' })),
  ...Array.from({ length: 15 }, () => ({ x: (Math.random() * 4 - 2) * 200 + 1500, y: (Math.random() * 4 - 2) * 500 + 2500, type: 'Scaled' }))
];

const normalDistData = Array.from({ length: 50 }, (_, i) => {
  const x = (i - 25) / 5;
  const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);
  return { x, y };
});

const typesLearningData = [
  { subject: 'Labeled Data', Supervised: 100, Unsupervised: 0, Reinforcement: 20 },
  { subject: 'Patterns', Supervised: 30, Unsupervised: 100, Reinforcement: 60 },
  { subject: 'Real-time', Supervised: 10, Unsupervised: 0, Reinforcement: 100 },
  { subject: 'Accuracy', Supervised: 100, Unsupervised: 40, Reinforcement: 80 },
  { subject: 'Humans', Supervised: 100, Unsupervised: 20, Reinforcement: 50 },
];

const INITIAL_BOARD = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9]
];

// --- VIZ COMPONENTS ---

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
                    {/* Grid */}
                    <line x1="0" y1="200" x2="200" y2="200" stroke="#1e293b" />
                    <line x1="0" y1="0" x2="0" y2="200" stroke="#1e293b" />
                    
                    {/* Vector B (Horizontal reference) */}
                    <line x1="0" y1="200" x2="160" y2="200" stroke="#10b981" strokeWidth="4" markerEnd="url(#arrow)" />
                    <text x="140" y="190" fill="#10b981" fontSize="10" fontWeight="bold">Vector B</text>

                    {/* Vector A */}
                    <line x1="0" y1="200" x2="100" y2="80" stroke="#6366f1" strokeWidth="4" markerEnd="url(#arrow)" />
                    <text x="80" y="70" fill="#6366f1" fontSize="10" fontWeight="bold">Vector A</text>

                    {/* Projection Line */}
                    <line x1="100" y1="80" x2="100" y2="200" stroke="#475569" strokeDasharray="4" />
                    
                    {/* Projected Vector */}
                    <line x1="0" y1="200" x2="100" y2="200" stroke="#f43f5e" strokeWidth="6" strokeOpacity="0.4" />
                    
                    {/* Angle Arc */}
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
                        The dot product measures the <strong className="text-slate-200">aligned magnitude</strong> of two vectors. In ML, this translates to <strong className="text-emerald-400">Similarity</strong>. If the dot product is high, the vectors point in the same direction.
                    </p>
                </div>
                <div className="bg-indigo-500/5 p-4 rounded-xl border border-indigo-500/20">
                    <code className="text-xs font-mono text-indigo-300">
                        similarity = a·b / (|a||b|)
                    </code>
                    <p className="text-[10px] text-slate-500 mt-2 italic">This is the basis of Cosine Similarity used in search engines and LLMs.</p>
                </div>
            </div>
        </div>
    );
};

const BiasVarianceViz = () => {
    const [complexity, setComplexity] = useState(8.5); 
    const idx = useMemo(() => Math.min(Math.max(0, Math.round((complexity - 1) * 2.5)), biasVarianceData.length - 1), [complexity]);
    const point = biasVarianceData[idx];

    let zoneLabel = "Optimal Balance";
    let zoneColor = "#10b981"; 
    if (complexity < 6) { zoneLabel = "Underfitting (High Bias)"; zoneColor = "#ef4444"; }
    else if (complexity > 13) { zoneLabel = "Overfitting (High Variance)"; zoneColor = "#818cf8"; }

    return (
        <div className="space-y-6">
            <div className="h-80 w-full bg-slate-950 p-4 rounded-2xl border border-slate-900">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={biasVarianceData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="complexity" type="number" domain={[1, 20]} hide />
                        <YAxis hide />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px' }} />
                        <ReferenceLine x={complexity} stroke={zoneColor} strokeWidth={2} strokeDasharray="4 4" />
                        <Line name="Bias²" type="monotone" dataKey="biasSq" stroke="#f472b6" strokeWidth={2} dot={false} />
                        <Line name="Variance" type="monotone" dataKey="variance" stroke="#818cf8" strokeWidth={2} dot={false} />
                        <Line name="Total Error" type="monotone" dataKey="totalError" stroke="#ffffff" strokeWidth={4} dot={false} />
                        <ReferenceDot x={point.complexity} y={point.totalError} r={8} fill={zoneColor} stroke="#fff" strokeWidth={2} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row gap-6 items-center">
                <div className="flex-1 w-full">
                    <div className="flex justify-between mb-2">
                        <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Model Complexity</span>
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color: zoneColor, backgroundColor: `${zoneColor}20`, border: `1px solid ${zoneColor}30` }}>{zoneLabel}</span>
                    </div>
                    <input type="range" min="1" max="20" step="0.5" value={complexity} onChange={(e) => setComplexity(parseFloat(e.target.value))} className="w-full h-2 bg-slate-800 rounded-lg accent-indigo-500" />
                </div>
                <div className="grid grid-cols-2 gap-4 w-48 text-center">
                    <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <div className="text-[8px] text-slate-600 uppercase">Bias</div>
                        <div className="text-sm font-bold text-slate-300">{point.biasSq.toFixed(1)}</div>
                    </div>
                    <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <div className="text-[8px] text-slate-600 uppercase">Variance</div>
                        <div className="text-sm font-bold text-slate-300">{point.variance.toFixed(1)}</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const OneHotViz = () => {
  const [activeCategory, setActiveCategory] = useState<string | null>("Blue");
  const categories = [
    { name: 'Red', color: 'text-rose-400', border: 'border-rose-500/40' },
    { name: 'Blue', color: 'text-indigo-400', border: 'border-indigo-500/40' },
    { name: 'Green', color: 'text-emerald-400', border: 'border-emerald-500/40' }
  ];
  return (
    <div className="space-y-8 py-4">
       <div className="flex justify-center gap-3">
          {categories.map((cat) => (
            <button key={cat.name} onClick={() => setActiveCategory(cat.name)} className={`px-4 py-2 rounded-lg border text-xs font-bold transition-all ${activeCategory === cat.name ? `bg-slate-800 border-indigo-500 text-indigo-400 shadow-lg shadow-indigo-900/20` : 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300'}`}>
              {cat.name}
            </button>
          ))}
       </div>
       <div className="flex items-center justify-center gap-6">
          <div className="w-24 h-24 rounded-2xl bg-slate-900 border-2 border-slate-800 flex items-center justify-center text-xl font-bold text-slate-500 shadow-inner italic">"{activeCategory}"</div>
          <div className="text-2xl text-slate-700">&rarr;</div>
          <div className="flex gap-2 p-4 bg-slate-950 rounded-2xl border border-slate-800 shadow-xl">
             {categories.map(cat => (
                 <div key={cat.name} className="flex flex-col items-center gap-2">
                    <span className="text-[8px] text-slate-600 font-mono font-bold uppercase">{cat.name}</span>
                    <div className={`w-12 h-14 flex items-center justify-center rounded-xl border-2 font-mono text-2xl font-black transition-all ${activeCategory === cat.name ? 'bg-indigo-500/10 border-indigo-500 text-indigo-400 shadow-lg shadow-indigo-500/10' : 'bg-slate-900 border-slate-800 text-slate-800'}`}>
                        {activeCategory === cat.name ? 1 : 0}
                    </div>
                 </div>
             ))}
          </div>
       </div>
    </div>
  );
};

export const FoundationsView: React.FC = () => {
  return (
    <div className="space-y-24 animate-fade-in pb-20">
      <header className="border-b border-slate-800 pb-12">
        <h1 className="text-6xl font-serif font-bold text-white mb-6">Theoretical Foundations</h1>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed font-light">
          The rigorous mathematical architecture and core logic that enables machines to extract meaningful intelligence from raw datasets.
        </p>
      </header>

      <section id="math-primer" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">01. Mathematical Engine</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <div className="space-y-12">
          <AlgorithmCard
              id="linear-algebra" title="Linear Algebra" complexity="Fundamental"
              theory="Data is structured as vectors and matrices. Linear Algebra provides the operations to transform these structures. The Dot Product is the fundamental operation for measuring feature similarity."
              math={<span>a &middot; b = &Sigma; a<sub>i</sub>b<sub>i</sub> = ||a|| ||b|| cos(&theta;)</span>} mathLabel="Vector Dot Product"
              code={`import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nsimilarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`}
              pros={['Foundational for Deep Learning', 'Mathematically elegant', 'Highly parallelizable']}
              cons={['Computationally expensive at scale', 'Susceptible to sparsity']}
          >
              <GeometricDotProduct />
          </AlgorithmCard>
          <AlgorithmCard
              id="calculus" title="Differential Calculus" complexity="Intermediate"
              theory="Calculus is how models learn. By taking the derivative of a loss function, we find the 'slope'—the direction of parameter change that reduces error most effectively."
              math={<span>f'(x) = lim<sub>h&rarr;0</sub> [f(x+h) - f(x)] / h</span>} mathLabel="Instantaneous Rate of Change"
              code={`def loss(w): return w**2\ndef gradient(w): return 2*w # Derivative\nw = 10\nw = w - 0.1 * gradient(w) # Single step of learning`}
              pros={['Enables non-linear optimization', 'Solid deterministic theory', 'Basis for Backpropagation']}
              cons={['Vanishing/Exploding gradients in deep networks', 'Requires differentiability']}
          />
        </div>
      </section>

      <section id="bias-variance" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">02. The Bias-Variance Dilemma</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
            id="bias-variance-card" title="Error Decomposition" complexity="Intermediate"
            theory="Total prediction error consists of Bias (underfitting), Variance (overfitting), and Noise. The goal is to minimize total error by finding the 'sweet spot' in model complexity."
            math={<span>E[Error] = Bias<sup>2</sup> + Variance + &epsilon;</span>} mathLabel="Generalization Error"
            code={`# High Bias: Too simple\nmodel = LinearRegression()\n# High Variance: Too complex\nmodel = PolynomialFeatures(degree=20)`}
            pros={['Provides diagnostic framework', 'Explains model failures', 'Guides regularization strategy']}
            cons={['Theoretical abstraction', 'Cannot be measured precisely in the wild']}
        >
            <BiasVarianceViz />
        </AlgorithmCard>
      </section>

      <section id="feature-engineering" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">03. Feature Representation</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         <AlgorithmCard
           id="feature-eng-card" title="Categorical Encoding" complexity="Fundamental"
           theory="Machines only speak numbers. Categorical data (like colors or cities) must be mapped into numerical vectors. One-Hot encoding prevents the model from assuming an artificial order between categories."
           math={<span>v = [0, \dots, 1, \dots, 0]</span>} mathLabel="One-Hot Sparse Vector"
           code={`import pandas as pd\ndf = pd.get_dummies(df, columns=['category'])`}
           pros={['Allows non-numeric data processing', 'Preserves category independence', 'Simple to implement']}
           cons={['Sparse vectors consume memory', 'Curse of Dimensionality if unique values are high']}
         >
           <OneHotViz />
         </AlgorithmCard>
      </section>
    </div>
  );
};