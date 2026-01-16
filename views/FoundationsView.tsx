
import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { LatexRenderer } from '../components/LatexRenderer';
import { MOTION_VARIANTS } from '../constants';
import { ResponsiveContainer, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ComposedChart, ReferenceDot } from 'recharts';

// --- VISUALIZATIONS ---

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

const CalculusViz = () => {
    const [x0, setX0] = useState(2);
    
    const data = useMemo(() => {
        const points = [];
        // f(x) = x^2, f'(x) = 2x
        // Tangent line at x0: y = f(x0) + f'(x0)(x - x0)
        // y = x0^2 + 2x0(x - x0) = 2*x0*x - x0^2
        for (let x = -4; x <= 4; x += 0.5) {
            points.push({
                x,
                curve: x * x,
                tangent: (2 * x0 * x) - (x0 * x0)
            });
        }
        return points;
    }, [x0]);

    const slope = 2 * x0;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Point (x): <span className="text-indigo-400 ml-2">{x0}</span></label>
                    <input 
                        type="range" min="-3" max="3" step="0.5" 
                        value={x0} onChange={(e) => setX0(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
                    />
                </div>
                <div className="text-right">
                    <div className="text-[10px] text-slate-500 font-mono uppercase">Instantaneous Rate of Change</div>
                    <div className="text-2xl font-bold text-white font-mono">f'(x) = {slope.toFixed(1)}</div>
                </div>
            </div>

            <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" domain={[-4, 4]} stroke="#475569" fontSize={10} />
                        <YAxis type="number" domain={[-5, 16]} hide />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Line type="monotone" dataKey="curve" stroke="#6366f1" strokeWidth={3} dot={false} name="f(x) = x²" />
                        <Line type="monotone" dataKey="tangent" stroke="#f43f5e" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Tangent" />
                        <ReferenceDot x={x0} y={x0 * x0} r={6} fill="#fff" stroke="#6366f1" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-center text-slate-500 uppercase tracking-widest font-mono">
                The gradient points in the direction of steepest ascent (The slope of the tangent).
            </p>
        </div>
    );
};

const ProbabilityViz = () => {
    const [sigma, setSigma] = useState(1);
    
    const data = useMemo(() => {
        const points = [];
        const mu = 0;
        // Normal Distribution PDF
        for (let x = -5; x <= 5; x += 0.2) {
            const pdf = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
            points.push({ x, pdf });
        }
        return points;
    }, [sigma]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                 <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Uncertainty (σ): <span className="text-emerald-400 ml-2">{sigma}</span></label>
                    <input 
                        type="range" min="0.5" max="2.5" step="0.1" 
                        value={sigma} onChange={(e) => setSigma(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500 mt-2"
                    />
                </div>
                <div className="text-[10px] font-mono px-3 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400">
                    Normal Distribution
                </div>
            </div>

            <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 20, right: 0, bottom: 0, left: 0 }}>
                        <defs>
                            <linearGradient id="colorPdf" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="x" hide />
                        <YAxis hide />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Area type="monotone" dataKey="pdf" stroke="#10b981" fillOpacity={1} fill="url(#colorPdf)" />
                        <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                    </AreaChart>
                </ResponsiveContainer>
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-[9px] text-slate-500 font-mono">μ (Mean)</div>
            </div>
        </div>
    );
};

// --- MAIN VIEW ---

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

      {/* SECTION 1: LINEAR ALGEBRA */}
      <motion.section variants={MOTION_VARIANTS.item} id="linear-algebra" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">01. Linear Algebra</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="vector-spaces" title="Vectors & Matrices" complexity="Fundamental"
              theory="Data is structured as vectors (1D arrays), matrices (2D tables), and tensors (nD blocks). Linear Algebra provides the rulebook for transforming these structures to extract patterns, rotate spaces, and map inputs to outputs."
              math={<LatexRenderer formula="A \cdot x = \lambda x" />} mathLabel="Eigenvalue Equation"
              code={`import numpy as np
# Dot Product: Similarity
a = np.array([1, 2])
b = np.array([3, 4])
similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Matrix Multiplication: Transformation
W = np.array([[0.1, 0.5], [-0.3, 0.8]])
output = np.matmul(W, a)`}
              pros={['Foundational for Neural Networks', 'Mathematically elegant', 'Highly parallelizable (GPUs)']}
              cons={['Computationally expensive at scale O(n³)', 'Susceptible to sparsity issues']}
              steps={[
                "Launch Google Colab.",
                "Import NumPy: `import numpy as np`.",
                "Create vectors: `v = np.array([1, 2, 3])`.",
                "Perform operations: `np.dot(v1, v2)` for similarity, `np.matmul(matrix, vector)` for transformation.",
                "Experiment with `np.linalg.eig` to find eigenvalues."
              ]}
          >
              <GeometricDotProduct />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 2: CALCULUS */}
      <motion.section variants={MOTION_VARIANTS.item} id="calculus" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">02. Calculus & Gradients</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="gradients" title="Multivariate Calculus" complexity="Intermediate"
              theory="Calculus measures change. In Machine Learning, we need to know how changing a weight affects the error. The Gradient is a vector of partial derivatives pointing in the direction of steepest ascent. To learn, we move opposite to the gradient."
              math={<LatexRenderer formula="\nabla J(\theta) = \left[ \frac{\partial J}{\partial \theta_1}, \dots, \frac{\partial J}{\partial \theta_n} \right]^T" />} mathLabel="Gradient Vector"
              code={`import torch

# Automatic Differentiation (Autograd)
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # f(x) = x^2

y.backward() # Compute gradients
print(x.grad) # f'(2) = 2*2 = 4.0`}
              pros={['Enables Backpropagation', 'Universal optimization method', 'Handles millions of parameters']}
              cons={['Vanishing/Exploding gradients in deep nets', 'Can get stuck in local minima']}
              steps={[
                "Open Colab. Import `torch`.",
                "Create a tensor with gradient tracking: `x = torch.tensor(2.0, requires_grad=True)`.",
                "Define a function: `y = x**2 + 3*x`.",
                "Perform backpropagation: `y.backward()`.",
                "Inspect the gradient: `print(x.grad)`. This tells you how to adjust 'x' to minimize 'y'."
              ]}
          >
              <CalculusViz />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 3: PROBABILITY */}
      <motion.section variants={MOTION_VARIANTS.item} id="probability" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">03. Probability & Statistics</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="bayes" title="Bayesian Inference" complexity="Advanced"
              theory="Machine learning is essentially finding the most probable model given the data. Probability theory quantifies uncertainty. Bayes' Theorem allows us to update our beliefs (Prior) with new evidence (Likelihood) to form a new belief (Posterior)."
              math={<LatexRenderer formula="P(A|B) = \frac{P(B|A)P(A)}{P(B)}" />} mathLabel="Bayes' Theorem"
              code={`from scipy.stats import norm

# Probability Density Function
x = 0
mean, std = 0, 1
prob = norm.pdf(x, mean, std)

# Updating Beliefs
prior = 0.5
likelihood = 0.8
posterior = (likelihood * prior) / 1.0 # simplified`}
              pros={['Handles uncertainty explicitly', 'Robust with small data', 'Prevents overfitting via priors']}
              cons={['Computationally intractable integrals', 'Requires prior assumptions']}
              steps={[
                "Use `scipy.stats` in Colab.",
                "Define a prior probability distribution.",
                "Observe new data points.",
                "Multiply Prior by Likelihood to get Posterior.",
                "Visualize the distribution shift using `matplotlib`."
              ]}
          >
              <ProbabilityViz />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 4: OPTIMIZATION */}
      <motion.section variants={MOTION_VARIANTS.item} id="optimization" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">04. Optimization</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8 hover:border-indigo-500/30 transition-colors">
            <h3 className="text-2xl font-bold text-white mb-4">The Quest for Minima</h3>
            <p className="text-slate-400 leading-relaxed mb-6">
                Optimization is the engine that drives learning. It involves traversing a high-dimensional "Loss Landscape" to find the set of parameters that minimizes error. 
                Algorithms like <strong>Gradient Descent</strong> and <strong>Adam</strong> navigate these valleys.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
                    <h4 className="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-2">Convex Optimization</h4>
                    <p className="text-xs text-slate-500">
                        Problems where there is only one global minimum (e.g., Linear Regression). These are easy to solve and guaranteed to converge.
                    </p>
                </div>
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
                    <h4 className="text-sm font-bold text-rose-400 uppercase tracking-widest mb-2">Non-Convex Optimization</h4>
                    <p className="text-xs text-slate-500">
                        Deep Learning landscapes are rugged with many local minima and saddle points. We need adaptive learning rates to navigate them.
                    </p>
                </div>
            </div>
            <div className="mt-8 pt-6 border-t border-slate-800 text-center">
                <p className="text-slate-500 text-sm">See the <strong className="text-white">Optimization Engines</strong> module for interactive simulations.</p>
            </div>
        </div>
      </motion.section>
    </motion.div>
  );
};
