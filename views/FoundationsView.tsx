
import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { LatexRenderer } from '../components/LatexRenderer';
import { MOTION_VARIANTS } from '../constants';
import { ResponsiveContainer, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ComposedChart, ReferenceDot } from 'recharts';

// --- VISUALIZATIONS ---

const GeometricDotProduct = () => {
    return (
        <div className="flex flex-col md:flex-row items-center justify-center gap-12 py-12 bg-surface border border-border-strong rounded-none shadow-inner relative overflow-hidden">
            <div className="absolute top-4 left-6 text-[8px] font-mono text-text-muted uppercase tracking-[0.4em]">VECTOR_SPACE_PROJECTION_v2.0</div>
            <div className="relative w-64 h-64 border-l border-b border-border-strong">
                <svg width="100%" height="100%" viewBox="0 0 200 200">
                    <defs>
                        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                            <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
                        </marker>
                    </defs>
                    <line x1="0" y1="200" x2="200" y2="200" stroke="var(--border-strong)" strokeOpacity="0.5" />
                    <line x1="0" y1="0" x2="0" y2="200" stroke="var(--border-strong)" strokeOpacity="0.5" />
                    <line x1="0" y1="200" x2="160" y2="200" stroke="var(--brand)" strokeWidth="2" markerEnd="url(#arrow)" className="text-brand" />
                    <text x="140" y="190" fill="var(--brand)" fontSize="8" fontStyle="italic" className="font-mono font-black uppercase tracking-widest">V_B</text>
                    <line x1="0" y1="200" x2="100" y2="80" stroke="var(--text-primary)" strokeWidth="2" markerEnd="url(#arrow)" className="text-text-primary" />
                    <text x="80" y="70" fill="var(--text-primary)" fontSize="8" fontStyle="italic" className="font-mono font-black uppercase tracking-widest">V_A</text>
                    <line x1="100" y1="80" x2="100" y2="200" stroke="var(--text-muted)" strokeDasharray="4" />
                    <line x1="0" y1="200" x2="100" y2="200" stroke="var(--brand)" strokeWidth="4" strokeOpacity="0.3" />
                    <path d="M 30 200 A 30 30 0 0 0 25 170" fill="none" stroke="var(--text-muted)" strokeWidth="1" />
                    <text x="35" y="180" fill="var(--text-muted)" fontSize="10" fontStyle="italic">θ</text>
                </svg>
                <div className="absolute top-0 right-0 p-3 text-[9px] font-mono text-text-muted bg-app border border-border-strong rounded-none">
                    PROJ = |A| cos(θ)
                </div>
            </div>

            <div className="space-y-6 max-w-xs">
                <div className="bg-app p-6 border border-border-strong rounded-none">
                    <h4 className="text-[9px] font-mono font-black text-brand uppercase tracking-[0.2em] mb-3">GEOMETRIC_INTUITION</h4>
                    <p className="text-[11px] text-text-secondary leading-relaxed font-mono uppercase tracking-tight">
                        The dot product measures the <strong className="text-text-primary">aligned magnitude</strong> of two vectors. In ML, this translates to <strong className="text-brand">Similarity</strong>.
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
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row justify-between items-center bg-app p-6 border border-border-strong rounded-none gap-6">
                <div className="w-full sm:w-1/2">
                    <label className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.2em]">INPUT_X: <span className="text-brand ml-2">{x0.toFixed(1)}</span></label>
                    <input 
                        type="range" min="-3" max="3" step="0.5" 
                        value={x0} onChange={(e) => setX0(Number(e.target.value))}
                        className="w-full h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand mt-4"
                    />
                </div>
                <div className="text-right w-full sm:w-auto">
                    <div className="text-[9px] text-text-muted font-mono font-black uppercase tracking-widest mb-1">INSTANTANEOUS_GRADIENT</div>
                    <div className="text-3xl font-display font-black text-text-primary uppercase tracking-tighter">f'(x) = {slope.toFixed(1)}</div>
                </div>
            </div>

            <div className="h-72 w-full bg-surface border border-border-strong rounded-none p-4 relative">
                <div className="absolute top-4 right-6 text-[8px] font-mono text-text-muted uppercase tracking-[0.4em]">GRADIENT_DESCENT_PATH_v1.0</div>
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-strong)" strokeOpacity="0.3" />
                        <XAxis type="number" dataKey="x" domain={[-4, 4]} stroke="var(--text-muted)" fontSize={9} fontStyle="italic" />
                        <YAxis type="number" domain={[-5, 16]} hide />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', borderRadius: '0px', fontSize: '10px', fontFamily: 'monospace' }} 
                            itemStyle={{ color: 'var(--text-primary)' }}
                        />
                        <Line type="monotone" dataKey="curve" stroke="var(--brand)" strokeWidth={2} dot={false} name="f(x) = x²" />
                        <Line type="monotone" dataKey="tangent" stroke="var(--text-muted)" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Tangent" />
                        <ReferenceDot x={x0} y={x0 * x0} r={4} fill="var(--text-primary)" stroke="var(--brand)" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[9px] text-center text-text-muted uppercase tracking-[0.3em] font-mono font-black">
                The gradient points in the direction of steepest ascent.
            </p>
        </div>
    );
};

const ProbabilityViz = () => {
    const [sigma, setSigma] = useState(1);
    
    const data = useMemo(() => {
        const points = [];
        const mu = 0;
        for (let x = -5; x <= 5; x += 0.2) {
            const pdf = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
            points.push({ x, pdf });
        }
        return points;
    }, [sigma]);

    return (
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row justify-between items-center bg-app p-6 border border-border-strong rounded-none gap-6">
                 <div className="w-full sm:w-1/2">
                    <label className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.2em]">UNCERTAINTY_SIGMA: <span className="text-brand ml-2">{sigma.toFixed(1)}</span></label>
                    <input 
                        type="range" min="0.5" max="2.5" step="0.1" 
                        value={sigma} onChange={(e) => setSigma(Number(e.target.value))}
                        className="w-full h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand mt-4"
                    />
                </div>
                <div className="text-[9px] font-mono font-black px-4 py-2 bg-brand/5 border border-brand/20 text-brand uppercase tracking-[0.2em]">
                    GAUSSIAN_DISTRIBUTION
                </div>
            </div>

            <div className="h-72 w-full bg-surface border border-border-strong rounded-none p-4 relative">
                <div className="absolute top-4 right-6 text-[8px] font-mono text-text-muted uppercase tracking-[0.4em]">PROBABILITY_DENSITY_FUNCTION</div>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 20, right: 0, bottom: 0, left: 0 }}>
                        <defs>
                            <linearGradient id="colorPdf" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--brand)" stopOpacity={0.2}/>
                                <stop offset="95%" stopColor="var(--brand)" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="x" hide />
                        <YAxis hide />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', borderRadius: '0px', fontSize: '10px', fontFamily: 'monospace' }} 
                        />
                        <Area type="monotone" dataKey="pdf" stroke="var(--brand)" strokeWidth={2} fillOpacity={1} fill="url(#colorPdf)" />
                        <ReferenceLine x={0} stroke="var(--border-strong)" strokeDasharray="3 3" />
                    </AreaChart>
                </ResponsiveContainer>
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-[9px] text-text-muted font-mono font-black uppercase tracking-widest">μ (MEAN)</div>
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
      className="space-y-32 pb-32"
    >
      <motion.header variants={MOTION_VARIANTS.item} className="border-b border-border-strong pb-16">
        <div className="flex items-center gap-4 mb-8">
           <span className="text-[10px] font-mono font-black text-brand uppercase tracking-[0.4em]">CORE_ARCHITECTURE_V1.0</span>
           <div className="h-px w-12 bg-brand" />
        </div>
        <h1 className="text-5xl md:text-8xl font-display font-black text-text-primary mb-10 leading-none uppercase tracking-tight">
          THEORETICAL_FOUNDATIONS
        </h1>
        <p className="text-text-secondary text-xl max-w-3xl leading-relaxed font-light italic">
          The rigorous mathematical architecture and core logic that enables machines to extract meaningful intelligence from raw datasets.
        </p>
      </motion.header>

      {/* SECTION 1: LINEAR ALGEBRA */}
      <motion.section variants={MOTION_VARIANTS.item} id="linear-algebra" className="scroll-mt-24">
        <div className="flex items-center gap-6 mb-16">
            <h2 className="text-3xl font-display font-black text-text-primary uppercase tracking-tight">01 // LINEAR_ALGEBRA</h2>
            <div className="h-px bg-border-strong flex-1"></div>
        </div>
        <AlgorithmCard
              id="vector-spaces" title="VECTORS_&_MATRICES" complexity="Fundamental"
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
        <div className="flex items-center gap-6 mb-16">
            <h2 className="text-3xl font-display font-black text-text-primary uppercase tracking-tight">02 // CALCULUS_&_GRADIENTS</h2>
            <div className="h-px bg-border-strong flex-1"></div>
        </div>
        <AlgorithmCard
              id="gradients" title="MULTIVARIATE_CALCULUS" complexity="Intermediate"
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
        <div className="flex items-center gap-6 mb-16">
            <h2 className="text-3xl font-display font-black text-text-primary uppercase tracking-tight">03 // PROBABILITY_&_STATISTICS</h2>
            <div className="h-px bg-border-strong flex-1"></div>
        </div>
        <AlgorithmCard
              id="bayes" title="BAYESIAN_INFERENCE" complexity="Advanced"
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
         <div className="flex items-center gap-6 mb-16">
            <h2 className="text-3xl font-display font-black text-text-primary uppercase tracking-tight">04 // OPTIMIZATION</h2>
            <div className="h-px bg-border-strong flex-1"></div>
        </div>
        <div className="bg-surface border border-border-strong rounded-none p-12 relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-1 h-full bg-brand" />
            <h3 className="text-3xl font-display font-black text-text-primary mb-6 uppercase tracking-tight">THE_QUEST_FOR_MINIMA</h3>
            <p className="text-text-secondary leading-relaxed mb-10 font-light italic max-w-3xl">
                Optimization is the engine that drives learning. It involves traversing a high-dimensional "Loss Landscape" to find the set of parameters that minimizes error. 
                Algorithms like <strong>Gradient Descent</strong> and <strong>Adam</strong> navigate these valleys.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-app p-8 border border-border-strong">
                    <h4 className="text-[10px] font-mono font-black text-brand uppercase tracking-[0.2em] mb-4">CONVEX_OPTIMIZATION</h4>
                    <p className="text-xs text-text-secondary leading-relaxed font-light">
                        Problems where there is only one global minimum (e.g., Linear Regression). These are easy to solve and guaranteed to converge.
                    </p>
                </div>
                <div className="bg-app p-8 border border-border-strong">
                    <h4 className="text-[10px] font-mono font-black text-rose-500 uppercase tracking-[0.2em] mb-4">NON-CONVEX_OPTIMIZATION</h4>
                    <p className="text-xs text-text-secondary leading-relaxed font-light">
                        Deep Learning landscapes are rugged with many local minima and saddle points. We need adaptive learning rates to navigate them.
                    </p>
                </div>
            </div>
            <div className="mt-12 pt-8 border-t border-border-strong text-center">
                <p className="text-text-muted text-[10px] font-mono font-black uppercase tracking-widest">See the <strong className="text-text-primary">OPTIMIZATION_ENGINES</strong> module for interactive simulations.</p>
            </div>
        </div>
      </motion.section>
    </motion.div>
  );
};
