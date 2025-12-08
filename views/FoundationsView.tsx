import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, ComposedChart, Scatter, ScatterChart, BarChart, Bar, Legend, ReferenceLine, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ReferenceArea } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { CodeBlock } from '../components/CodeBlock';
import { MathBlock } from '../components/MathBlock';

// --- DATA FOR CHARTS ---

// 1. Definition of Learning (True Function vs Noise)
const learningDefData = Array.from({ length: 40 }, (_, i) => {
  const x = (i / 40) * 10; // 0 to 10
  const trueFx = Math.sin(x);
  const noise = (Math.random() - 0.5) * 0.5;
  const y = trueFx + noise;
  const learnedFx = Math.sin(x) * 0.95 + 0.05; // Slightly off
  return { x, y, trueFx, learnedFx };
});

// 2. Bias-Variance Tradeoff (Refined for clearer U-Shape)
const biasVarianceData = Array.from({ length: 50 }, (_, i) => {
  const complexity = (i / 2.5) + 1; // Range 1 to ~20
  // Bias drops sharply as complexity increases
  const biasSq = 60 * Math.exp(-0.45 * complexity); 
  // Variance rises slowly then sharply
  const variance = 0.5 * Math.pow(complexity, 1.6);
  // Irreducible error constant
  const noise = 5;
  const totalError = biasSq + variance + noise;
  return { complexity, biasSq, variance, totalError };
});

// 3. Optimization: Good vs Bad Learning Rates
const learningRateData = Array.from({ length: 30 }, (_, i) => {
  const epoch = i;
  const goodLR = 2 * Math.exp(-0.2 * i);
  const badLR = 2 * Math.exp(-0.05 * i) + 0.3 * Math.sin(i); // Slow & Oscillating
  return { epoch, goodLR, badLR };
});

// 4. Data Splitting
const splitData = [
  { name: 'Dataset', Train: 70, Validation: 15, Test: 15 }
];

// 5. Feature Scaling (Raw vs Scaled)
// Generating two clusters to visually demonstrate scaling
const scalingData = [
  // Raw Data (Large Scale)
  ...Array.from({ length: 10 }, () => ({ x: Math.random() * 1000 + 1000, y: Math.random() * 5000, type: 'Raw (High Variance)' })),
  // Scaled Data (Standardized)
  ...Array.from({ length: 10 }, () => ({ x: Math.random() * 4 - 2, y: Math.random() * 4 - 2, type: 'Scaled (Z-Score)' }))
];

// 6. Normal Distribution Data (Math Primer)
const normalDistData = Array.from({ length: 50 }, (_, i) => {
  const x = (i - 25) / 5; // Range -5 to 5
  const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);
  return { x, y };
});

// 7. Derivative / Tangent Visualization (y = x^2 at x=2)
const parabolaData = Array.from({ length: 41 }, (_, i) => {
  const x = (i - 20) / 5; // -4 to 4
  const y = x * x;
  const tangent = 4 * x - 4;
  return { x, y, tangent: (x > 0 && x < 4) ? tangent : null }; 
});

// 8. Types of Learning (Radar Chart Data)
const typesLearningData = [
  { subject: 'Labeled Data Needed', Supervised: 100, Unsupervised: 0, Reinforcement: 20 },
  { subject: 'Pattern Discovery', Supervised: 30, Unsupervised: 100, Reinforcement: 60 },
  { subject: 'Real-time Feedback', Supervised: 10, Unsupervised: 0, Reinforcement: 100 },
  { subject: 'Prediction Accuracy', Supervised: 100, Unsupervised: 40, Reinforcement: 80 },
  { subject: 'Human Supervision', Supervised: 100, Unsupervised: 20, Reinforcement: 50 },
];

// --- SUB-COMPONENTS ---

const DotProductViz = () => (
  <div className="flex flex-col md:flex-row items-center justify-center gap-8 py-6 select-none">
    {/* Vector A */}
    <div className="flex flex-col gap-1">
      <div className="text-xs text-center text-indigo-400 font-mono mb-1">Vector A</div>
      <div className="bg-slate-800 border border-indigo-500 rounded p-1 flex md:flex-col gap-1">
        {[1, 3, -5].map((val, i) => (
          <div key={i} className="w-10 h-10 flex items-center justify-center bg-slate-900 text-white font-mono text-sm border border-slate-700">
            {val}
          </div>
        ))}
      </div>
    </div>
    <div className="text-2xl text-slate-500 font-bold">·</div>
    {/* Vector B */}
    <div className="flex flex-col gap-1">
      <div className="text-xs text-center text-emerald-400 font-mono mb-1">Vector B</div>
      <div className="bg-slate-800 border border-emerald-500 rounded p-1 flex md:flex-col gap-1">
        {[4, -2, -1].map((val, i) => (
          <div key={i} className="w-10 h-10 flex items-center justify-center bg-slate-900 text-white font-mono text-sm border border-slate-700">
            {val}
          </div>
        ))}
      </div>
    </div>
    <div className="text-2xl text-slate-500 font-bold">=</div>
    {/* Calculation */}
    <div className="bg-slate-900 border border-slate-700 rounded p-4 shadow-lg">
      <div className="flex flex-col gap-2 font-mono text-sm text-slate-300">
        <div className="flex items-center gap-2">
            <span className="text-indigo-400">(1)</span> * <span className="text-emerald-400">(4)</span> = 4
        </div>
        <div className="flex items-center gap-2">
            <span className="text-indigo-400">(3)</span> * <span className="text-emerald-400">(-2)</span> = -6
        </div>
        <div className="flex items-center gap-2">
            <span className="text-indigo-400">(-5)</span> * <span className="text-emerald-400">(-1)</span> = 5
        </div>
        <div className="h-px bg-slate-600 w-full my-1"></div>
        <div className="text-right text-white font-bold text-lg">
            Result: <span className="text-fuchsia-400">3</span>
        </div>
      </div>
    </div>
  </div>
);

const DataFrameViz = () => (
  <div className="overflow-hidden rounded-lg border border-slate-700 shadow-md">
    <table className="w-full text-sm text-left text-slate-400">
      <thead className="text-xs text-slate-200 uppercase bg-slate-800">
        <tr>
          <th className="px-4 py-2 border-r border-slate-700">Index</th>
          <th className="px-4 py-2 text-indigo-400">Feature_1</th>
          <th className="px-4 py-2 text-emerald-400">Feature_2</th>
          <th className="px-4 py-2 text-rose-400">Target</th>
        </tr>
      </thead>
      <tbody className="bg-slate-900">
        <tr className="border-b border-slate-800 hover:bg-slate-800/50 transition-colors">
          <td className="px-4 py-2 font-mono text-slate-500 border-r border-slate-800">0</td>
          <td className="px-4 py-2">1.2</td>
          <td className="px-4 py-2">0.5</td>
          <td className="px-4 py-2">0</td>
        </tr>
        <tr className="border-b border-slate-800 hover:bg-slate-800/50 transition-colors">
          <td className="px-4 py-2 font-mono text-slate-500 border-r border-slate-800">1</td>
          <td className="px-4 py-2">2.4</td>
          <td className="px-4 py-2">0.8</td>
          <td className="px-4 py-2">1</td>
        </tr>
        <tr className="hover:bg-slate-800/50 transition-colors">
          <td className="px-4 py-2 font-mono text-slate-500 border-r border-slate-800">2</td>
          <td className="px-4 py-2">3.1</td>
          <td className="px-4 py-2">1.2</td>
          <td className="px-4 py-2">1</td>
        </tr>
      </tbody>
    </table>
    <div className="bg-slate-800 p-2 text-xs text-center text-slate-500 font-mono">
      df.shape = (3, 3)
    </div>
  </div>
);

const DataStructuresViz = () => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs select-none">
    {/* List Viz */}
    <div className="bg-slate-900 border border-slate-700 rounded p-3 flex flex-col gap-2 relative shadow-sm hover:border-indigo-500/50 transition-colors">
       <div className="text-indigo-400 font-bold border-b border-slate-800 pb-1 flex justify-between">
          <span>List</span>
          <span className="text-[10px] text-slate-500 font-normal">Ordered, Mutable</span>
       </div>
       <div className="flex items-center gap-1 overflow-x-auto py-1">
          {[10, 20, 30].map((v, i) => (
             <React.Fragment key={i}>
                <div className="w-8 h-8 flex flex-shrink-0 items-center justify-center bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 rounded hover:bg-indigo-500/20">
                   {v}
                </div>
                {i < 2 && <div className="text-slate-600">→</div>}
             </React.Fragment>
          ))}
       </div>
       <div className="text-slate-500 text-[10px] italic">Batch of data points</div>
    </div>

    {/* Dict Viz */}
    <div className="bg-slate-900 border border-slate-700 rounded p-3 flex flex-col gap-2 relative shadow-sm hover:border-emerald-500/50 transition-colors">
       <div className="text-emerald-400 font-bold border-b border-slate-800 pb-1 flex justify-between">
          <span>Dict</span>
          <span className="text-[10px] text-slate-500 font-normal">Key-Value Map</span>
       </div>
       <div className="flex flex-col gap-1.5 py-1">
          <div className="flex justify-between items-center bg-emerald-900/10 p-1 px-2 rounded border border-emerald-500/20">
             <span className="text-emerald-200">"lr"</span>
             <span className="text-slate-500 text-[10px]">:</span>
             <span className="text-emerald-400 font-bold">0.01</span>
          </div>
          <div className="flex justify-between items-center bg-emerald-900/10 p-1 px-2 rounded border border-emerald-500/20">
             <span className="text-emerald-200">"opt"</span>
             <span className="text-slate-500 text-[10px]">:</span>
             <span className="text-emerald-400 font-bold">"Adam"</span>
          </div>
       </div>
       <div className="text-slate-500 text-[10px] italic">Model Configuration</div>
    </div>

    {/* Set Viz */}
    <div className="bg-slate-900 border border-slate-700 rounded p-3 flex flex-col gap-2 relative shadow-sm hover:border-rose-500/50 transition-colors">
       <div className="text-rose-400 font-bold border-b border-slate-800 pb-1 flex justify-between">
          <span>Set</span>
          <span className="text-[10px] text-slate-500 font-normal">Unique Items</span>
       </div>
       <div className="flex flex-wrap gap-1.5 py-1 justify-center">
          {['cat', 'dog', 'bird'].map((v) => (
             <div key={v} className="px-2 py-0.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-full text-[10px] hover:bg-rose-500/20">
                {v}
             </div>
          ))}
       </div>
       <div className="text-slate-500 text-[10px] italic">Unique Vocabulary</div>
    </div>
  </div>
);

export const FoundationsView: React.FC = () => {
  return (
    <div className="space-y-24 animate-fade-in pb-12">
      <header className="border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Theoretical Foundations</h1>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed">
          The rigorous mathematical framework and core concepts that enable machines to learn from data.
        </p>
      </header>

      {/* --- SECTION 1: MATH PRIMER --- */}
      <section id="math-primer" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">01</span>
            <h2 className="text-2xl font-bold text-indigo-400 uppercase tracking-widest">Mathematical Prerequisites</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <div className="grid grid-cols-1 gap-12">
          <AlgorithmCard
              id="linear-algebra"
              title="Linear Algebra"
              theory="Linear algebra provides the language for representing data. Vectors represent features, matrices represent datasets, and operations like the dot product measure similarity."
              math={<span>a &middot; b = &Sigma;<sub>i=1</sub><sup>n</sup> a<sub>i</sub>b<sub>i</sub> = ||a|| ||b|| cos(&theta;)</span>}
              mathLabel="Dot Product"
              code={`import numpy as np

a = np.array([1, 3, -5])
b = np.array([4, -2, -1])

# Dot Product
result = np.dot(a, b) 
# Calculation: (1*4) + (3*-2) + (-5*-1) = 3`}
              pros={['Basis of Neural Networks', 'Efficient parallel computation', 'Geometric interpretation']}
              cons={['Computationally intensive for large matrices', 'Numerical instability']}
          >
              <DotProductViz />
          </AlgorithmCard>

          <AlgorithmCard
              id="calculus"
              title="Calculus"
              theory="Calculus, specifically derivatives, tells us how to change model parameters to minimize error. The gradient points in the direction of steepest ascent."
              math={<span>f'(x) = lim<sub>h&rarr;0</sub> <sup>f(x+h) - f(x)</sup>&frasl;<sub>h</sub></span>}
              mathLabel="Derivative Definition"
              code={`def f(x):
    return x**2

def derivative(x):
    return 2*x

x = 2
slope = derivative(x) # Slope at x=2 is 4`}
              pros={['Enables optimization (Gradient Descent)', 'Understanding of change', 'Backpropagation foundation']}
              cons={['Requires differentiable functions', 'Vanishing gradients in deep networks']}
          >
              <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={parabolaData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="x" type="number" stroke="#94a3b8" />
                          <YAxis stroke="#94a3b8" hide />
                          <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                          <Line type="monotone" dataKey="y" stroke="#818cf8" strokeWidth={3} dot={false} name="f(x) = x²" />
                          <Line type="linear" dataKey="tangent" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Tangent at x=2" />
                          <Scatter data={[{x: 2, y: 4}]} fill="#f472b6" />
                      </ComposedChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-center text-slate-500 mt-2">Tangent line (derivative) represents the instantaneous rate of change.</p>
              </div>
          </AlgorithmCard>

          <AlgorithmCard
              id="probability"
              title="Probability & Statistics"
              theory="Probability quantifies uncertainty. ML models predict probabilities (classification) or expected values (regression). The Normal (Gaussian) distribution is central due to the Central Limit Theorem."
              math={<span>P(A|B) = <sup>P(B|A)P(A)</sup>&frasl;<sub>P(B)</sub></span>}
              mathLabel="Bayes' Theorem"
              code={`import numpy as np

# Sampling from a Normal Distribution
mu, sigma = 0, 0.1 
s = np.random.normal(mu, sigma, 1000)

mean = np.mean(s)
variance = np.var(s)`}
              pros={['Handles uncertainty', 'Rigorous theoretical foundation', 'Basis for Generative AI']}
              cons={['Assumptions (like independence) often violated', 'Counter-intuitive results']}
          >
              <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={normalDistData}>
                          <defs>
                              <linearGradient id="colorNormal" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#818cf8" stopOpacity={0.8}/>
                                  <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
                              </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="x" stroke="#94a3b8" tickFormatter={(val) => val.toFixed(1)} />
                          <YAxis hide />
                          <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                          <Area type="monotone" dataKey="y" stroke="#818cf8" fillOpacity={1} fill="url(#colorNormal)" />
                      </AreaChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-center text-slate-500 mt-2">Standard Normal Distribution (Mean=0, Std=1)</p>
              </div>
          </AlgorithmCard>
        </div>
      </section>

      {/* --- SECTION 2: PYTHON CORE --- */}
      <section id="python-core" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">02</span>
            <h2 className="text-2xl font-bold text-emerald-400 uppercase tracking-widest">Python & Stack</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            
            {/* Core Data Structures - NEW BLOCK */}
            <div className="lg:col-span-2">
               <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg h-full">
                  <h3 className="text-lg font-bold text-white mb-4">Core Data Structures</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    <CodeBlock code={`# Lists (Sequences)
losses = [0.5, 0.4, 0.3]
losses.append(0.2)

# Dictionaries (Mappings)
config = {'lr': 0.01, 'optimizer': 'Adam'}
lr = config['lr']

# Sets (Unique Collection)
text = "ai ml ai"
vocab = set(text.split()) # {'ai', 'ml'}`} />
                    <DataStructuresViz />
                  </div>
               </div>
            </div>

            <div className="space-y-8">
               <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg h-full">
                  <h3 className="text-lg font-bold text-white mb-4">Python Essentials</h3>
                  <CodeBlock code={`# List Comprehension
squares = [x**2 for x in range(10)]

# Slicing
data = [10, 20, 30, 40, 50]
train = data[:3]  # [10, 20, 30]

# Lambda Functions
add = lambda x, y: x + y`} />
               </div>
            </div>
            
            <div className="space-y-8">
               <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg h-full">
                  <h3 className="text-lg font-bold text-white mb-4">Object Oriented Programming (Models)</h3>
                   <p className="text-slate-400 text-sm mb-4">Understanding Classes is vital for building models in PyTorch/Sklearn.</p>
                  <CodeBlock code={`class LinearModel:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.weights = None
        
    def fit(self, X, y):
        # Training logic uses self.lr
        pass
        
    def predict(self, X):
        return X * self.weights`} />
               </div>
            </div>

            <div className="lg:col-span-2">
               <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg h-full">
                  <h3 className="text-lg font-bold text-white mb-4">Data Manipulation with Pandas</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    <CodeBlock code={`import pandas as pd
import numpy as np

# Broadcasting (NumPy)
arr = np.array([1, 2]) * 2 # [2, 4]

# Filtering (Pandas)
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
subset = df[df['A'] > 1]`} />
                    <DataFrameViz />
                  </div>
               </div>
            </div>
        </div>
      </section>

      {/* --- SECTION 3: KEY ML LIBRARIES --- */}
      <section id="ml-libraries" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">03</span>
            <h2 className="text-2xl font-bold text-teal-400 uppercase tracking-widest">Key ML Libraries</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* NumPy */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg hover:border-teal-500/50 transition-colors">
                <div className="w-12 h-12 bg-blue-500/20 text-blue-400 rounded-lg flex items-center justify-center mb-4 border border-blue-500/30">
                    <span className="font-mono font-bold text-lg">Np</span>
                </div>
                <h3 className="text-lg font-bold text-white mb-2">NumPy</h3>
                <p className="text-sm text-slate-400 mb-4">
                    The fundamental package for scientific computing. It provides high-performance multidimensional array objects and tools for working with these arrays.
                </p>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <code className="text-xs text-blue-300">import numpy as np<br/>X = np.array([[1, 2], [3, 4]])</code>
                </div>
            </div>

            {/* Pandas */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg hover:border-teal-500/50 transition-colors">
                <div className="w-12 h-12 bg-indigo-500/20 text-indigo-400 rounded-lg flex items-center justify-center mb-4 border border-indigo-500/30">
                    <span className="font-mono font-bold text-lg">Pd</span>
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Pandas</h3>
                <p className="text-sm text-slate-400 mb-4">
                    Built on top of NumPy, it provides easy-to-use data structures (DataFrames) and data analysis tools for manipulating numerical tables and time series.
                </p>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <code className="text-xs text-indigo-300">import pandas as pd<br/>df = pd.read_csv('data.csv')</code>
                </div>
            </div>

            {/* Scikit-Learn */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg hover:border-teal-500/50 transition-colors">
                 <div className="w-12 h-12 bg-orange-500/20 text-orange-400 rounded-lg flex items-center justify-center mb-4 border border-orange-500/30">
                    <span className="font-mono font-bold text-lg">Sk</span>
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Scikit-Learn</h3>
                <p className="text-sm text-slate-400 mb-4">
                    Simple and efficient tools for predictive data analysis. It features various algorithms like SVM, random forests, and k-neighbors.
                </p>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <code className="text-xs text-orange-300">from sklearn.svm import SVC<br/>clf = SVC().fit(X, y)</code>
                </div>
            </div>
        </div>
      </section>

      {/* --- SECTION 4: DEFINITION OF LEARNING --- */}
      <section id="learning-definition" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">04</span>
            <h2 className="text-2xl font-bold text-fuchsia-400 uppercase tracking-widest">The Definition of Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <AlgorithmCard
          id="func-approximation"
          title="Function Approximation"
          theory="Machine learning is essentially function approximation. Given a set of inputs X and outputs y, we assume there is a relationship y = f(x) + ε, where ε is irreducible error (noise). Our goal is to learn a function f̂(x) that best approximates f(x) by minimizing a loss function."
          math={<span>y = f(x) + <span className="math-serif">&epsilon;</span> <br/> <span className="text-sm text-slate-500">Goal: minimize </span> Loss(y, f&#770;(x))</span>}
          mathLabel="The Fundamental Equation"
          code={`# Generating synthetic data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# True function is sin(x)
y = np.sin(X).ravel() 
# Add noise (+ epsilon)
y[::5] += 3 * (0.5 - np.random.rand(8))

# Fit a model (f_hat)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)`}
          pros={['Generalizes to unseen data', 'Automates rule discovery', 'Handles high-dimensional mappings']}
          cons={['Requires representative data', 'Cannot exceed the signal-to-noise ratio', 'Susceptible to bias']}
        >
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={learningDefData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="x" type="number" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                <Legend />
                <Scatter name="Noisy Observations (Data)" dataKey="y" fill="#94a3b8" opacity={0.6} />
                <Line name="True Function f(x)" type="monotone" dataKey="trueFx" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line name="Learned Function f̂(x)" type="monotone" dataKey="learnedFx" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="text-xs text-center text-slate-500 mt-2">The model tries to recover the Green line from the Grey dots.</p>
          </div>
        </AlgorithmCard>
      </section>

      {/* --- SECTION 5: TYPES OF LEARNING --- */}
      <section id="types-of-learning" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">05</span>
            <h2 className="text-2xl font-bold text-yellow-400 uppercase tracking-widest">Types of Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-lg">
             <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                 <div>
                    <h3 className="text-xl font-serif font-bold text-white mb-6">The Three Paradigms</h3>
                    <div className="space-y-6">
                        <div className="p-4 rounded-lg bg-indigo-900/20 border border-indigo-500/30">
                            <h4 className="font-bold text-indigo-400 mb-1">Supervised Learning</h4>
                            <p className="text-sm text-slate-400">Learning with a teacher. The model learns from labeled examples (Input X &rarr; Output Y).</p>
                        </div>
                        <div className="p-4 rounded-lg bg-emerald-900/20 border border-emerald-500/30">
                            <h4 className="font-bold text-emerald-400 mb-1">Unsupervised Learning</h4>
                            <p className="text-sm text-slate-400">Learning without a teacher. The model finds hidden structures or patterns in unlabeled data.</p>
                        </div>
                        <div className="p-4 rounded-lg bg-rose-900/20 border border-rose-500/30">
                            <h4 className="font-bold text-rose-400 mb-1">Reinforcement Learning</h4>
                            <p className="text-sm text-slate-400">Learning by trial and error. An agent learns to make decisions by receiving rewards/penalties.</p>
                        </div>
                    </div>
                 </div>
                 <div className="h-80 w-full relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={typesLearningData}>
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                            <Radar name="Supervised" dataKey="Supervised" stroke="#818cf8" fill="#818cf8" fillOpacity={0.3} />
                            <Radar name="Unsupervised" dataKey="Unsupervised" stroke="#34d399" fill="#34d399" fillOpacity={0.3} />
                            <Radar name="Reinforcement" dataKey="Reinforcement" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.3} />
                            <Legend />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                        </RadarChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-center text-slate-500 mt-2">Paradigm Comparison</p>
                 </div>
             </div>
        </div>
      </section>

      {/* --- SECTION 6: DATA PREPROCESSING --- */}
      <section id="data-preprocessing" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">06</span>
            <h2 className="text-2xl font-bold text-orange-400 uppercase tracking-widest">Data Processing</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Data Splitting */}
          <AlgorithmCard
             id="data-splitting"
             title="Data Splitting"
             theory="To evaluate a model's performance honestly, we must not test it on the same data it learned from. We split data into: Training (to learn), Validation (to tune hyperparameters), and Test (final evaluation)."
             math={<span>D = D<sub>train</sub> &cup; D<sub>val</sub> &cup; D<sub>test</sub></span>}
             mathLabel="Set Partitioning"
             code={`from sklearn.model_selection import train_test_split

# First split: Train (80%) and Temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)

# Second split: Validation (10%) and Test (10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)`}
             pros={['Prevents data leakage', 'Simulates real-world performance', 'Allows safe tuning']}
             cons={['Reduces data available for training', 'Sensitive to how the split is made (random seed)']}
          >
             <div className="h-40 w-full">
               <ResponsiveContainer width="100%" height="100%">
                 <BarChart layout="vertical" data={splitData} stackOffset="expand">
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" hide />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                    <Legend />
                    <Bar dataKey="Train" stackId="a" fill="#818cf8" />
                    <Bar dataKey="Validation" stackId="a" fill="#f472b6" />
                    <Bar dataKey="Test" stackId="a" fill="#34d399" />
                 </BarChart>
               </ResponsiveContainer>
               <p className="text-xs text-center text-slate-500 mt-2">Standard 70-15-15 Split</p>
             </div>
          </AlgorithmCard>

          {/* Feature Scaling */}
          <AlgorithmCard
             id="feature-scaling"
             title="Feature Scaling"
             theory="Machine learning algorithms (especially those based on distance like KNN or gradients like Linear Regression) perform poorly when features have vastly different scales. Normalization puts all features on a level playing field."
             math={<span>z = <sup>(x - <span className="math-serif">&mu;</span>)</sup>&frasl;<sub><span className="math-serif">&sigma;</span></sub></span>}
             mathLabel="Standardization (Z-Score)"
             code={`from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-Score Standardization (Mean=0, Std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max Normalization (0 to 1)
min_max = MinMaxScaler()
X_norm = min_max.fit_transform(X)`}
             pros={['Faster convergence for Gradient Descent', 'Required for distance-based algos (KNN, SVM)', 'Prevents dominance of large features']}
             cons={['Must store scaler parameters for inference', 'Sensitive to outliers (Min-Max)']}
          >
             <div className="h-64 w-full">
               <ResponsiveContainer width="100%" height="100%">
                 <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" dataKey="x" stroke="#94a3b8" label={{ value: 'Value X', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" stroke="#94a3b8" label={{ value: 'Value Y', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                    <Legend />
                    <Scatter name="Raw (Large Scale)" data={scalingData.filter(d => d.type.includes('Raw'))} fill="#ef4444" />
                    <Scatter name="Scaled (Centered)" data={scalingData.filter(d => d.type.includes('Scaled'))} fill="#34d399" />
                 </ScatterChart>
               </ResponsiveContainer>
               <p className="text-xs text-center text-slate-500 mt-2">Red: Large range [1000, 2000]. Green: Centered [-2, 2].</p>
             </div>
          </AlgorithmCard>
        </div>
      </section>

      {/* --- SECTION 7: OPTIMIZATION --- */}
      <section id="optimization" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">07</span>
            <h2 className="text-2xl font-bold text-sky-400 uppercase tracking-widest">Optimization</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <AlgorithmCard
            id="gradient-descent"
            title="Gradient Descent"
            theory="The engine of modern ML. It is an iterative algorithm used to minimize the loss function. By calculating the gradient (slope) of the loss with respect to parameters, we know which direction to step to reduce error."
            math={<span><span className="math-serif">&theta;</span> := <span className="math-serif">&theta;</span> - <span className="math-serif">&alpha;</span> <span className="math-serif">&nabla;</span>J(<span className="math-serif">&theta;</span>)</span>}
            mathLabel="Update Rule (alpha = learning rate)"
            code={`# Gradient Descent Loop
for i in range(epochs):
    # 1. Calculate predictions
    y_pred = np.dot(X, theta)
    
    # 2. Calculate Gradient
    gradient = (1/m) * X.T.dot(y_pred - y)
    
    # 3. Update Parameters
    theta = theta - learning_rate * gradient`}
            pros={['Simple and effective', 'Scales to large datasets (SGD)', 'Universal optimizer']}
            cons={['Can get stuck in local minima', 'Sensitive to Learning Rate (&alpha;)', 'Requires scaling']}
        >
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                <LineChart data={learningRateData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b' }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'Loss J(θ)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                    <Legend />
                    <Line name="Good Learning Rate" type="monotone" dataKey="goodLR" stroke="#34d399" strokeWidth={3} dot={false} />
                    <Line name="Bad Learning Rate (Oscillating)" type="monotone" dataKey="badLR" stroke="#ef4444" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                </LineChart>
                </ResponsiveContainer>
                <p className="text-xs text-center text-slate-500 mt-2">Effect of &alpha;: Smooth convergence vs Instability.</p>
            </div>
        </AlgorithmCard>
      </section>

      {/* --- SECTION 8: BIAS-VARIANCE TRADEOFF --- */}
      <section id="bias-variance" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">08</span>
            <h2 className="text-2xl font-bold text-rose-400 uppercase tracking-widest">Bias-Variance Tradeoff</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <AlgorithmCard
            id="bias-variance-card"
            title="The Tradeoff"
            theory="The central problem in Supervised Learning. We want a model that captures the signal (Low Bias) but ignores the noise (Low Variance). Simple models (high bias) underfit. Complex models (high variance) overfit. The sweet spot is in the middle."
            math={<span>E[Error] = Bias<sup>2</sup> + Variance + <span className="math-serif">&sigma;</span><sup>2</sup></span>}
            mathLabel="Error Decomposition"
            code={`# High Bias (Underfitting)
model = LinearRegression() # Too simple for curved data

# High Variance (Overfitting)
model = DecisionTree(max_depth=None) # Memorizes noise

# Balanced
model = RandomForest(max_depth=10)`}
            pros={['Fundamental framework for diagnostics', 'Explains Overfitting/Underfitting', 'Guides regularization']}
            cons={['Theoretical concept (cannot exactly calculate bias/variance in practice)']}
        >
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={biasVarianceData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="complexity" type="number" domain={[1, 20]} stroke="#94a3b8" label={{ value: 'Model Complexity', position: 'insideBottom', offset: -10, fill: '#64748b' }} />
                        <YAxis stroke="#94a3b8" label={{ value: 'Error', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                        <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} formatter={(value: number) => value.toFixed(2)} />
                        <Legend verticalAlign="top" height={36} />
                        
                        {/* Zones */}
                        <ReferenceArea x1={1} x2={6} stroke="none" fill="#ef4444" fillOpacity={0.1} label={{ value: "Underfitting (High Bias)", position: 'insideTop', fill: '#f87171', fontSize: 12, fontWeight: 'bold' }} />
                        <ReferenceArea x1={13} x2={20} stroke="none" fill="#818cf8" fillOpacity={0.1} label={{ value: "Overfitting (High Variance)", position: 'insideTop', fill: '#818cf8', fontSize: 12, fontWeight: 'bold' }} />
                        
                        {/* Optimal Line */}
                        <ReferenceLine x={8.5} stroke="#10b981" strokeDasharray="3 3" label={{ value: "Optimal Balance", position: 'top', fill: '#34d399', fontSize: 12, fontWeight: 'bold' }} />
                        
                        <Line name="Bias²" type="monotone" dataKey="biasSq" stroke="#f472b6" strokeWidth={2} dot={false} />
                        <Line name="Variance" type="monotone" dataKey="variance" stroke="#818cf8" strokeWidth={2} dot={false} />
                        <Line name="Total Error" type="monotone" dataKey="totalError" stroke="#ffffff" strokeWidth={4} dot={false} />
                    </ComposedChart>
                </ResponsiveContainer>
                <p className="text-xs text-center text-slate-500 mt-2">Low Complexity = High Bias (Underfitting). High Complexity = High Variance (Overfitting).</p>
            </div>
        </AlgorithmCard>
      </section>
    </div>
  );
};