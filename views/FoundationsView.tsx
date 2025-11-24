import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, ComposedChart, Scatter } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { CodeBlock } from '../components/CodeBlock';
import { MathBlock } from '../components/MathBlock';

// --- DATA FOR CHARTS ---

// 1. Optimization Data (Gradient Descent)
const gdData = Array.from({ length: 20 }, (_, i) => ({
  epoch: i,
  loss: 10 * Math.exp(-0.3 * i) + Math.random() * 0.5
}));

// 2. Normal Distribution Data
const normalDistData = Array.from({ length: 50 }, (_, i) => {
  const x = (i - 25) / 5; // Range -5 to 5
  const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);
  return { x, y };
});

// 3. Derivative / Tangent Visualization (y = x^2 at x=2)
const parabolaData = Array.from({ length: 41 }, (_, i) => {
  const x = (i - 20) / 5; // -4 to 4
  const y = x * x;
  // Tangent at x=2: y = 4x - 4
  // Slope (derivative) is 2x -> 4
  const tangent = 4 * x - 4;
  return { x, y, tangent: (x > 0 && x < 4) ? tangent : null }; 
});

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

export const FoundationsView: React.FC = () => {
  return (
    <div className="space-y-16 animate-fade-in pb-12">
      <header>
        <h1 className="text-4xl font-serif font-bold text-white mb-2">Machine Learning Foundations</h1>
        <p className="text-slate-400 text-lg">The bedrock of algorithms: Mathematics, Python Programming, and Optimization.</p>
      </header>

      {/* --- SECTION 1: MATH PRIMER --- */}
      <section id="math-primer" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-6">
            <div className="h-px bg-slate-800 flex-1"></div>
            <h2 className="text-2xl font-bold text-indigo-400 uppercase tracking-widest">Mathematical Prerequisites</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

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
# Calculation: (1*4) + (3*-2) + (-5*-1) = 4 - 6 + 5 = 3`}
            pros={['Basis of Neural Networks', 'Efficient parallel computation', 'Geometric interpretation of data']}
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
      </section>

      {/* --- SECTION 2: PYTHON CORE --- */}
      <section id="python-core" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-6">
            <div className="h-px bg-slate-800 flex-1"></div>
            <h2 className="text-2xl font-bold text-emerald-400 uppercase tracking-widest">Python Core</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-bold text-white mb-4">Essential Syntax</h3>
                <p className="text-slate-400 text-sm mb-4">Python's readability and concise syntax make it the lingua franca of Data Science. Mastering List Comprehensions and Slicing is crucial for data manipulation.</p>
                <CodeBlock code={`# List Comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary Comprehension
mapping = {x: x**2 for x in range(5)}

# Slicing
data = [10, 20, 30, 40, 50]
train = data[:3]  # [10, 20, 30]
test = data[3:]   # [40, 50]`} />
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-bold text-white mb-4">Advanced Concepts</h3>
                <p className="text-slate-400 text-sm mb-4">Classes form the structure of libraries like PyTorch. Decorators are used for logging or timing execution.</p>
                <CodeBlock code={`# Class for a Model
class SimpleModel:
    def __init__(self, w):
        self.w = w
    
    def predict(self, x):
        return x * self.w

# Lambda Functions
add = lambda x, y: x + y
# Used often in Pandas:
# df['col'].apply(lambda x: x*2)`} />
            </div>
        </div>
      </section>

      {/* --- SECTION 3: DS LIBRARIES --- */}
      <section id="data-stack" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-6">
            <div className="h-px bg-slate-800 flex-1"></div>
            <h2 className="text-2xl font-bold text-fuchsia-400 uppercase tracking-widest">Data Science Stack</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <AlgorithmCard
            id="numpy"
            title="NumPy (Numerical Python)"
            theory="The fundamental package for scientific computing. It provides a high-performance multidimensional array object. Unlike Python lists, NumPy arrays are stored in contiguous memory blocks, making operations up to 50x faster."
            math={<span>Broadcasting: A(3x1) + B(1x3) &rarr; C(3x3)</span>}
            mathLabel="Broadcasting Logic"
            code={`import numpy as np

# Creating Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Element-wise operations (No loops needed!)
arr = arr * 2 

# Reshaping
flat = arr.reshape(-1)`}
            pros={['Blazing fast vectorization', 'Broadcasting capabilities', 'Integration with C/C++']}
            cons={['Fixed type constraint', 'Steep learning curve for broadcasting rules']}
        >
            <div className="flex flex-col gap-4 items-center justify-center p-4">
                 <div className="flex items-center gap-4">
                     <div className="text-xs font-mono text-slate-500">Python List</div>
                     <div className="flex gap-2">
                         {[1, 2, 3].map(n => (
                             <div key={n} className="w-8 h-8 rounded-full border border-slate-600 flex items-center justify-center text-xs text-slate-400 relative">
                                 <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-0.5 h-2 bg-slate-700"></div>
                                 val
                             </div>
                         ))}
                     </div>
                     <span className="text-[10px] text-slate-600">(Pointers scattered)</span>
                 </div>
                 <div className="flex items-center gap-4">
                     <div className="text-xs font-mono text-slate-500">NumPy Array</div>
                     <div className="flex border-2 border-fuchsia-500 bg-fuchsia-900/20 rounded overflow-hidden">
                         {[1, 2, 3].map(n => (
                             <div key={n} className="w-8 h-8 flex items-center justify-center text-xs text-white border-r border-fuchsia-500/50 last:border-none">
                                 {n}
                             </div>
                         ))}
                     </div>
                     <span className="text-[10px] text-fuchsia-400">(Contiguous Memory)</span>
                 </div>
            </div>
        </AlgorithmCard>

        <AlgorithmCard
            id="pandas"
            title="Pandas"
            theory="Built on top of NumPy, Pandas introduces the DataFrame, a structured table-like object. It excels at data manipulation, cleaning, merging, and time-series analysis."
            math={<span>df.groupby('col').mean()</span>}
            mathLabel="Aggregation API"
            code={`import pandas as pd

# DataFrame Creation
data = {'Features': [1.2, 2.4], 'Target': [0, 1]}
df = pd.DataFrame(data)

# Selecting Data
subset = df.loc[df['Target'] == 1]

# Missing Data
df.fillna(0, inplace=True)`}
            pros={['Intuitive API for tabular data', 'Powerful Time-Series tools', 'Handles missing data gracefully']}
            cons={['High memory usage', 'Slow on very large datasets (use Polars/Spark instead)']}
        >
            <div className="p-4 flex justify-center">
                <DataFrameViz />
            </div>
        </AlgorithmCard>
      </section>

      {/* --- SECTION 4: OPTIMIZATION (Existing) --- */}
      <section id="gradient-descent" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-6">
            <div className="h-px bg-slate-800 flex-1"></div>
            <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Optimization</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <AlgorithmCard
            id="gradient-descent-algo"
            title="Gradient Descent"
            theory="Gradient Descent is an iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point."
            math={<span>&theta;<sub>new</sub> = &theta;<sub>old</sub> - &alpha; &nabla;J(&theta;)</span>}
            mathLabel="Update Rule"
            code={`def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1/m) * learning_rate * (X.T.dot(prediction - y))
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history`}
            pros={['Simple to implement', 'Computationally efficient for simple convex problems', 'Basis for advanced optimizers (Adam, RMSprop)']}
            cons={['Can get stuck in local minima', 'Sensitive to learning rate choice', 'Slow convergence near minimum']}
            hyperparameters={[
            {
                name: 'learning_rate (alpha)',
                description: 'Determines the step size at each iteration. Too small: slow convergence. Too large: overshooting.',
                default: '0.01',
                range: '(0, 1)'
            },
            {
                name: 'epochs',
                description: 'The number of times the algorithm will run through the entire training dataset.',
                default: '100',
                range: 'Integer'
            }
            ]}
        >
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gdData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b' }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                    <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }}
                    itemStyle={{ color: '#818cf8' }}
                    />
                    <Line type="monotone" dataKey="loss" stroke="#818cf8" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                </LineChart>
                </ResponsiveContainer>
                <p className="text-xs text-center text-slate-500 mt-2">Loss decreasing over epochs.</p>
            </div>
        </AlgorithmCard>
      </section>
    </div>
  );
};
