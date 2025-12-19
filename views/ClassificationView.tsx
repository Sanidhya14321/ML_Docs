import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ScatterChart, Scatter, ReferenceDot, ComposedChart, Cell, AreaChart, Area } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

// --- DATA ---

const sigmoidData = Array.from({ length: 41 }, (_, i) => {
  const x = (i - 20) / 2;
  const y = 1 / (1 + Math.exp(-x));
  return { x, y };
});

const knnPoints = [
  { id: 1, x: 35, y: 35, class: 'A', fill: '#ef4444' }, 
  { id: 2, x: 40, y: 30, class: 'A', fill: '#ef4444' }, 
  { id: 3, x: 30, y: 40, class: 'A', fill: '#ef4444' },  
  { id: 4, x: 25, y: 35, class: 'A', fill: '#ef4444' },
  { id: 5, x: 60, y: 60, class: 'B', fill: '#3b82f6' },
  { id: 6, x: 65, y: 55, class: 'B', fill: '#3b82f6' },
  { id: 7, x: 55, y: 65, class: 'B', fill: '#3b82f6' },
  { id: 8, x: 70, y: 60, class: 'B', fill: '#3b82f6' },
  { id: 9, x: 45, y: 55, class: 'B', fill: '#3b82f6' }, 
  { id: 10, x: 20, y: 25, class: 'A', fill: '#ef4444' }
];

const queryPoint = { x: 48, y: 48 };

const svmData = [
  { x: 2, y: 3, class: 'A', isSupport: false }, 
  { x: 3, y: 2, class: 'A', isSupport: true }, 
  { x: 1, y: 1, class: 'A', isSupport: false },
  { x: 7, y: 8, class: 'B', isSupport: false }, 
  { x: 6, y: 9, class: 'B', isSupport: true }, 
  { x: 8, y: 7, class: 'B', isSupport: false },
];
const svmLine = [{x: 0, y: 1}, {x: 10, y: 11}]; 
const svmMarginA = [{x: 0, y: -1}, {x: 10, y: 9}]; 
const svmMarginB = [{x: 0, y: 3}, {x: 10, y: 13}]; 

// --- INTERACTIVE COMPONENTS ---

const KNNViz = () => {
  const [k, setK] = useState(3);

  const { neighbors, radius } = useMemo(() => {
    const sorted = [...knnPoints].map(p => ({
        ...p,
        dist: Math.sqrt(Math.pow(p.x - queryPoint.x, 2) + Math.pow(p.y - queryPoint.y, 2))
    })).sort((a, b) => a.dist - b.dist);
    
    const nearest = sorted.slice(0, k);
    const maxDist = nearest[nearest.length - 1].dist;
    
    return { neighbors: nearest, radius: maxDist + 2 }; 
  }, [k]);

  const classACount = neighbors.filter(n => n.class === 'A').length;
  const classBCount = neighbors.filter(n => n.class === 'B').length;
  const predictedClass = classACount > classBCount ? 'A (Red)' : 'B (Blue)';

  return (
    <div className="flex flex-col gap-4">
       <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
          <div className="flex items-center gap-6">
             <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Neighbors (k): <span className="text-indigo-400 text-sm ml-2">{k}</span></label>
             <input 
               type="range" min="1" max="9" step="2" 
               value={k} onChange={(e) => setK(Number(e.target.value))}
               className="w-32 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
             />
          </div>
          <div className="text-[10px] font-mono uppercase tracking-widest bg-slate-900 px-3 py-1 rounded-full border border-slate-700">
             Result: <span className={classACount > classBCount ? "text-red-400 font-bold" : "text-blue-400 font-bold"}>{predictedClass}</span>
          </div>
       </div>

       <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative shadow-inner overflow-hidden">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
              <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
              
              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={radius * 2.5} fill="#fbbf24" fillOpacity={0.03} stroke="#fbbf24" strokeDasharray="3 3" strokeOpacity={0.4} />

              {neighbors.map((n, i) => (
                 <ReferenceLine key={i} segment={[queryPoint, {x: n.x, y: n.y}]} stroke="#fbbf24" strokeOpacity={0.2} strokeWidth={1} />
              ))}

              <Scatter name="Data" data={knnPoints}>
                  {knnPoints.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} stroke={neighbors.find(n => n.id === entry.id) ? "#fbbf24" : "none"} strokeWidth={2} />
                  ))}
              </Scatter>

              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={6} fill="#ffffff" stroke="#fbbf24" strokeWidth={2} />
            </ScatterChart>
          </ResponsiveContainer>
       </div>
    </div>
  );
};

const NaiveBayesViz = () => {
    const [overlap, setOverlap] = useState(4);
    
    const data = useMemo(() => {
        return Array.from({ length: 60 }, (_, i) => {
            const x = (i / 60) * 12;
            const probA = (1 / (1.2 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 3) / 1.2, 2));
            const probB = (1 / (1.2 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - (3 + overlap)) / 1.2, 2));
            return { x, probA, probB };
        });
    }, [overlap]);

    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Class Separation: <span className="text-indigo-400 ml-2">{overlap.toFixed(1)}</span></label>
                <input 
                    type="range" min="0" max="8" step="0.5" 
                    value={overlap} onChange={(e) => setOverlap(parseFloat(e.target.value))}
                    className="w-40 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
            </div>
            <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorNB_A" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorNB_B" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="x" hide />
                        <YAxis hide />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '10px' }} />
                        <Area type="monotone" dataKey="probA" stroke="#ef4444" fillOpacity={1} fill="url(#colorNB_A)" name="P(x|C1)" />
                        <Area type="monotone" dataKey="probB" stroke="#3b82f6" fillOpacity={1} fill="url(#colorNB_B)" name="P(x|C2)" />
                        <ReferenceLine x={3 + overlap/2} stroke="#ffffff" strokeDasharray="3 3" strokeOpacity={0.5} label={{ value: 'Boundary', fill: '#fff', fontSize: 9, position: 'top' }} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const DecisionTreeViz = () => {
  // SVG Tree layout
  const width = 400;
  const height = 240;
  const nodes = [
    { x: 200, y: 40, label: 'Entropy: 0.98\nFeature X > 5.2', color: 'indigo' },
    { x: 100, y: 120, label: 'Feature Y ≤ 2.1', color: 'indigo' },
    { x: 300, y: 120, label: 'Pure Leaf', color: 'emerald' },
    { x: 50, y: 200, label: 'Class A', color: 'rose' },
    { x: 150, y: 200, label: 'Class B', color: 'emerald' },
  ];

  const links = [
    { from: 0, to: 1 },
    { from: 0, to: 2 },
    { from: 1, to: 3 },
    { from: 1, to: 4 },
  ];

  return (
    <div className="flex flex-col items-center w-full py-8 bg-slate-950 rounded-2xl border border-slate-800/50 shadow-inner overflow-hidden select-none">
       <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="max-w-md w-full">
          {/* Links */}
          {links.map((link, i) => {
             const from = nodes[link.from];
             const to = nodes[link.to];
             return (
               <path 
                 key={i} 
                 d={`M ${from.x} ${from.y} C ${from.x} ${(from.y + to.y) / 2}, ${to.x} ${(from.y + to.y) / 2}, ${to.x} ${to.y}`}
                 stroke="#334155"
                 strokeWidth="2"
                 fill="none"
                 strokeDasharray="4 2"
               />
             );
          })}

          {/* Nodes */}
          {nodes.map((node, i) => {
             const isLeaf = i >= 2;
             const colorMap = {
               indigo: { bg: '#1e1b4b', border: '#6366f1', text: '#818cf8' },
               emerald: { bg: '#064e3b', border: '#10b981', text: '#34d399' },
               rose: { bg: '#4c0519', border: '#f43f5e', text: '#fb7185' }
             };
             const theme = colorMap[node.color as keyof typeof colorMap];

             return (
               <g key={i} transform={`translate(${node.x}, ${node.y})`}>
                  <rect 
                    x={isLeaf ? -30 : -50} 
                    y={isLeaf ? -15 : -25} 
                    width={isLeaf ? 60 : 100} 
                    height={isLeaf ? 30 : 50} 
                    rx="8" 
                    fill={theme.bg} 
                    stroke={theme.border} 
                    strokeWidth="2"
                    className="transition-all hover:scale-110 cursor-help"
                  />
                  {node.label.split('\n').map((line, lineIdx) => (
                    <text 
                      key={lineIdx}
                      y={isLeaf ? 5 : (lineIdx === 0 ? -2 : 12)} 
                      textAnchor="middle" 
                      fill={theme.text} 
                      fontSize={isLeaf ? "10" : "8"} 
                      fontWeight="bold"
                      className="pointer-events-none uppercase font-mono tracking-tighter"
                    >
                      {line}
                    </text>
                  ))}
               </g>
             );
          })}
       </svg>
       <p className="text-[9px] text-slate-600 mt-6 uppercase tracking-[0.3em] font-mono">Recursive Binary Splitting</p>
    </div>
  );
};

// --- MAIN VIEW ---

export const ClassificationView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header className="mb-12 border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Supervised: Classification</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
          The task of predicting a discrete class label. Models learn to find the optimal decision boundary that generalizes well to unseen samples in complex feature spaces.
        </p>
      </header>

      <AlgorithmCard
        id="logistic-regression"
        title="Logistic Regression"
        complexity="Fundamental"
        theory="The cornerstone of classification. It estimates probabilities using a logistic (sigmoid) function. While it is a linear model, it outputs a probabilistic score between 0 and 1, usually thresholded at 0.5 for binary classification."
        math={<span>P(y=1|x) = &sigma;(w&sdot;x + b) = <sup>1</sup>&frasl;<sub>1 + e<sup>-(w&sdot;x + b)</sup></sub></span>}
        mathLabel="Logistic Activation"
        code={`from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)`}
        pros={['Excellent baseline model', 'Interpretability via coefficients', 'Computationally very efficient']}
        cons={['Assumes linear decision boundary', 'Struggles with complex interactions without manual feature engineering']}
        hyperparameters={[
          { name: 'C', description: 'Inverse of regularization strength; smaller values specify stronger regularization.', default: '1.0' },
          { name: 'penalty', description: 'Used to specify the norm used in the penalization (l1, l2).', default: 'l2' }
        ]}
      >
        <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sigmoidData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="x" stroke="#475569" type="number" domain={[-10, 10]} hide />
              <YAxis stroke="#475569" domain={[0, 1]} fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Threshold', fill: '#ef4444', fontSize: 9 }} />
              <Line type="monotone" dataKey="y" stroke="#818cf8" strokeWidth={4} dot={false} animationDuration={1000} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="knn"
        title="K-Nearest Neighbors (KNN)"
        complexity="Fundamental"
        theory="A non-parametric, lazy learner. It doesn't find a global function; instead, it stores the training data and classifies new points by taking a majority vote of their nearest 'k' neighbors."
        math={<span>d(p, q) = &radic;&Sigma; (p<sub>i</sub> - q<sub>i</sub>)<sup>2</sup></span>}
        mathLabel="Euclidean Distance"
        code={`from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)`}
        pros={['No training phase', 'Adapts quickly to new data', 'Intuitive decision logic']}
        cons={['Slow at inference time (must scan all data)', 'Curse of Dimensionality', 'Sensitive to outliers']}
        hyperparameters={[
          { name: 'n_neighbors', description: 'Number of neighbors to consider.', default: '5' },
          { name: 'metric', description: 'Distance function used for the tree.', default: 'minkowski' }
        ]}
      >
         <KNNViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="svm"
        title="Support Vector Machines (SVM)"
        complexity="Intermediate"
        theory="SVM finds the hyperplane that separates classes with the maximum margin. Using kernels, it can solve non-linear problems by projecting data into higher dimensions where a linear separation exists."
        math={<span>max <sup>2</sup>&frasl;<sub>||w||</sub> s.t. y<sub>i</sub>(w&sdot;x<sub>i</sub> - b) &ge; 1</span>}
        mathLabel="Margin Optimization"
        code={`from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)`}
        pros={['Effective in high dimensions', 'Memory efficient (uses support vectors)', 'Versatile via kernel trick']}
        cons={['Doesn\'t directly provide probability estimates', 'Slow on large datasets', 'Hard to interpret']}
      >
        <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2">
            <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="x" domain={[0, 10]} hide />
                <YAxis type="number" dataKey="y" domain={[0, 10]} hide />
                <Line data={svmMarginA} dataKey="y" stroke="#475569" strokeDasharray="3 3" dot={false} strokeWidth={1} />
                <Line data={svmMarginB} dataKey="y" stroke="#475569" strokeDasharray="3 3" dot={false} strokeWidth={1} />
                <Line data={svmLine} dataKey="y" stroke="#ffffff" strokeWidth={2} dot={false} />
                {svmData.filter(d => d.isSupport).map((d, i) => (
                    <ReferenceDot key={i} x={d.x} y={d.y} r={10} fill="none" stroke="#fbbf24" strokeWidth={2} strokeOpacity={0.6} />
                ))}
                <Scatter data={svmData}>
                    {svmData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.class === 'A' ? '#ef4444' : '#3b82f6'} />
                    ))}
                </Scatter>
            </ComposedChart>
            </ResponsiveContainer>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="naive-bayes"
        title="Naive Bayes"
        complexity="Fundamental"
        theory="A probabilistic classifier based on Bayes' Theorem. It makes the 'naive' assumption of conditional independence between every pair of features, allowing for extremely fast computation and high scalability."
        math={<span>P(C|x) = <sup>P(x|C)P(C)</sup>&frasl;<sub>P(x)</sub></span>}
        mathLabel="Bayesian Inference"
        code={`from sklearn.naive_bayes import GaussianNB
nb = GaussianNB().fit(X_train, y_train)`}
        pros={['Extremely fast and scalable', 'Works well with small datasets', 'Excellent for text (Spam detection)']}
        cons={['Strong independence assumption rarely holds', 'Zero-frequency problem for unseen categories']}
      >
        <NaiveBayesViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="decision-trees"
        title="Decision Trees"
        complexity="Intermediate"
        theory="Models decisions through a branching tree structure. It greedily splits the data at each node to maximize information gain or minimize impurity metrics like Gini or Entropy."
        math={<span>Gini = 1 - &Sigma; (p<sub>i</sub>)<sup>2</sup></span>}
        mathLabel="Impurity Metric"
        code={`from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5)`}
        pros={['White-box interpretability', 'Requires zero data scaling', 'Implicit feature selection']}
        cons={['Highly prone to overfitting', 'Unstable: small data changes yield different trees', 'Sensitive to imbalanced data']}
      >
        <DecisionTreeViz />
      </AlgorithmCard>
    </div>
  );
};