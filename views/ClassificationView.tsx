import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ScatterChart, Scatter, ReferenceDot, ComposedChart, Cell, AreaChart, Area, Legend } from 'recharts';
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
       <div className="flex justify-between items-center bg-slate-800 p-3 rounded-lg border border-slate-700">
          <div className="flex items-center gap-4">
             <label className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Neighbors (k): <span className="text-indigo-400 text-sm ml-1">{k}</span></label>
             <input 
               type="range" min="1" max="9" step="2" 
               value={k} onChange={(e) => setK(Number(e.target.value))}
               className="w-32 h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-indigo-500"
             />
          </div>
          <div className="text-xs font-mono">
             Result: <span className={classACount > classBCount ? "text-red-400 font-bold" : "text-blue-400 font-bold"}>{predictedClass}</span>
          </div>
       </div>

       <div className="h-64 w-full bg-slate-950 rounded-lg border border-slate-800 p-2 relative shadow-inner">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
              <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
              
              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={radius * 2.5} fill="#fbbf24" fillOpacity={0.05} stroke="#fbbf24" strokeDasharray="3 3" />

              {neighbors.map((n, i) => (
                 <ReferenceLine key={i} segment={[queryPoint, {x: n.x, y: n.y}]} stroke="#fbbf24" strokeOpacity={0.3} />
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
            <div className="flex justify-between items-center bg-slate-800 p-3 rounded-lg border border-slate-700">
                <label className="text-xs font-bold text-slate-400 uppercase">Class Separation: <span className="text-indigo-400">{overlap.toFixed(1)}</span></label>
                <input 
                    type="range" min="0" max="8" step="0.5" 
                    value={overlap} onChange={(e) => setOverlap(parseFloat(e.target.value))}
                    className="w-40 h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
            </div>
            <div className="h-64 w-full bg-slate-950 rounded-lg border border-slate-800 p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorNB_A" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorNB_B" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="x" hide />
                        <YAxis hide />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                        <Area type="monotone" dataKey="probA" stroke="#ef4444" fillOpacity={1} fill="url(#colorNB_A)" name="P(x|C1)" />
                        <Area type="monotone" dataKey="probB" stroke="#3b82f6" fillOpacity={1} fill="url(#colorNB_B)" name="P(x|C2)" />
                        <ReferenceLine x={3 + overlap/2} stroke="#ffffff" strokeDasharray="3 3" label={{ value: 'Decision Boundary', fill: '#fff', fontSize: 10, position: 'top' }} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-slate-500 text-center uppercase tracking-widest italic">Modeling conditional probabilities with Gaussian Likelihoods</p>
        </div>
    );
};

const DecisionTreeViz = () => (
  <div className="flex flex-col items-center w-full py-8 select-none bg-slate-950 rounded-lg border border-slate-800 shadow-inner">
    <div className="border-2 border-indigo-500 bg-slate-900 text-indigo-100 rounded-lg px-6 py-3 text-sm font-mono font-bold z-10 shadow-[0_0_20px_rgba(99,102,241,0.3)]">
      Entropy: 0.98 | Split: Feature X &gt; 5.2
    </div>
    <div className="h-10 w-px bg-slate-700"></div>
    <div className="w-64 h-px bg-slate-700 relative">
      <div className="absolute left-0 top-0 h-6 w-px bg-slate-700"></div>
      <div className="absolute right-0 top-0 h-6 w-px bg-slate-700"></div>
    </div>
    <div className="flex justify-between w-80 mt-6">
      <div className="flex flex-col items-center group">
        <div className="border border-indigo-400/50 bg-slate-900 text-indigo-200 rounded px-4 py-2 text-xs font-mono shadow-sm group-hover:border-indigo-400 transition-colors">
          Feature Y &le; 2.1
        </div>
        <div className="h-6 w-px bg-slate-700"></div>
        <div className="w-24 h-px bg-slate-700 relative">
             <div className="absolute left-0 top-0 h-4 w-px bg-slate-700"></div>
             <div className="absolute right-0 top-0 h-4 w-px bg-slate-700"></div>
        </div>
        <div className="flex justify-between w-32 mt-4">
             <div className="w-10 h-10 rounded bg-rose-500/20 border border-rose-500 flex items-center justify-center text-[10px] text-rose-300 font-bold shadow-[0_0_15px_rgba(244,63,94,0.2)]">Class A</div>
             <div className="w-10 h-10 rounded bg-emerald-500/20 border border-emerald-500 flex items-center justify-center text-[10px] text-emerald-300 font-bold shadow-[0_0_15px_rgba(16,185,129,0.2)]">Class B</div>
        </div>
      </div>
      <div className="flex flex-col items-center">
        <div className="w-20 h-20 rounded-full border-4 border-indigo-500/20 bg-emerald-500/10 flex flex-col items-center justify-center text-emerald-400 shadow-lg border-dashed">
            <span className="text-xl font-bold">B</span>
            <span className="text-[8px] uppercase font-mono">Pure Leaf</span>
        </div>
        <span className="text-[10px] text-slate-500 mt-2 font-mono">Confidence: 100%</span>
      </div>
    </div>
  </div>
);

// --- MAIN VIEW ---

export const ClassificationView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Supervised: Classification</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed">
          The art of assigning objects to discrete categories. Classification models learn complex decision boundaries in feature space to predict class membership for new data.
        </p>
      </header>

      <AlgorithmCard
        id="logistic-regression"
        title="Logistic Regression"
        theory="Despite its name, it's a fundamental classification algorithm. It models the probability that an input belongs to a specific class using the sigmoid function, which maps inputs to a range between 0 and 1."
        math={<span>P(y=1|x) = &sigma;(w&sdot;x + b) = <sup>1</sup>&frasl;<sub>1 + e<sup>-(w&sdot;x + b)</sup></sub></span>}
        mathLabel="Logistic (Sigmoid) Activation"
        code={`from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
probabilities = clf.predict_proba(X_test)`}
        pros={['Fast to train', 'Output has a probabilistic interpretation', 'Provides feature weights']}
        cons={['Assumes linear relationship', 'Vulnerable to multicollinearity']}
        hyperparameters={[
          { name: 'C', description: 'Inverse of regularization strength.', default: '1.0' },
          { name: 'penalty', description: 'Regularization type (l1, l2).', default: 'l2' }
        ]}
      >
        <div className="h-64 w-full bg-slate-950 rounded-lg border border-slate-800 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sigmoidData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="x" stroke="#475569" type="number" domain={[-10, 10]} hide />
              <YAxis stroke="#475569" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Decision Threshold', fill: '#ef4444', fontSize: 10 }} />
              <Line type="monotone" dataKey="y" stroke="#818cf8" strokeWidth={4} dot={false} animationDuration={1000} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="knn"
        title="K-Nearest Neighbors (KNN)"
        theory="A simple but powerful lazy learner. It doesn't build a model; instead, it looks at the 'k' closest data points to a new observation and assigns it the most frequent class among them."
        math={<span>d(p, q) = &radic;&Sigma; (p<sub>i</sub> - q<sub>i</sub>)<sup>2</sup></span>}
        mathLabel="Euclidean Distance Metric"
        code={`from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)`}
        pros={['No training phase required', 'Naturally multi-class', 'Effective for complex datasets']}
        cons={['Slow at prediction time', 'Highly sensitive to feature scaling', 'Memory intensive']}
        hyperparameters={[
          { name: 'n_neighbors', description: 'Number of neighbors (k).', default: '5' },
          { name: 'metric', description: 'Distance calculation method.', default: 'minkowski' }
        ]}
      >
         <KNNViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="svm"
        title="Support Vector Machines (SVM)"
        theory="SVM finds the optimal hyperplane that maximizes the margin between classes. For data that isn't linearly separable, it uses the Kernel Trick to map data into higher dimensions."
        math={<span>max <sup>2</sup>&frasl;<sub>||w||</sub> s.t. y<sub>i</sub>(w&sdot;x<sub>i</sub> - b) &ge; 1</span>}
        mathLabel="Margin Maximization Objective"
        code={`from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)`}
        pros={['High accuracy in high dimensions', 'Robust to outliers', 'Versatile kernels']}
        cons={['High training time on large data', 'Hyperparameters are hard to tune']}
      >
        <div className="h-64 w-full bg-slate-950 rounded-lg border border-slate-800 p-2">
            <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="x" domain={[0, 10]} hide />
                <YAxis type="number" dataKey="y" domain={[0, 10]} hide />
                <Line data={svmMarginA} dataKey="y" stroke="#475569" strokeDasharray="3 3" dot={false} strokeWidth={1} />
                <Line data={svmMarginB} dataKey="y" stroke="#475569" strokeDasharray="3 3" dot={false} strokeWidth={1} />
                <Line data={svmLine} dataKey="y" stroke="#ffffff" strokeWidth={2} dot={false} />
                {svmData.filter(d => d.isSupport).map((d, i) => (
                    <ReferenceDot key={i} x={d.x} y={d.y} r={10} fill="none" stroke="#fbbf24" strokeWidth={2} />
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
        theory="Based on Bayes' Theorem, it calculates the probability of each class given the input features, assuming features are independent of each other (the 'naive' assumption)."
        math={<span>P(C|x) = <sup>P(x|C)P(C)</sup>&frasl;<sub>P(x)</sub></span>}
        mathLabel="Bayes' Rule"
        code={`from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)`}
        pros={['Extremely fast', 'Scales well with features', 'Performs well on text']}
        cons={['Independence assumption is often false', 'Zero frequency problem']}
      >
        <NaiveBayesViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="decision-trees"
        title="Decision Trees"
        theory="A flowchart-like structure where each internal node represents a 'test' on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label."
        math={<span>Gini = 1 - &Sigma; (p<sub>i</sub>)<sup>2</sup></span>}
        mathLabel="Gini Impurity Metric"
        code={`from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)`}
        pros={['Human-interpretable', 'Handles non-linear data', 'No scaling required']}
        cons={['Highly prone to overfitting', 'Unstable (small data changes change tree)']}
      >
        <DecisionTreeViz />
      </AlgorithmCard>
    </div>
  );
};