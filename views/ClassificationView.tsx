import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ScatterChart, Scatter, ReferenceDot, ComposedChart, Cell, LabelList, AreaChart, Area } from 'recharts';
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
  { id: 9, x: 45, y: 55, class: 'B', fill: '#3b82f6' }, // Outlierish
  { id: 10, x: 20, y: 25, class: 'A', fill: '#ef4444' }
];

const queryPoint = { x: 48, y: 48 };

// SVM Data
const svmData = [
  { x: 2, y: 3, class: 'A', isSupport: false }, 
  { x: 3, y: 2, class: 'A', isSupport: true }, // Support
  { x: 1, y: 1, class: 'A', isSupport: false },
  { x: 7, y: 8, class: 'B', isSupport: false }, 
  { x: 6, y: 9, class: 'B', isSupport: true }, // Support
  { x: 8, y: 7, class: 'B', isSupport: false },
];
// Hyperplane approx: y = x + 1. Margin approx width 2
const svmLine = [{x: 0, y: 1}, {x: 10, y: 11}]; 
const svmMarginA = [{x: 0, y: -1}, {x: 10, y: 9}]; // Below
const svmMarginB = [{x: 0, y: 3}, {x: 10, y: 13}]; // Above

const naiveBayesData = Array.from({ length: 50 }, (_, i) => {
  const x = (i / 50) * 10;
  const probA = (1 / (1.2 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 3) / 1.2, 2));
  const probB = (1 / (1.2 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 7) / 1.2, 2));
  return { x, probA, probB };
});

// --- SUB-COMPONENTS ---

const KNNViz = () => {
  const [k, setK] = useState(3);

  const { neighbors, radius } = useMemo(() => {
    const sorted = [...knnPoints].map(p => ({
        ...p,
        dist: Math.sqrt(Math.pow(p.x - queryPoint.x, 2) + Math.pow(p.y - queryPoint.y, 2))
    })).sort((a, b) => a.dist - b.dist);
    
    const nearest = sorted.slice(0, k);
    const maxDist = nearest[nearest.length - 1].dist;
    
    return { neighbors: nearest, radius: maxDist + 2 }; // +2 padding
  }, [k]);

  const classACount = neighbors.filter(n => n.class === 'A').length;
  const classBCount = neighbors.filter(n => n.class === 'B').length;
  const predictedClass = classACount > classBCount ? 'A (Red)' : 'B (Blue)';

  return (
    <div className="flex flex-col gap-4">
       <div className="flex justify-between items-center bg-slate-800 p-3 rounded-lg border border-slate-700">
          <div className="flex items-center gap-4">
             <label className="text-xs font-bold text-slate-400">Neighbors (k): <span className="text-indigo-400 text-sm">{k}</span></label>
             <input 
               type="range" min="1" max="9" step="2" 
               value={k} onChange={(e) => setK(Number(e.target.value))}
               className="w-32 h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-indigo-500"
             />
          </div>
          <div className="text-xs font-mono">
             Prediction: <span className={classACount > classBCount ? "text-red-400 font-bold" : "text-blue-400 font-bold"}>{predictedClass}</span>
          </div>
       </div>

       <div className="h-64 w-full bg-slate-900 rounded-lg border border-slate-800 p-2 relative">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
              <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              
              {/* Radius Circle */}
              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={radius * 2.5} fill="#fbbf24" fillOpacity={0.1} stroke="#fbbf24" strokeDasharray="3 3" />

              {/* Neighbors Connections */}
              {neighbors.map((n, i) => (
                 <ReferenceLine key={i} segment={[queryPoint, {x: n.x, y: n.y}]} stroke="#fbbf24" strokeOpacity={0.5} />
              ))}

              {/* Data Points */}
              <Scatter name="Data" data={knnPoints}>
                  {knnPoints.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} stroke={neighbors.find(n => n.id === entry.id) ? "#fbbf24" : "none"} strokeWidth={2} />
                  ))}
              </Scatter>

              {/* Query Point */}
              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={6} fill="#ffffff" stroke="#fbbf24" strokeWidth={2} />
            </ScatterChart>
          </ResponsiveContainer>
       </div>
       <p className="text-xs text-center text-slate-500">
          The algorithm classifies the White point based on the majority vote of the {k} nearest neighbors inside the yellow circle.
       </p>
    </div>
  );
};

const SVMViz = () => (
    <div className="h-64 w-full bg-slate-900 rounded-lg border border-slate-800 p-2">
        <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" domain={[0, 10]} hide />
            <YAxis type="number" dataKey="y" domain={[0, 10]} hide />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
            
            {/* Margins */}
            <Line data={svmMarginA} dataKey="y" stroke="#94a3b8" strokeDasharray="3 3" dot={false} strokeWidth={1} />
            <Line data={svmMarginB} dataKey="y" stroke="#94a3b8" strokeDasharray="3 3" dot={false} strokeWidth={1} />
            
            {/* Hyperplane */}
            <Line data={svmLine} dataKey="y" stroke="#ffffff" strokeWidth={2} dot={false} />

            {/* Support Vectors Highlight */}
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
        <p className="text-xs text-center text-slate-500 mt-2">
            <span className="text-yellow-400 font-bold">O</span> Support Vectors define the margin boundaries. The white line is the optimal hyperplane.
        </p>
    </div>
);

const DecisionTreeViz = () => (
  <div className="flex flex-col items-center w-full py-6 select-none bg-slate-900 rounded-lg border border-slate-800">
    {/* Root Node */}
    <div className="border-2 border-indigo-500 bg-slate-800 text-indigo-100 rounded px-4 py-2 text-sm font-mono font-bold z-10 shadow-[0_0_15px_rgba(99,102,241,0.3)]">
      Age &gt; 50?
    </div>
    
    {/* Root Link */}
    <div className="h-8 w-px bg-slate-600"></div>
    
    {/* Level 1 Crossbar */}
    <div className="w-48 h-px bg-slate-600 relative">
      <div className="absolute left-0 top-0 h-4 w-px bg-slate-600"></div>
      <div className="absolute right-0 top-0 h-4 w-px bg-slate-600"></div>
    </div>
    
    {/* Level 2 Container */}
    <div className="flex justify-between w-64 mt-4">
      
      {/* Left Branch (Decision Node) */}
      <div className="flex flex-col items-center -mt-4">
        <div className="border border-slate-600 bg-slate-800 text-slate-300 rounded px-3 py-1 text-xs font-mono mb-2 z-10 relative">
          Cholesterol &gt; 240?
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-slate-900 text-[9px] text-emerald-400 px-1">Yes</div>
        </div>
        
        {/* Link 2 */}
        <div className="h-4 w-px bg-slate-600"></div>
        
        {/* Level 2 Crossbar */}
        <div className="w-20 h-px bg-slate-600 relative">
             <div className="absolute left-0 top-0 h-4 w-px bg-slate-600"></div>
             <div className="absolute right-0 top-0 h-4 w-px bg-slate-600"></div>
        </div>

        {/* Level 3 Container (Leaves) */}
        <div className="flex justify-between w-28 mt-4">
             {/* Leaf High */}
             <div className="flex flex-col items-center -mt-4">
                 <div className="w-10 h-10 rounded-full bg-rose-500/20 border border-rose-500 flex items-center justify-center text-[10px] text-rose-300 font-bold shadow-[0_0_10px_rgba(244,63,94,0.2)]">
                    High
                 </div>
             </div>
             {/* Leaf Low */}
             <div className="flex flex-col items-center -mt-4">
                 <div className="w-10 h-10 rounded-full bg-emerald-500/20 border border-emerald-500 flex items-center justify-center text-[10px] text-emerald-300 font-bold shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                    Low
                 </div>
             </div>
        </div>
      </div>

      {/* Right Branch (Leaf Node) */}
      <div className="flex flex-col items-center -mt-4 relative">
         <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900 text-[9px] text-rose-400 px-1">No</div>
         <div className="border border-emerald-500 bg-emerald-900/30 text-emerald-200 rounded px-3 py-1 text-xs font-mono shadow-[0_0_10px_rgba(16,185,129,0.1)]">
            Low Risk
         </div>
         <div className="h-4 w-px bg-emerald-500/50"></div>
         <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
      </div>

    </div>
  </div>
);

export const ClassificationView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Supervised Learning: Classification</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Classification algorithms categorize data into discrete classes. They learn decision boundaries to separate different categories of data points based on feature inputs.
        </p>
      </header>

      <AlgorithmCard
        id="logistic-regression"
        title="Logistic Regression"
        theory="A probabilistic linear classifier that predicts the probability of an instance belonging to a default class using the sigmoid function. It maps any real-valued number into a value between 0 and 1."
        math={<span>&sigma;(z) = <sup>1</sup>&frasl;<sub>1 + e<sup>-z</sup></sub></span>}
        mathLabel="Sigmoid Activation"
        code={`from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)`}
        pros={['Probabilistic interpretation', 'Low variance', 'Efficient to train']}
        cons={['Assumes linear decision boundary', 'Not suitable for complex non-linear problems']}
        hyperparameters={[
          { name: 'C', description: 'Inverse of regularization strength. Smaller values (e.g., 0.01) increase regularization.', default: '1.0' },
          { name: 'penalty', description: 'Specifies the norm used in penalization (l1, l2).', default: 'l2' }
        ]}
      >
        <div className="h-64 w-full bg-slate-900 rounded-lg border border-slate-800 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sigmoidData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="x" stroke="#94a3b8" type="number" domain={[-10, 10]} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Decision Threshold (0.5)', fill: '#ef4444', fontSize: 10 }} />
              <Line type="monotone" dataKey="y" stroke="#818cf8" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Sigmoid: Maps input (-&infin;, +&infin;) to Probability (0, 1)</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="knn"
        title="K-Nearest Neighbors (KNN)"
        theory="A non-parametric, lazy learning algorithm that classifies a data point based on the majority class of its 'k' nearest neighbors in the feature space. It uses distance metrics like Euclidean distance."
        math={<span>d(p, q) = &radic;(&Sigma;<sub>i=1</sub><sup>n</sup> (p<sub>i</sub> - q<sub>i</sub>)<sup>2</sup>)</span>}
        mathLabel="Euclidean Distance"
        code={`from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)`}
        pros={['Simple and effective', 'No training phase (Lazy)', 'Naturally handles multi-class']}
        cons={['Computationally expensive at prediction time', 'Sensitive to scale of data', 'Struggles with high dimensionality']}
        hyperparameters={[
          { name: 'n_neighbors', description: 'Number of neighbors to use.', default: '5', range: 'Integer' },
          { name: 'weights', description: 'Weight function used in prediction (uniform/distance).', default: 'uniform' }
        ]}
      >
         <KNNViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="svm"
        title="Support Vector Machines (SVM)"
        theory="Finds the optimal hyperplane that maximizes the margin between different classes. For non-linear data, it uses the Kernel Trick to map data into higher-dimensional spaces where it becomes linearly separable."
        math={<span>min <sup>1</sup>&frasl;<sub>2</sub> ||w||<sup>2</sup> s.t. y<sub>i</sub>(w &sdot; x<sub>i</sub> + b) &ge; 1</span>}
        mathLabel="Optimization Objective (Hard Margin)"
        code={`from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
pred = svm.predict(X_test)`}
        pros={['Effective in high dimensions', 'Robust to outliers (with soft margin)', 'Versatile kernels']}
        cons={['Memory intensive', 'Hard to interpret probability', 'Sensitive to noise with large C']}
        hyperparameters={[
          { name: 'C', description: 'Regularization parameter. Lower C allows more misclassifications (softer margin).', default: '1.0' },
          { name: 'kernel', description: 'Specifies the kernel type (linear, poly, rbf).', default: 'rbf' }
        ]}
      >
         <SVMViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="naive-bayes"
        title="Naive Bayes"
        theory="A probabilistic classifier based on Bayes' Theorem with the 'naive' assumption of conditional independence between every pair of features given the value of the class variable."
        math={<span>P(y|X) &prop; P(y) &Pi;<sub>i=1</sub><sup>n</sup> P(x<sub>i</sub>|y)</span>}
        mathLabel="Bayes' Theorem Application"
        code={`from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)`}
        pros={['Extremely fast', 'Works well with small data', 'Good for text classification']}
        cons={['Independence assumption rarely holds', 'Zero probability problem for unseen features']}
      >
        <div className="h-64 w-full bg-slate-900 rounded-lg border border-slate-800 p-2">
          <ResponsiveContainer width="100%" height="100%">
             <AreaChart data={naiveBayesData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorProbA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorProbB" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="x" stroke="#94a3b8" />
                <YAxis hide />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                <Area type="monotone" dataKey="probA" stroke="#ef4444" fillOpacity={1} fill="url(#colorProbA)" name="Class A Dist" />
                <Area type="monotone" dataKey="probB" stroke="#3b82f6" fillOpacity={1} fill="url(#colorProbB)" name="Class B Dist" />
                <ReferenceLine x={5} stroke="#ffffff" strokeDasharray="3 3" label={{ value: 'Decision Boundary', fill: '#fff', fontSize: 10, position: 'top' }} />
             </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Class Probability Distributions (Gaussian)</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="decision-trees"
        title="Decision Trees"
        theory="Uses a tree-like model of decisions. It splits the data into subsets based on the value of input features, aiming to maximize information gain (or minimize entropy/Gini impurity) at each split."
        math={<span>H(S) = - &Sigma;<sub>i</sub> p<sub>i</sub> log<sub>2</sub>(p<sub>i</sub>)</span>}
        mathLabel="Entropy (Information Theory)"
        code={`from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)`}
        pros={['Easy to interpret/visualize', 'Requires little data preparation', 'Captures non-linear patterns']}
        cons={['Prone to overfitting (high variance)', 'Unstable (small data changes change tree)']}
        hyperparameters={[
           { name: 'max_depth', description: 'The maximum depth of the tree.', default: 'None', range: 'Integer' },
           { name: 'min_samples_split', description: 'Minimum number of samples required to split an internal node.', default: '2' }
        ]}
      >
        <DecisionTreeViz />
      </AlgorithmCard>
    </div>
  );
};