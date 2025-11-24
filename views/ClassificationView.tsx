import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ScatterChart, Scatter, ReferenceDot, ComposedChart, Cell, LabelList, AreaChart, Area } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

// Data for Sigmoid
const sigmoidData = Array.from({ length: 41 }, (_, i) => {
  const x = (i - 20) / 2;
  const y = 1 / (1 + Math.exp(-x));
  return { x, y };
});

// Data for KNN
// Refined dataset to clearly show neighbors
const knnData = [
  { id: 1, x: 12, y: 15, class: 'A', fill: '#ef4444' }, // Neighbor 1
  { id: 2, x: 10, y: 10, class: 'A', fill: '#ef4444' }, // Neighbor 2
  { id: 3, x: 8, y: 12, class: 'A', fill: '#ef4444' },  // Neighbor 3 (if k=3, depends on query)
  { id: 4, x: 25, y: 25, class: 'B', fill: '#3b82f6' },
  { id: 5, x: 22, y: 28, class: 'B', fill: '#3b82f6' },
  { id: 6, x: 28, y: 22, class: 'B', fill: '#3b82f6' },
];

const queryPoint = { x: 14, y: 14 };

// Calculate distances to identify neighbors (Simulation for K=3)
// Distances from (14, 14):
// 1. (12, 15): sqrt(4 + 1) = 2.23
// 2. (10, 10): sqrt(16 + 16) = 5.65
// 3. (8, 12): sqrt(36 + 4) = 6.32
// 4. (25, 25): sqrt(121 + 121) = 15.5
// So neighbors are indices 0, 1, 2.

const neighbors = [knnData[0], knnData[1], knnData[2]];

// Data for SVM (Linearly separable)
const svmData = [
  { x: 1, y: 2, class: 'A' }, { x: 2, y: 1, class: 'A' }, { x: 3, y: 3, class: 'A' },
  { x: 7, y: 8, class: 'B' }, { x: 8, y: 7, class: 'B' }, { x: 9, y: 9, class: 'B' },
];
// Hyperplane: y = -x + 10 (approx)
const svmLine = [{ x: 0, y: 10 }, { x: 10, y: 0 }];
const svmMargin1 = [{ x: 0, y: 8 }, { x: 8, y: 0 }];
const svmMargin2 = [{ x: 2, y: 10 }, { x: 10, y: 2 }];

// Data for Naive Bayes (Gaussian Distributions)
const naiveBayesData = Array.from({ length: 50 }, (_, i) => {
  const x = (i / 50) * 10;
  // Class A: Mean 3, Std 1.5
  const probA = (1 / (1.5 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 3) / 1.5, 2));
  // Class B: Mean 7, Std 1.5
  const probB = (1 / (1.5 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 7) / 1.5, 2));
  return { x, probA, probB };
});

// Custom CSS Tree Component
const DecisionTreeViz = () => (
  <div className="flex flex-col items-center w-full py-6 select-none">
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
        <div className="border border-slate-600 bg-slate-800 text-slate-300 rounded px-3 py-1 text-xs font-mono mb-2 z-10">
          Cholesterol &gt; 240?
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
                 <span className="text-[9px] text-slate-500 mt-1 uppercase tracking-wider">Risk</span>
             </div>
             {/* Leaf Low */}
             <div className="flex flex-col items-center -mt-4">
                 <div className="w-10 h-10 rounded-full bg-emerald-500/20 border border-emerald-500 flex items-center justify-center text-[10px] text-emerald-300 font-bold shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                    Low
                 </div>
                 <span className="text-[9px] text-slate-500 mt-1 uppercase tracking-wider">Safe</span>
             </div>
        </div>
      </div>

      {/* Right Branch (Leaf Node) */}
      <div className="flex flex-col items-center -mt-4">
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
          {
            name: 'C',
            description: 'Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.',
            default: '1.0',
            range: '[0, infinity)'
          },
          {
            name: 'penalty',
            description: 'Used to specify the norm used in the penalization.',
            default: 'l2',
            range: 'l1, l2, elasticnet, none'
          },
          {
            name: 'solver',
            description: 'Algorithm to use in the optimization problem.',
            default: 'lbfgs',
            range: 'liblinear, newton-cg, lbfgs, sag, saga'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sigmoidData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="x" stroke="#94a3b8" type="number" domain={[-10, 10]} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="y" stroke="#818cf8" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Sigmoid Function: Mapping inputs to Probabilities</p>
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
          {
            name: 'n_neighbors',
            description: 'Number of neighbors to use for k_neighbors queries.',
            default: '5',
            range: 'Integer'
          },
          {
            name: 'weights',
            description: 'Weight function used in prediction. Uniform means all points carry equal weight; distance means closer points have more influence.',
            default: 'uniform',
            range: 'uniform, distance'
          },
          {
            name: 'metric',
            description: 'The distance metric to use for the tree.',
            default: 'minkowski',
            range: 'euclidean, manhattan, chebyshev, minkowski'
          }
        ]}
      >
         <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" domain={[0, 40]} />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" domain={[0, 40]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              
              {/* Highlight Neighbors with Halo */}
              <Scatter data={neighbors} shape="circle" fill="transparent" stroke="#fbbf24" strokeWidth={2}>
                 <LabelList dataKey="x" content={() => null} />
                 {neighbors.map((entry, index) => (
                    // This creates a glowing ring around neighbors
                    <ReferenceDot key={`halo-${index}`} x={entry.x} y={entry.y} r={8} stroke="#fbbf24" fill="none" strokeDasharray="2 2" />
                 ))}
              </Scatter>

              {/* Data Points */}
              <Scatter name="Data" data={knnData} fill="#8884d8">
                  {knnData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
              </Scatter>

              {/* Connection Lines to Neighbors */}
              {neighbors.map((neighbor, i) => (
                <ReferenceLine 
                    key={`link-${i}`} 
                    segment={[{ x: queryPoint.x, y: queryPoint.y }, { x: neighbor.x, y: neighbor.y }]} 
                    stroke="#fbbf24" 
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    opacity={0.8}
                />
              ))}

              {/* Query Point */}
              <ReferenceDot x={queryPoint.x} y={queryPoint.y} r={6} fill="#ffffff" stroke="#fbbf24" strokeWidth={2} />
              
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Classifying white point based on 3 nearest neighbors (connected)</p>
        </div>
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
          {
            name: 'C',
            description: 'Regularization parameter. Controls trade-off between smooth decision boundary and classifying training points correctly. Lower C allows more misclassifications (softer margin).',
            default: '1.0',
            range: '[0, infinity)'
          },
          {
            name: 'kernel',
            description: 'Specifies the kernel type used to map data into higher dimensions. The "rbf" kernel is effective for non-linear data.',
            default: 'rbf',
            range: 'linear, poly, rbf, sigmoid'
          },
          {
            name: 'gamma',
            description: 'Kernel coefficient. Defines how far the influence of a single training example reaches. High gamma leads to complex, tight boundaries (overfitting risk).',
            default: 'scale',
            range: 'scale, auto, float'
          }
        ]}
      >
         <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" domain={[0, 10]} />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" domain={[0, 10]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              
              <Scatter name="Class A" data={svmData.slice(0, 3)} fill="#ef4444" />
              <Scatter name="Class B" data={svmData.slice(3, 6)} fill="#3b82f6" />
              
              {/* Hyperplane */}
              <Line data={svmLine} dataKey="y" stroke="#ffffff" strokeWidth={2} dot={false} activeDot={false} />
              {/* Margins */}
              <Line data={svmMargin1} dataKey="y" stroke="#94a3b8" strokeDasharray="3 3" dot={false} activeDot={false} />
              <Line data={svmMargin2} dataKey="y" stroke="#94a3b8" strokeDasharray="3 3" dot={false} activeDot={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Maximal Margin Hyperplane</p>
        </div>
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
        hyperparameters={[
          {
            name: 'var_smoothing',
            description: 'Portion of the largest variance of all features that is added to variances for calculation stability.',
            default: '1e-9',
            range: 'Float'
          },
          {
            name: 'priors',
            description: 'Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.',
            default: 'None',
            range: 'Array-like'
          }
        ]}
      >
        <div className="h-64 w-full">
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
             </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Modeling classes as Gaussian Probabilistic Distributions.</p>
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
           {
             name: 'criterion',
             description: 'The function to measure the quality of a split.',
             default: 'gini',
             range: 'gini, entropy, log_loss'
           },
           {
             name: 'max_depth',
             description: 'The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.',
             default: 'None',
             range: 'Integer'
           },
           {
             name: 'min_samples_split',
             description: 'The minimum number of samples required to split an internal node.',
             default: '2',
             range: 'Integer or Float'
           }
        ]}
      >
        <DecisionTreeViz />
      </AlgorithmCard>
    </div>
  );
};