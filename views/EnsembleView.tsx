import React, { useState, useEffect, useMemo } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, ComposedChart, Scatter, ReferenceLine } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { Play, Pause, RotateCcw } from 'lucide-react';

const featureImportanceData = [
  { feature: 'Age', importance: 0.15 },
  { feature: 'BMI', importance: 0.12 },
  { feature: 'Glucose', importance: 0.35 },
  { feature: 'BP', importance: 0.08 },
  { feature: 'Insulin', importance: 0.25 },
];

const adaWeightsData = [
    { estimator: 'Tree 1', weight: 0.3 },
    { estimator: 'Tree 2', weight: 0.5 },
    { estimator: 'Tree 3', weight: 0.8 },
    { estimator: 'Tree 4', weight: 1.1 },
    { estimator: 'Tree 5', weight: 1.5 },
];

const GradientBoostingViz = () => {
    const [iter, setIter] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const MAX_ITER = 10;

    useEffect(() => {
        let interval: any;
        if (isPlaying) {
            interval = setInterval(() => {
                setIter(prev => {
                    if (prev >= MAX_ITER) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 800);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    // Generate Data (Quadratic with noise)
    const data = useMemo(() => {
        const points = [];
        for (let i = 0; i <= 20; i++) {
            const x = (i / 20) * 10 - 5; // -5 to 5
            const noise = (Math.random() - 0.5) * 5;
            const y = x * x + noise; // y = x^2 + noise
            points.push({ x, y });
        }
        return points;
    }, []);

    // Compute boosting steps
    const boostingSteps = useMemo(() => {
        const steps = [];
        // Initial Prediction: Mean
        const meanY = data.reduce((sum, p) => sum + p.y, 0) / data.length;
        let currentPreds = data.map(p => ({ x: p.x, y: p.y, pred: meanY }));

        steps.push(currentPreds); // Iter 0

        for (let i = 0; i < MAX_ITER; i++) {
            // Gradient Boosting Logic Simulation
            // 1. Calculate Residuals (Gradient of Loss)
            const learningRate = 0.3;
            const nextPreds = currentPreds.map(p => {
                const residual = p.y - p.pred;
                // Simulate Weak Learner: In reality, we fit a tree to residuals.
                // Here, we act as if the weak learner captures 70% of the residual signal locally.
                const weakLearnerOutput = residual * 0.7; 
                return {
                    ...p,
                    pred: p.pred + learningRate * weakLearnerOutput,
                    residual: residual
                };
            });
            steps.push(nextPreds);
            currentPreds = nextPreds;
        }
        return steps;
    }, [data]);

    const currentStepData = boostingSteps[iter];
    // Residuals for the *next* step to fix (or current error)
    const residualsData = currentStepData.map(d => ({ x: d.x, residual: d.y - d.pred }));
    
    // Calculate MSE for display
    const mse = currentStepData.reduce((sum, p) => sum + (p.y - p.pred) ** 2, 0) / currentStepData.length;

    return (
        <div className="flex flex-col gap-4">
             {/* Controls */}
             <div className="flex justify-between items-center bg-slate-800 p-3 rounded-lg border border-slate-700">
                <div className="flex items-center gap-4">
                    <button 
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="p-2 bg-indigo-600 hover:bg-indigo-500 rounded-full text-white transition-colors shadow-lg shadow-indigo-900/50"
                    >
                        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                    </button>
                    <button 
                         onClick={() => { setIsPlaying(false); setIter(0); }}
                         className="p-2 bg-slate-700 hover:bg-slate-600 rounded-full text-white transition-colors border border-slate-600"
                    >
                        <RotateCcw size={16} />
                    </button>
                    <div className="text-sm font-mono text-slate-300">
                        Iteration: <span className="text-indigo-400 font-bold">{iter}</span> / {MAX_ITER}
                    </div>
                </div>
                <div className="text-sm font-mono text-slate-400">
                    MSE: <span className="text-emerald-400 font-bold">{mse.toFixed(2)}</span>
                </div>
             </div>

             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                 {/* Main Plot */}
                 <div className="h-64 bg-slate-900 rounded-lg border border-slate-800 p-2 shadow-inner">
                    <p className="text-xs text-center text-slate-500 mb-1 font-mono uppercase">Model Prediction vs True Data</p>
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={currentStepData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="x" type="number" hide domain={[-5, 5]} />
                            <YAxis hide domain={[-5, 30]} />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f8fafc' }} />
                            <Scatter name="True Data" dataKey="y" fill="#94a3b8" opacity={0.6} shape="circle" />
                            <Line type="monotone" dataKey="pred" stroke="#f43f5e" strokeWidth={3} dot={false} animationDuration={300} />
                        </ComposedChart>
                    </ResponsiveContainer>
                 </div>

                 {/* Residuals Plot */}
                 <div className="h-64 bg-slate-900 rounded-lg border border-slate-800 p-2 shadow-inner">
                     <p className="text-xs text-center text-slate-500 mb-1 font-mono uppercase">Residuals (Errors to fix)</p>
                     <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={residualsData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="x" hide />
                            <YAxis hide domain={[-10, 10]} />
                            <ReferenceLine y={0} stroke="#475569" />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f8fafc' }} />
                            <Bar dataKey="residual" fill="#818cf8" radius={[2, 2, 0, 0]} animationDuration={300} />
                        </BarChart>
                     </ResponsiveContainer>
                 </div>
             </div>
             
             <div className="text-xs text-slate-400 bg-slate-950 p-3 rounded border border-slate-800 leading-relaxed">
                <strong className="text-indigo-400">How it works:</strong> At each iteration, the algorithm calculates the <span className="text-indigo-300">residuals</span> (blue bars) — the difference between the true target and the current prediction. It then trains a new weak learner to predict these residuals and adds a fraction of that prediction (learning rate) to the ensemble (red line). This progressively "nudges" the model towards the true data.
             </div>
        </div>
    );
};

export const EnsembleView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Ensemble Methods</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Ensemble methods combine multiple machine learning models to create a more powerful and robust model. They typically reduce variance (bagging), bias (boosting), or improve predictions (stacking).
        </p>
      </header>

      <AlgorithmCard
        id="random-forest"
        title="Random Forest"
        theory="An ensemble learning method that constructs a multitude of decision trees at training time. It uses 'Bagging' (Bootstrap Aggregating) and feature randomness to create diverse trees and outputs the mode of the classes (classification) or mean prediction (regression)."
        math={<span>y&#770; = <sup>1</sup>&frasl;<sub>B</sub> &Sigma;<sub>b=1</sub><sup>B</sup> f<sub>b</sub>(x)</span>}
        mathLabel="Averaging Predictions"
        code={`from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)`}
        pros={['Reduces overfitting compared to decision trees', 'Handles missing values', 'Provides feature importance']}
        cons={['Slow to train and predict', 'Complex model (black box)', 'Large memory footprint']}
        hyperparameters={[
          {
            name: 'n_estimators',
            description: 'The number of trees in the forest.',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'max_depth',
            description: 'The maximum depth of the tree.',
            default: 'None',
            range: 'Integer'
          },
          {
            name: 'max_features',
            description: 'The number of features to consider when looking for the best split.',
            default: 'sqrt',
            range: 'sqrt, log2, None'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={featureImportanceData} margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={80} />
              <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
              <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} name="Importance" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Feature Importance Extraction</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="adaboost"
        title="AdaBoost (Adaptive Boosting)"
        theory="A boosting technique that trains predictors sequentially. Each subsequent model attempts to correct the errors of its predecessor by increasing the weights of misclassified instances."
        math={<span>H(x) = sign(&Sigma;<sub>t=1</sub><sup>T</sup> &alpha;<sub>t</sub>h<sub>t</sub>(x))</span>}
        mathLabel="Weighted Sum of Weak Learners"
        code={`from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
pred = ada.predict(X_test)`}
        pros={['Less prone to overfitting', 'Easy to implement', 'Can use various base classifiers']}
        cons={['Sensitive to noisy data and outliers', 'Slower training than bagging']}
        hyperparameters={[
          {
            name: 'n_estimators',
            description: 'The maximum number of estimators at which boosting is terminated.',
            default: '50',
            range: 'Integer'
          },
          {
            name: 'learning_rate',
            description: 'Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.',
            default: '1.0',
            range: 'Float'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
             <BarChart data={adaWeightsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="estimator" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                <Bar dataKey="weight" fill="#f472b6" radius={[4, 4, 0, 0]} name="Estimator Weight (Alpha)" />
             </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Subsequent estimators get higher weight (alpha) as they correct previous errors.</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="gradient-boosting"
        title="Gradient Boosting (XGBoost/LightGBM)"
        theory="Gradient Boosting builds an additive model in a forward stage-wise fashion. Instead of updating weights of data points (like AdaBoost), it generalizes boosting by allowing optimization of an arbitrary differentiable loss function. It effectively trains new models to predict the residuals (errors) of prior models."
        math={<span>F<sub>m</sub>(x) = F<sub>m-1</sub>(x) + <span className="math-serif">&nu;</span> &Sigma; h<sub>m</sub>(x)</span>}
        mathLabel="Additive Model Update (v = learning rate)"
        code={`import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
pred = model.predict(X_test)`}
        pros={['State-of-the-art performance on tabular data', 'Handles missing data automatically', 'Flexible objective functions']}
        cons={['Many hyperparameters to tune', 'Computationally expensive (sequential)', 'Can overfit if not tuned properly']}
        hyperparameters={[
          {
            name: 'learning_rate',
            description: 'Step size shrinkage used in update to prevent overfitting. Lower values require more trees.',
            default: '0.1',
            range: '[0, 1]'
          },
          {
            name: 'n_estimators',
            description: 'Number of boosting rounds (trees to build).',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'subsample',
            description: 'Subsample ratio of the training instances. Setting it to 0.5 means XGBoost random samples 50% of data.',
            default: '1.0',
            range: '(0, 1]'
          }
        ]}
      >
         <GradientBoostingViz />
      </AlgorithmCard>
    </div>
  );
};