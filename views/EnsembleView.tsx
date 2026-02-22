
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

const DecisionBoundaryViz = () => {
    const [mode, setMode] = useState<'single' | 'forest'>('single');
    return (
        <div className="flex flex-col gap-6 items-center">
            <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800">
                <button onClick={() => setMode('single')} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${mode === 'single' ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}>Single Decision Tree</button>
                <button onClick={() => setMode('forest')} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${mode === 'forest' ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}>Random Forest</button>
            </div>
            
            <div className="relative w-80 h-80 bg-slate-950 rounded-2xl border-2 border-slate-800 overflow-hidden shadow-inner">
                {/* Background Grid simulating decision space */}
                <div className="absolute inset-0 grid grid-cols-10 grid-rows-10 opacity-20">
                    {Array.from({length: 100}).map((_, i) => {
                        // Complex boundary logic
                        const r = Math.floor(i / 10);
                        const c = i % 10;
                        const isClassA = mode === 'single' 
                            ? (r < 5 && c < 4) || (r > 7 && c > 7) // Jagged, rigid splits
                            : (r + c < 11 && r > 1 && c > 1); // Smoother diagonal boundary
                        return <div key={i} className={`border border-white/5 transition-colors duration-500 ${isClassA ? 'bg-indigo-500' : 'bg-emerald-500'}`}></div>;
                    })}
                </div>
                {/* Simulated Data Points */}
                <div className="absolute inset-0 p-4">
                     {Array.from({length: 20}).map((_, i) => (
                         <div key={i} className={`absolute w-3 h-3 rounded-full border border-white/20 shadow-lg ${i % 2 === 0 ? 'bg-indigo-300' : 'bg-emerald-300'}`} style={{ top: `${Math.random()*80 + 10}%`, left: `${Math.random()*80 + 10}%` }}></div>
                     ))}
                </div>
            </div>
            <p className="text-[10px] text-slate-500 text-center max-w-xs leading-relaxed font-mono uppercase">
                {mode === 'single' ? "High Variance: Boundaries are jagged and axis-aligned, overfitting to noise." : "Low Variance: Averaging multiple trees creates a smooth, generalized boundary."}
            </p>
        </div>
    );
};

const GradientBoostingViz = () => {
    const [iter, setIter] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const MAX_ITER = 10;
    useEffect(() => {
        let interval: any;
        if (isPlaying) {
            interval = setInterval(() => {
                setIter(prev => prev >= MAX_ITER ? (setIsPlaying(false), prev) : prev + 1);
            }, 800);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);
    const data = useMemo(() => Array.from({ length: 21 }, (_, i) => ({ x: (i / 20) * 10 - 5, y: ((i / 20) * 10 - 5) ** 2 + (Math.random() - 0.5) * 5 })), []);
    const boostingSteps = useMemo(() => {
        const steps = [];
        const meanY = data.reduce((sum, p) => sum + p.y, 0) / data.length;
        let currentPreds = data.map(p => ({ x: p.x, y: p.y, pred: meanY }));
        steps.push(currentPreds);
        for (let i = 0; i < MAX_ITER; i++) {
            const nextPreds = currentPreds.map(p => ({ ...p, pred: p.pred + 0.3 * (p.y - p.pred) * 0.7 }));
            steps.push(nextPreds);
            currentPreds = nextPreds;
        }
        return steps;
    }, [data]);
    const currentStepData = boostingSteps[iter];
    return (
        <div className="flex flex-col gap-6">
             <div className="flex justify-between items-center bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-4">
                    <button onClick={() => setIsPlaying(!isPlaying)} className="p-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-white shadow-lg shadow-indigo-900/40">{isPlaying ? <Pause size={18} /> : <Play size={18} />}</button>
                    <button onClick={() => { setIsPlaying(false); setIter(0); }} className="p-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-white border border-slate-700"><RotateCcw size={18} /></button>
                    <div className="text-xs font-black font-mono text-slate-500">ITERATION: <span className="text-indigo-400">{iter}</span></div>
                </div>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 <div className="h-64 bg-slate-950 rounded-2xl border border-slate-900 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={currentStepData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="x" type="number" hide domain={[-5, 5]} />
                            <YAxis hide domain={[-5, 30]} />
                            <Scatter dataKey="y" fill="#475569" opacity={0.4} />
                            <Line type="monotone" dataKey="pred" stroke="#f43f5e" strokeWidth={4} dot={false} animationDuration={300} />
                        </ComposedChart>
                    </ResponsiveContainer>
                 </div>
                 <div className="h-64 bg-slate-950 rounded-2xl border border-slate-900 p-4">
                     <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={currentStepData.map(d => ({x: d.x, r: d.y - d.pred}))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="x" hide />
                            <YAxis hide domain={[-10, 10]} />
                            <Bar dataKey="r" fill="#6366f1" radius={[4, 4, 0, 0]} animationDuration={300} />
                        </BarChart>
                     </ResponsiveContainer>
                 </div>
             </div>
        </div>
    );
};

import { motion } from 'framer-motion';

export const EnsembleView: React.FC = () => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-16 pb-20"
    >
      <header className="border-b border-slate-800 pb-12">
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-6xl font-serif font-bold text-white mb-6"
        >
          Ensemble Methods
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light"
        >
          Combining the wisdom of crowds. Ensemble models orchestrate multiple weak learners to achieve a collective intelligence that is robust, stable, and accurate.
        </motion.p>
      </header>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.1
            }
          }
        }}
      >
      <AlgorithmCard
        id="random-forest" title="Random Forest" complexity="Intermediate"
        theory="A massive ensemble of decision trees. Using Bootstrap Aggregating (Bagging), each tree is trained on a different subset of data. The variance of individual trees is averaged away, resulting in a smooth and generalized decision boundary."
        math={<span>y&#770; = <sup>1</sup>&frasl;<sub>B</sub> &Sigma; f<sub>b</sub>(x)</span>} mathLabel="Aggregate Wisdom"
        code={`from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)`}
        pros={['Extremely robust to outliers', 'Inherent feature importance calculation', 'Low risk of overfitting']}
        cons={['Slow inference time', 'Large storage requirements', 'Not easily interpretable']}
        steps={[
            "Open Google Colab. Import `RandomForestClassifier` from `sklearn.ensemble`.",
            "Load and split your dataset.",
            "Initialize: `model = RandomForestClassifier(n_estimators=100)`.",
            "Fit the model to the training set.",
            "Use `model.feature_importances_` to see which variables matter most.",
            "Predict and evaluate accuracy."
        ]}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
             <DecisionBoundaryViz />
             <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={featureImportanceData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis type="number" stroke="#475569" fontSize={10} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <YAxis dataKey="feature" type="category" stroke="#475569" fontSize={10} width={60} />
                    <Tooltip 
                        cursor={{fill: '#1e293b', opacity: 0.4}} 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                    />
                    <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} activeBar={{ fill: '#818cf8' }} />
                    </BarChart>
                </ResponsiveContainer>
                <p className="text-[10px] text-center text-slate-600 mt-4 uppercase tracking-widest font-mono">Relative Feature Contribution</p>
            </div>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="gradient-boosting" title="Gradient Boosting" complexity="Advanced"
        theory="Sequential optimization. Instead of training trees in parallel, Boosting trains trees one after another. Each new tree focuses exclusively on the errors (residuals) of the entire ensemble that came before it."
        math={<span>F<sub>m</sub>(x) = F<sub>m-1</sub>(x) + &nu; h<sub>m</sub>(x)</span>} mathLabel="Additive Model Update"
        code={`import xgboost as xgb\nmodel = xgb.XGBClassifier(learning_rate=0.1)`}
        pros={['Best-in-class performance on tabular data', 'Handles missing values natively', 'Flexible cost functions']}
        cons={['Requires careful hyperparameter tuning', 'Sensitive to noise', 'Sequential training is slower']}
        steps={[
            "Install XGBoost in Colab: `!pip install xgboost`.",
            "Import `XGBClassifier` or `XGBRegressor`.",
            "Prepare data. XGBoost can handle missing values, but encoding categoricals is still needed.",
            "Initialize: `model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)`.",
            "Fit the model. Watch for overfitting if `n_estimators` is too high.",
            "Evaluate performance."
        ]}
      >
         <GradientBoostingViz />
      </AlgorithmCard>
      </motion.div>
    </motion.div>
  );
};
