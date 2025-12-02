import React, { useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter, ReferenceLine, ReferenceDot } from 'recharts';
import { MLModelType } from '../types';
import { Sliders, Activity, TrendingUp } from 'lucide-react';

// --- DATA & CONFIGURATION ---

// Comparison Data Dictionary (Existing)
const comparisonData = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    type: 'Linear',
    interpretability: 'High',
    trainingSpeed: 'Very Fast',
    overfittingRisk: 'Low (if regularized)',
    dataScale: 'Small to Medium',
    bestFor: 'Binary classification, baseline models',
    mathComplexity: 'Low'
  },
  [MLModelType.RANDOM_FOREST]: {
    type: 'Ensemble (Bagging)',
    interpretability: 'Medium',
    trainingSpeed: 'Medium',
    overfittingRisk: 'Low',
    dataScale: 'Large',
    bestFor: 'Tabular data, high accuracy requirements',
    mathComplexity: 'High'
  },
  [MLModelType.SVM]: {
    type: 'Geometry-based',
    interpretability: 'Low',
    trainingSpeed: 'Slow (Quadratic)',
    overfittingRisk: 'Medium',
    dataScale: 'Small to Medium',
    bestFor: 'High dimensional data, complex boundaries',
    mathComplexity: 'Very High'
  },
  [MLModelType.KNN]: {
    type: 'Instance-based',
    interpretability: 'High',
    trainingSpeed: 'None (Lazy)',
    overfittingRisk: 'Medium (depends on k)',
    dataScale: 'Small',
    bestFor: 'Simple patterns, small datasets',
    mathComplexity: 'Low'
  },
  [MLModelType.XGBOOST]: {
    type: 'Ensemble (Boosting)',
    interpretability: 'Low',
    trainingSpeed: 'Fast (Optimized)',
    overfittingRisk: 'High (needs tuning)',
    dataScale: 'Very Large',
    bestFor: 'Competitions, Structured data',
    mathComplexity: 'High'
  }
};

// Hyperparameter Definitions
const HYPERPARAMETERS = {
  [MLModelType.LOGISTIC_REGRESSION]: [
    { name: 'C', desc: 'Inverse of regularization strength. Smaller values specify stronger regularization (prevents overfitting).' },
    { name: 'penalty', desc: 'Norm used in penalization (l1 for Lasso, l2 for Ridge).' },
    { name: 'solver', desc: 'Algorithm to use in optimization (e.g., "liblinear" for small data, "saga" for large).' }
  ],
  [MLModelType.RANDOM_FOREST]: [
    { name: 'n_estimators', desc: 'The number of trees in the forest. More is usually better but slower.' },
    { name: 'max_depth', desc: 'Maximum depth of the tree. Controls complexity and overfitting.' },
    { name: 'min_samples_split', desc: 'Minimum number of samples required to split an internal node.' }
  ],
  [MLModelType.SVM]: [
    { name: 'C', desc: 'Regularization parameter. Controls trade-off between smooth decision boundary and classifying training points correctly.' },
    { name: 'kernel', desc: 'Specifies the kernel type (e.g., "linear", "rbf", "poly") to handle non-linear data.' },
    { name: 'gamma', desc: 'Kernel coefficient. Defines how far the influence of a single training example reaches.' }
  ],
  [MLModelType.KNN]: [
    { name: 'n_neighbors (k)', desc: 'Number of neighbors to use. Small k = noise sensitive, Large k = smooth boundaries.' },
    { name: 'weights', desc: 'Weight function used in prediction ("uniform" or "distance").' },
    { name: 'metric', desc: 'Distance metric to use (e.g., "euclidean", "manhattan").' }
  ],
  [MLModelType.XGBOOST]: [
    { name: 'learning_rate', desc: 'Step size shrinkage used to prevent overfitting. Range: [0, 1].' },
    { name: 'max_depth', desc: 'Maximum depth of a tree. Increasing this value will make the model more complex.' },
    { name: 'subsample', desc: 'Subsample ratio of the training instances. Setting it to 0.5 means XGBoost random samples 50% of data.' }
  ]
};

// Impact Simulation Data
interface ParamImpact {
  paramName: string;
  data: { x: number | string, y: number }[];
  labelX: string;
}

const MODEL_IMPACTS: Record<MLModelType, ParamImpact[]> = {
  [MLModelType.LOGISTIC_REGRESSION]: [
    {
      paramName: 'C (Inverse Reg)',
      labelX: 'C Value (Log Scale)',
      data: [
        { x: '0.001', y: 0.65 }, { x: '0.01', y: 0.72 }, { x: '0.1', y: 0.82 }, { x: '1.0', y: 0.85 }, { x: '10', y: 0.84 }, { x: '100', y: 0.83 }
      ]
    }
  ],
  [MLModelType.RANDOM_FOREST]: [
    {
      paramName: 'n_estimators',
      labelX: 'Number of Trees',
      data: [
        { x: 10, y: 0.78 }, { x: 50, y: 0.88 }, { x: 100, y: 0.91 }, { x: 200, y: 0.92 }, { x: 500, y: 0.925 }
      ]
    },
    {
      paramName: 'max_depth',
      labelX: 'Max Depth',
      data: [
        { x: 2, y: 0.70 }, { x: 5, y: 0.82 }, { x: 10, y: 0.89 }, { x: 20, y: 0.91 }, { x: 50, y: 0.88 } // Overfitting at 50
      ]
    }
  ],
  [MLModelType.SVM]: [
    {
      paramName: 'C (Regularization)',
      labelX: 'C Value',
      data: [
         { x: '0.1', y: 0.75 }, { x: '1', y: 0.86 }, { x: '10', y: 0.84 }, { x: '100', y: 0.81 }
      ]
    },
    {
      paramName: 'gamma',
      labelX: 'Gamma',
      data: [
         { x: '0.001', y: 0.70 }, { x: '0.01', y: 0.82 }, { x: '0.1', y: 0.88 }, { x: '1', y: 0.75 }, { x: '10', y: 0.65 }
      ]
    }
  ],
  [MLModelType.KNN]: [
    {
      paramName: 'n_neighbors',
      labelX: 'k Neighbors',
      data: [
        { x: 1, y: 0.74 }, { x: 3, y: 0.81 }, { x: 5, y: 0.85 }, { x: 10, y: 0.83 }, { x: 20, y: 0.79 }, { x: 50, y: 0.72 }
      ]
    }
  ],
  [MLModelType.XGBOOST]: [
    {
      paramName: 'learning_rate',
      labelX: 'Learning Rate',
      data: [
        { x: 0.01, y: 0.82 }, { x: 0.05, y: 0.89 }, { x: 0.1, y: 0.93 }, { x: 0.3, y: 0.91 }, { x: 0.5, y: 0.86 }, { x: 1.0, y: 0.78 }
      ]
    },
    {
      paramName: 'n_estimators',
      labelX: 'Boosting Rounds',
      data: [
        { x: 50, y: 0.85 }, { x: 100, y: 0.91 }, { x: 200, y: 0.93 }, { x: 500, y: 0.935 }
      ]
    }
  ]
};

// Visualization Data Generators
const generateSigmoid = () => Array.from({ length: 20 }, (_, i) => ({ x: i - 10, y: 1 / (1 + Math.exp(-(i - 10))) }));
const generateTrees = () => Array.from({ length: 10 }, (_, i) => ({ trees: (i + 1) * 10, accuracy: 0.7 + (0.25 * (1 - Math.exp(-0.3 * i))) }));
const generateBoosting = () => Array.from({ length: 15 }, (_, i) => ({ iter: i, loss: Math.exp(-0.4 * i) + 0.1 }));
const generateSVMData = () => [
  { x: 2, y: 2, class: 'A' }, { x: 3, y: 3, class: 'A' }, 
  { x: 7, y: 7, class: 'B' }, { x: 8, y: 8, class: 'B' }
];
const generateKNNData = () => [
  { x: 50, y: 50, type: 'Target' },
  { x: 45, y: 45, type: 'N' }, { x: 55, y: 55, type: 'N' }, { x: 52, y: 48, type: 'N' },
  { x: 20, y: 20, type: 'Other' }, { x: 80, y: 80, type: 'Other' }
];

// --- HELPER COMPONENTS ---

interface SensitivityPanelProps {
  model: MLModelType;
  impacts: ParamImpact[];
  color: string;
}

const SensitivityPanel: React.FC<SensitivityPanelProps> = ({ model, impacts, color }) => {
  const [selectedParamIdx, setSelectedParamIdx] = useState(0);
  const [sliderIndex, setSliderIndex] = useState(0);

  // When model changes, reset param index
  useEffect(() => {
    setSelectedParamIdx(0);
  }, [model]);

  // When param (or model) changes, reset slider to middle
  useEffect(() => {
    if (impacts && impacts[selectedParamIdx]) {
      setSliderIndex(Math.floor(impacts[selectedParamIdx].data.length / 2));
    }
  }, [selectedParamIdx, impacts, model]);

  if (!impacts) return <div className="text-slate-500 text-sm">No sensitivity data available for this model.</div>;

  const currentParam = impacts[selectedParamIdx];
  const currentDataPoint = currentParam.data[sliderIndex];

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header & Dropdown */}
      <div className="flex justify-between items-center bg-slate-800/50 p-2 rounded-t-lg border-b border-slate-800">
        <span className="text-sm font-bold pl-2" style={{ color }}>{model}</span>
        <select 
          value={selectedParamIdx}
          onChange={(e) => setSelectedParamIdx(Number(e.target.value))}
          className="bg-slate-900 text-xs text-slate-300 p-1.5 rounded border border-slate-700 focus:outline-none focus:border-indigo-500 cursor-pointer"
        >
          {impacts.map((p, i) => (
            <option key={i} value={i}>Effect of {p.paramName}</option>
          ))}
        </select>
      </div>

      {/* Chart */}
      <div className="h-48 bg-slate-950 rounded-lg border border-slate-800 p-2 relative">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={currentParam.data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis 
              dataKey="x" 
              stroke="#64748b" 
              fontSize={10} 
              label={{ value: currentParam.labelX, position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }} 
            />
            <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} tickFormatter={(val) => val.toFixed(1)} />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} formatter={(val: number) => [val.toFixed(2), 'Accuracy']} />
            <Line 
              type="monotone" 
              dataKey="y" 
              stroke={color} 
              strokeWidth={2} 
              dot={{ r: 3, fill: color }} 
              activeDot={{ r: 6 }}
              animationDuration={500}
              name="Accuracy"
            />
            {/* Interactive Dot tracking the slider */}
            <ReferenceDot 
              x={currentDataPoint.x} 
              y={currentDataPoint.y} 
              r={6} 
              fill="#ffffff" 
              stroke={color} 
              strokeWidth={2} 
              isFront={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Interactive Slider Control */}
      <div className="bg-slate-800 p-4 rounded-b-lg border border-t-0 border-slate-800 shadow-inner">
        <div className="flex justify-between text-xs text-slate-400 mb-3 font-mono">
          <span>Hyperparameter: <strong className="text-white text-sm ml-1">{currentDataPoint.x}</strong></span>
          <span>Accuracy: <strong style={{ color }} className="text-sm ml-1">{(currentDataPoint.y * 100).toFixed(1)}%</strong></span>
        </div>
        <input 
          type="range" 
          min={0} 
          max={currentParam.data.length - 1} 
          value={sliderIndex}
          onChange={(e) => setSliderIndex(Number(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          style={{ accentColor: color }}
        />
        <div className="flex justify-between text-[10px] text-slate-600 mt-2 font-mono uppercase">
           <span>Low</span>
           <span>High</span>
        </div>
      </div>
    </div>
  );
};

// --- MAIN COMPONENT ---

export const ModelComparisonView: React.FC = () => {
  const [modelA, setModelA] = useState<MLModelType>(MLModelType.LOGISTIC_REGRESSION);
  const [modelB, setModelB] = useState<MLModelType>(MLModelType.XGBOOST);

  const renderVisual = (model: MLModelType, color: string) => {
    switch (model) {
      case MLModelType.LOGISTIC_REGRESSION:
        return (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={generateSigmoid()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis hide />
                <YAxis domain={[0, 1]} hide />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a' }} />
                <Line type="monotone" dataKey="y" stroke={color} strokeWidth={3} dot={false} />
                <ReferenceLine y={0.5} stroke="#64748b" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-center text-xs text-slate-500 mt-2">Sigmoid Activation Curve</p>
          </div>
        );
      case MLModelType.RANDOM_FOREST:
        return (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={generateTrees()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="trees" tick={{ fontSize: 10 }} stroke="#64748b" />
                <YAxis domain={[0, 1]} hide />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a' }} />
                <Bar dataKey="accuracy" fill={color} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="text-center text-xs text-slate-500 mt-2">Accuracy vs. No. of Trees</p>
          </div>
        );
      case MLModelType.XGBOOST:
        return (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={generateBoosting()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis hide />
                <YAxis hide />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a' }} />
                <Area type="monotone" dataKey="loss" stroke={color} fill={color} fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
            <p className="text-center text-xs text-slate-500 mt-2">Loss Reduction over Iterations</p>
          </div>
        );
      case MLModelType.SVM:
        return (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" hide domain={[0, 10]} />
                <YAxis type="number" dataKey="y" hide domain={[0, 10]} />
                <ReferenceLine segment={[{x: 0, y: 0}, {x: 10, y: 10}]} stroke={color} strokeWidth={2} />
                <ReferenceLine segment={[{x: 1, y: 0}, {x: 10, y: 9}]} stroke="#64748b" strokeDasharray="3 3" />
                <ReferenceLine segment={[{x: 0, y: 1}, {x: 9, y: 10}]} stroke="#64748b" strokeDasharray="3 3" />
                <Scatter data={generateSVMData()} fill="#cbd5e1" />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-center text-xs text-slate-500 mt-2">Maximal Margin Hyperplane</p>
          </div>
        );
      case MLModelType.KNN:
        return (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" hide domain={[0, 100]} />
                <YAxis type="number" dataKey="y" hide domain={[0, 100]} />
                <ReferenceDot x={50} y={50} r={25} fill={color} fillOpacity={0.2} stroke="none" />
                <Scatter data={generateKNNData()} fill="#cbd5e1" />
                <ReferenceDot x={50} y={50} r={4} fill={color} stroke="#fff" />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-center text-xs text-slate-500 mt-2">Local Neighborhood (k=3)</p>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      <header className="mb-8">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">The Model Battleground</h1>
        <p className="text-slate-400 text-lg">
          Directly compare algorithms to understand trade-offs in speed, accuracy, and interpretability.
        </p>
      </header>

      {/* Selectors */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <div className="bg-slate-900 p-4 rounded-t-lg border-b-4 border-indigo-500 shadow-lg">
           <label className="block text-xs font-bold text-indigo-400 uppercase mb-2">Challenger A</label>
           <select 
             value={modelA} 
             onChange={(e) => setModelA(e.target.value as MLModelType)}
             className="w-full bg-slate-800 text-white p-2 rounded border border-slate-700 focus:ring-1 focus:ring-indigo-500"
           >
             {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
           </select>
        </div>
        <div className="bg-slate-900 p-4 rounded-t-lg border-b-4 border-emerald-500 shadow-lg">
           <label className="block text-xs font-bold text-emerald-400 uppercase mb-2">Challenger B</label>
           <select 
             value={modelB} 
             onChange={(e) => setModelB(e.target.value as MLModelType)}
             className="w-full bg-slate-800 text-white p-2 rounded border border-slate-700 focus:ring-1 focus:ring-emerald-500"
           >
             {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
           </select>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-800 shadow-xl">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-950 text-slate-400 text-sm uppercase">
              <th className="p-4 font-medium w-1/3 border-b border-slate-800">Criterion</th>
              <th className="p-4 font-medium w-1/3 text-indigo-400 border-b border-slate-800 border-l border-slate-800">{modelA}</th>
              <th className="p-4 font-medium w-1/3 text-emerald-400 border-b border-slate-800 border-l border-slate-800">{modelB}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {[
              { label: 'Algorithm Type', key: 'type' },
              { label: 'Interpretability', key: 'interpretability' },
              { label: 'Training Speed', key: 'trainingSpeed' },
              { label: 'Overfitting Risk', key: 'overfittingRisk' },
              { label: 'Ideal Data Scale', key: 'dataScale' },
              { label: 'Math Complexity', key: 'mathComplexity' },
              { label: 'Best Use Case', key: 'bestFor' },
            ].map((row) => (
              <tr key={row.key} className="hover:bg-slate-800/50 transition-colors">
                <td className="p-4 text-slate-300 font-medium text-sm">{row.label}</td>
                {/* @ts-ignore */}
                <td className="p-4 text-indigo-200 text-sm border-l border-slate-800">{comparisonData[modelA][row.key]}</td>
                {/* @ts-ignore */}
                <td className="p-4 text-emerald-200 text-sm border-l border-slate-800">{comparisonData[modelB][row.key]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Visual & Hyperparameter Deep Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">
        {/* Model A Deep Dive */}
        <div className="space-y-4">
            <h3 className="text-xl font-bold text-indigo-400 flex items-center gap-2">
                <Activity size={20} /> Visual Analysis: {modelA}
            </h3>
            <div className="h-48 bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-lg">
                {renderVisual(modelA, '#818cf8')}
            </div>
            
            <div className="mt-6">
                <h4 className="text-sm font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Sliders size={16} /> Key Hyperparameters
                </h4>
                <div className="bg-slate-900 rounded-lg border border-slate-800 divide-y divide-slate-800">
                    {HYPERPARAMETERS[modelA].map((param) => (
                        <div key={param.name} className="p-3">
                            <div className="flex items-center justify-between mb-1">
                                <code className="text-xs font-mono text-indigo-300 bg-indigo-900/30 px-1.5 py-0.5 rounded">{param.name}</code>
                            </div>
                            <p className="text-xs text-slate-400 leading-relaxed">{param.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>

        {/* Model B Deep Dive */}
        <div className="space-y-4">
            <h3 className="text-xl font-bold text-emerald-400 flex items-center gap-2">
                <Activity size={20} /> Visual Analysis: {modelB}
            </h3>
            <div className="h-48 bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-lg">
                {renderVisual(modelB, '#34d399')}
            </div>

            <div className="mt-6">
                <h4 className="text-sm font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Sliders size={16} /> Key Hyperparameters
                </h4>
                <div className="bg-slate-900 rounded-lg border border-slate-800 divide-y divide-slate-800">
                    {HYPERPARAMETERS[modelB].map((param) => (
                        <div key={param.name} className="p-3">
                            <div className="flex items-center justify-between mb-1">
                                <code className="text-xs font-mono text-emerald-300 bg-emerald-900/30 px-1.5 py-0.5 rounded">{param.name}</code>
                            </div>
                            <p className="text-xs text-slate-400 leading-relaxed">{param.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
      </div>

      {/* Hyperparameter Sensitivity Analysis Section (Interactive) */}
      <div className="mt-16 bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg">
         <div className="mb-6 border-b border-slate-800 pb-4">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
               <TrendingUp size={24} className="text-fuchsia-400" /> 
               Hyperparameter Sensitivity Analysis
            </h3>
            <p className="text-slate-400 text-sm mt-1">
              Simulate hyperparameter tuning. Adjust the sliders below to see how critical parameters impact model accuracy on validation data.
            </p>
         </div>

         <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <SensitivityPanel 
              model={modelA} 
              impacts={MODEL_IMPACTS[modelA]} 
              color="#818cf8" 
            />
            <SensitivityPanel 
              model={modelB} 
              impacts={MODEL_IMPACTS[modelB]} 
              color="#34d399" 
            />
         </div>
      </div>
    </div>
  );
};