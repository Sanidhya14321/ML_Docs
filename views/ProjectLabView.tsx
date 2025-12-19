import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Cell, ScatterChart, Scatter } from 'recharts';
import { MLModelType, ModelMetrics } from '../types';
import { Loader2, Code, Activity, Database, Info, ShieldCheck, TrendingUp, BarChart3 } from 'lucide-react';
import { CodeBlock } from '../components/CodeBlock';

const MODEL_DATA: Record<MLModelType, ModelMetrics> = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    accuracy: 85.4, precision: 82.1, recall: 78.5,
    confusionMatrix: [{ name: 'TP', value: 120 }, { name: 'FP', value: 30 }, { name: 'TN', value: 145 }, { name: 'FN', value: 25 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.1, tpr: 0.6 }, { fpr: 0.3, tpr: 0.8 }, { fpr: 0.5, tpr: 0.88 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.RANDOM_FOREST]: {
    accuracy: 94.2, precision: 93.5, recall: 91.0,
    confusionMatrix: [{ name: 'TP', value: 140 }, { name: 'FP', value: 10 }, { name: 'TN', value: 160 }, { name: 'FN', value: 10 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.05, tpr: 0.8 }, { fpr: 0.1, tpr: 0.92 }, { fpr: 0.2, tpr: 0.96 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.SVM]: {
    accuracy: 88.9, precision: 86.4, recall: 84.2,
    confusionMatrix: [{ name: 'TP', value: 130 }, { name: 'FP', value: 20 }, { name: 'TN', value: 150 }, { name: 'FN', value: 20 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.15, tpr: 0.7 }, { fpr: 0.25, tpr: 0.85 }, { fpr: 0.4, tpr: 0.9 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.KNN]: {
    accuracy: 86.1, precision: 83.0, recall: 81.5,
    confusionMatrix: [{ name: 'TP', value: 125 }, { name: 'FP', value: 28 }, { name: 'TN', value: 147 }, { name: 'FN', value: 22 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.12, tpr: 0.65 }, { fpr: 0.28, tpr: 0.82 }, { fpr: 0.45, tpr: 0.89 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.XGBOOST]: {
    accuracy: 96.5, precision: 95.8, recall: 94.2,
    confusionMatrix: [{ name: 'TP', value: 145 }, { name: 'FP', value: 5 }, { name: 'TN', value: 165 }, { name: 'FN', value: 8 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.02, tpr: 0.85 }, { fpr: 0.08, tpr: 0.95 }, { fpr: 0.15, tpr: 0.98 }, { fpr: 1, tpr: 1 }]
  }
};

const CorrelationHeatmap = () => {
    const features = ['Age', 'BP', 'Chol', 'HR', 'Target'];
    const matrix = [
        [1.0, 0.3, 0.2, -0.4, 0.2],
        [0.3, 1.0, 0.1, -0.1, 0.4],
        [0.2, 0.1, 1.0, -0.1, 0.1],
        [-0.4, -0.1, -0.1, 1.0, -0.4],
        [0.2, 0.4, 0.1, -0.4, 1.0]
    ];

    const getColor = (val: number) => {
        if (val > 0.7) return 'bg-emerald-500';
        if (val > 0.4) return 'bg-emerald-600';
        if (val > 0.2) return 'bg-emerald-700';
        if (val < -0.4) return 'bg-rose-600';
        if (val < -0.2) return 'bg-rose-700';
        return 'bg-slate-800';
    };

    return (
        <div className="flex flex-col items-center">
            <div className="grid grid-cols-6 gap-2">
                <div className="w-16 h-12"></div>
                {features.map(f => <div key={f} className="w-16 h-12 flex items-center justify-center text-[10px] font-black text-slate-500 uppercase">{f}</div>)}
                
                {matrix.map((row, i) => (
                    <React.Fragment key={i}>
                        <div className="w-16 h-12 flex items-center justify-end pr-2 text-[10px] font-black text-slate-500 uppercase">{features[i]}</div>
                        {row.map((val, j) => (
                            <div key={j} className={`w-16 h-12 flex items-center justify-center rounded-lg border border-white/5 transition-all hover:scale-105 ${getColor(val)}`}>
                                <span className="text-[10px] font-mono font-bold text-white/90">{val.toFixed(1)}</span>
                            </div>
                        ))}
                    </React.Fragment>
                ))}
            </div>
            <div className="mt-6 flex gap-4 text-[9px] font-mono text-slate-500">
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-emerald-500"></div> Strong Pos</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-rose-500"></div> Strong Neg</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-slate-800"></div> Weak</div>
            </div>
        </div>
    );
};

const ConfusionMatrixHeatmap = ({ matrix, isTraining }: { matrix: any[], isTraining: boolean }) => {
  if (isTraining) return <div className="h-64 flex items-center justify-center text-indigo-400"><Loader2 className="animate-spin w-8 h-8" /></div>;
  const data = {
    tp: matrix.find(m => m.name === 'TP')?.value || 0,
    fp: matrix.find(m => m.name === 'FP')?.value || 0,
    tn: matrix.find(m => m.name === 'TN')?.value || 0,
    fn: matrix.find(m => m.name === 'FN')?.value || 0,
  };
  const max = Math.max(data.tp, data.fp, data.tn, data.fn);
  const Cell = ({ label, value, sub, colorClass, bgClass }: any) => (
    <div className={`relative flex flex-col items-center justify-center p-6 rounded-2xl border border-white/5 transition-all hover:scale-[1.02] ${bgClass}`}>
       <div className={`absolute inset-0 rounded-2xl opacity-20 ${colorClass}`} style={{ opacity: (value / max) * 0.4 }}></div>
       <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 z-10">{label}</span>
       <span className="text-3xl font-mono font-black text-white z-10">{value}</span>
       <span className="text-[9px] text-slate-400 font-mono mt-2 z-10 italic">{sub}</span>
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-4 h-full">
      <Cell label="True Positive" value={data.tp} sub="Correctly Flagged" colorClass="bg-emerald-500" bgClass="bg-emerald-500/5" />
      <Cell label="False Positive" value={data.fp} sub="False Alarm" colorClass="bg-rose-500" bgClass="bg-rose-500/5" />
      <Cell label="False Negative" value={data.fn} sub="Missed Case" colorClass="bg-rose-500" bgClass="bg-rose-500/5" />
      <Cell label="True Negative" value={data.tn} sub="Correctly Safe" colorClass="bg-emerald-500" bgClass="bg-emerald-500/5" />
    </div>
  );
};

export const ProjectLabView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'eda' | 'code' | 'performance'>('performance');
  const [selectedModel, setSelectedModel] = useState<MLModelType>(MLModelType.LOGISTIC_REGRESSION);
  const [isTraining, setIsTraining] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics>(MODEL_DATA[MLModelType.LOGISTIC_REGRESSION]);

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const model = e.target.value as MLModelType;
    setSelectedModel(model);
    setIsTraining(true);
    setTimeout(() => {
      setCurrentMetrics(MODEL_DATA[model]);
      setIsTraining(false);
    }, 800);
  };

  return (
    <div className="pb-20 space-y-12">
      <header className="border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white flex items-center gap-4">
          <span className="bg-indigo-600 px-4 py-1 rounded-xl text-lg shadow-2xl shadow-indigo-900/40">Lab</span>
          Medical Case Study
        </h1>
        <p className="text-slate-400 mt-4 text-xl font-light">
          A high-fidelity simulation of heart disease detection using the end-to-end Machine Learning lifecycle.
        </p>
      </header>

      <div className="flex bg-slate-900/50 p-1 rounded-2xl border border-slate-800 max-w-2xl">
         {[
           { id: 'eda', icon: <BarChart3 size={18} />, label: 'EDA' },
           { id: 'performance', icon: <TrendingUp size={18} />, label: 'Benchmarks' },
           { id: 'code', icon: <Code size={18} />, label: 'Notebook' }
         ].map(tab => (
           <button 
             key={tab.id}
             onClick={() => setActiveTab(tab.id as any)}
             className={`flex-1 py-3 text-sm font-bold flex items-center justify-center gap-2 rounded-xl transition-all ${activeTab === tab.id ? 'bg-indigo-600 text-white shadow-xl shadow-indigo-900/30' : 'text-slate-500 hover:text-slate-300'}`}
           >
             {tab.icon} {tab.label}
           </button>
         ))}
      </div>

      {activeTab === 'performance' && (
        <div className="space-y-12 animate-fade-in">
          <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 flex flex-col md:flex-row gap-12 items-center justify-between shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl -mr-32 -mt-32"></div>
            <div className="w-full md:w-1/3 z-10">
              <label className="block text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-4">Select Algorithm</label>
              <select value={selectedModel} onChange={handleModelChange} disabled={isTraining} className="w-full bg-slate-950 text-white border-2 border-slate-800 rounded-2xl p-4 focus:ring-4 focus:ring-indigo-500/10 font-black text-sm cursor-pointer hover:border-slate-700 transition-all">
                {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="grid grid-cols-3 gap-4 flex-1 w-full z-10">
              {[
                { label: 'Accuracy', key: 'accuracy', color: 'text-emerald-400' },
                { label: 'Precision', key: 'precision', color: 'text-indigo-400' },
                { label: 'Recall', key: 'recall', color: 'text-fuchsia-400' }
              ].map((metric) => (
                <div key={metric.label} className="bg-slate-950 p-6 rounded-2xl border border-slate-800 text-center shadow-inner group hover:scale-105 transition-all">
                  <div className="text-[9px] text-slate-600 uppercase font-black tracking-widest mb-1">{metric.label}</div>
                  <div className={`text-3xl font-mono font-black ${isTraining ? 'text-slate-800 animate-pulse' : metric.color}`}>
                    {isTraining ? '---' : `${currentMetrics[metric.key as keyof ModelMetrics]}%`}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <div className="bg-slate-900 p-10 rounded-3xl border border-slate-800 shadow-xl">
              <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-8">Confusion Matrix Heatmap</h4>
              <div className="h-72"><ConfusionMatrixHeatmap matrix={currentMetrics.confusionMatrix} isTraining={isTraining} /></div>
            </div>
            <div className="bg-slate-900 p-10 rounded-3xl border border-slate-800 shadow-xl">
              <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-8">ROC Probability Curve</h4>
              <div className="h-72">
                {isTraining ? <div className="h-full flex items-center justify-center text-indigo-400"><Loader2 className="animate-spin w-12 h-12" /></div> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentMetrics.rocCurve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={10} />
                      <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', borderRadius: '12px' }} />
                      <Line type="monotone" dataKey="tpr" stroke="#f472b6" strokeWidth={4} dot={{ r: 6, fill: '#f472b6' }} activeDot={{ r: 8 }} animationDuration={1000} />
                      <Line dataKey="fpr" stroke="#334155" strokeDasharray="5 5" dot={false} strokeWidth={1} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'eda' && (
        <div className="space-y-12 animate-fade-in">
           <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              <div className="bg-slate-900 p-10 rounded-3xl border border-slate-800 shadow-xl">
                 <h3 className="text-xl font-bold text-white mb-8 flex items-center gap-3"><Database className="text-indigo-400" /> Feature Correlation Matrix</h3>
                 <CorrelationHeatmap />
              </div>
              <div className="bg-slate-900 p-10 rounded-3xl border border-slate-800 shadow-xl flex flex-col justify-center">
                 <h3 className="text-xl font-bold text-white mb-6">EDA Insights</h3>
                 <div className="space-y-6">
                    <div className="p-4 bg-slate-950 rounded-xl border-l-4 border-emerald-500">
                        <span className="text-xs font-black text-emerald-500 uppercase">Strong Interaction</span>
                        <p className="text-sm text-slate-400 mt-1">Target shows a high positive correlation (0.4) with <strong className="text-slate-200">Blood Pressure</strong>, indicating it is a primary risk predictor.</p>
                    </div>
                    <div className="p-4 bg-slate-950 rounded-xl border-l-4 border-rose-500">
                        <span className="text-xs font-black text-rose-500 uppercase">Inverse Relationship</span>
                        <p className="text-sm text-slate-400 mt-1"><strong className="text-slate-200">Max Heart Rate</strong> is inversely correlated (-0.4) with age and risk, suggesting higher fitness levels mitigate heart disease risk.</p>
                    </div>
                 </div>
              </div>
           </div>
        </div>
      )}

      {activeTab === 'code' && (
        <div className="animate-fade-in space-y-6">
          <div className="bg-slate-900 rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
            <div className="bg-slate-850 px-8 py-4 flex items-center justify-between border-b border-slate-800">
              <span className="text-xs font-black text-slate-400 uppercase tracking-widest">Medical_Classifier_v1.py</span>
              <div className="flex gap-2"><div className="w-3 h-3 rounded-full bg-rose-500"></div><div className="w-3 h-3 rounded-full bg-amber-500"></div><div className="w-3 h-3 rounded-full bg-emerald-500"></div></div>
            </div>
            <CodeBlock code={`import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom xgboost import XGBClassifier\n\n# 1. Load medical telemetry data\ndf = pd.read_csv('heart_vitals.csv')\n\n# 2. Split features and targets\nX = df.drop('risk_score', axis=1)\ny = df['risk_score']\n\n# 3. Scale biometric features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# 4. Train high-performance ensemble\nmodel = XGBClassifier(n_estimators=1000, learning_rate=0.01)\nmodel.fit(X_scaled, y)`} />
          </div>
        </div>
      )}
    </div>
  );
};