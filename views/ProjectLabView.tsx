import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, Area } from 'recharts';
import { MLModelType, ModelMetrics } from '../types';
import { Loader2, Code, Activity, Database, Info, ShieldCheck, AlertCircle } from 'lucide-react';
import { CodeBlock } from '../components/CodeBlock';

const MODEL_DATA: Record<MLModelType, ModelMetrics> = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    accuracy: 85.4,
    precision: 82.1,
    recall: 78.5,
    confusionMatrix: [
      { name: 'TP', value: 120 }, { name: 'FP', value: 30 },
      { name: 'TN', value: 145 }, { name: 'FN', value: 25 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.1, tpr: 0.6 }, { fpr: 0.3, tpr: 0.8 }, { fpr: 0.5, tpr: 0.88 }, { fpr: 1, tpr: 1 }
    ]
  },
  [MLModelType.RANDOM_FOREST]: {
    accuracy: 94.2,
    precision: 93.5,
    recall: 91.0,
    confusionMatrix: [
      { name: 'TP', value: 140 }, { name: 'FP', value: 10 },
      { name: 'TN', value: 160 }, { name: 'FN', value: 10 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.05, tpr: 0.8 }, { fpr: 0.1, tpr: 0.92 }, { fpr: 0.2, tpr: 0.96 }, { fpr: 1, tpr: 1 }
    ]
  },
  [MLModelType.SVM]: {
    accuracy: 88.9,
    precision: 86.4,
    recall: 84.2,
    confusionMatrix: [
      { name: 'TP', value: 130 }, { name: 'FP', value: 20 },
      { name: 'TN', value: 150 }, { name: 'FN', value: 20 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.15, tpr: 0.7 }, { fpr: 0.25, tpr: 0.85 }, { fpr: 0.4, tpr: 0.9 }, { fpr: 1, tpr: 1 }
    ]
  },
  [MLModelType.KNN]: {
    accuracy: 86.1,
    precision: 83.0,
    recall: 81.5,
    confusionMatrix: [
      { name: 'TP', value: 125 }, { name: 'FP', value: 28 },
      { name: 'TN', value: 147 }, { name: 'FN', value: 22 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.12, tpr: 0.65 }, { fpr: 0.28, tpr: 0.82 }, { fpr: 0.45, tpr: 0.89 }, { fpr: 1, tpr: 1 }
    ]
  },
  [MLModelType.XGBOOST]: {
    accuracy: 96.5,
    precision: 95.8,
    recall: 94.2,
    confusionMatrix: [
      { name: 'TP', value: 145 }, { name: 'FP', value: 5 },
      { name: 'TN', value: 165 }, { name: 'FN', value: 8 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.02, tpr: 0.85 }, { fpr: 0.08, tpr: 0.95 }, { fpr: 0.15, tpr: 0.98 }, { fpr: 1, tpr: 1 }
    ]
  }
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
    <div className={`relative flex flex-col items-center justify-center p-4 rounded-lg border border-white/5 transition-all hover:scale-[1.02] ${bgClass}`}>
       <div className={`absolute inset-0 rounded-lg opacity-20 ${colorClass}`} style={{ opacity: (value / max) * 0.4 }}></div>
       <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter mb-1 z-10">{label}</span>
       <span className="text-2xl font-mono font-bold text-white z-10">{value}</span>
       <span className="text-[9px] text-slate-400 font-mono mt-1 z-10 italic">{sub}</span>
    </div>
  );

  return (
    <div className="grid grid-cols-2 gap-3 h-full">
      <Cell label="True Positive" value={data.tp} sub="Correctly Identified" colorClass="bg-emerald-500" bgClass="bg-emerald-500/5" />
      <Cell label="False Positive" value={data.fp} sub="Type I Error" colorClass="bg-rose-500" bgClass="bg-rose-500/5" />
      <Cell label="False Negative" value={data.fn} sub="Type II Error" colorClass="bg-rose-500" bgClass="bg-rose-500/5" />
      <Cell label="True Negative" value={data.tn} sub="Correctly Rejected" colorClass="bg-emerald-500" bgClass="bg-emerald-500/5" />
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
    <div className="pb-10">
      <header className="border-b border-slate-800 pb-6 mb-6">
        <h1 className="text-4xl font-serif font-bold text-white flex items-center gap-3">
          <span className="bg-indigo-600 px-3 py-1 rounded text-lg shadow-lg shadow-indigo-900/40">Case Study</span>
          Heart Disease Classification
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Analyzing a medical dataset using the full Machine Learning lifecycle.
        </p>
      </header>

      <div className="flex border-b border-slate-800 mb-8 overflow-x-auto no-scrollbar">
         {[
           { id: 'eda', icon: <Database size={16} />, label: 'Data Analysis' },
           { id: 'code', icon: <Code size={16} />, label: 'Implementation' },
           { id: 'performance', icon: <Activity size={16} />, label: 'Model Benchmarks' }
         ].map(tab => (
           <button 
             key={tab.id}
             onClick={() => setActiveTab(tab.id as any)}
             className={`px-8 py-4 text-sm font-bold flex items-center gap-2 border-b-2 transition-all whitespace-nowrap ${activeTab === tab.id ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
           >
             {tab.icon} {tab.label}
           </button>
         ))}
      </div>

      {activeTab === 'performance' && (
        <div className="space-y-8 animate-fade-in">
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row gap-8 items-center justify-between shadow-2xl">
            <div className="w-full md:w-1/3">
              <label className="block text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-3">Model Selection</label>
              <div className="relative group">
                <select 
                  value={selectedModel}
                  onChange={handleModelChange}
                  disabled={isTraining}
                  className="w-full bg-slate-950 text-white border border-slate-700 rounded-xl p-4 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none cursor-pointer transition-all hover:border-slate-500 font-bold text-sm"
                >
                  {Object.values(MLModelType).map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 transition-transform group-hover:translate-y-[-40%]">▲</div>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 transition-transform group-hover:translate-y-[-10%]">▼</div>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4 flex-1 w-full">
              {[
                { label: 'Accuracy', key: 'accuracy', color: 'text-emerald-400' },
                { label: 'Precision', key: 'precision', color: 'text-indigo-400' },
                { label: 'Recall', key: 'recall', color: 'text-fuchsia-400' }
              ].map((metric) => (
                <div key={metric.label} className="bg-slate-950/50 p-5 rounded-2xl border border-slate-800/50 text-center shadow-inner group hover:border-slate-700 transition-colors">
                  <div className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">{metric.label}</div>
                  <div className={`text-3xl font-mono font-black ${isTraining ? 'text-slate-800 animate-pulse' : metric.color}`}>
                    {isTraining ? '--' : `${currentMetrics[metric.key as keyof ModelMetrics]}%`}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-xl">
              <div className="flex justify-between items-center mb-6">
                <h4 className="text-sm font-black text-slate-300 uppercase tracking-widest">Confusion Matrix</h4>
                <div className="flex gap-2">
                   <span className="flex items-center gap-1 text-[9px] text-slate-500"><div className="w-2 h-2 rounded bg-emerald-500/40"></div> Correct</span>
                   <span className="flex items-center gap-1 text-[9px] text-slate-500"><div className="w-2 h-2 rounded bg-rose-500/40"></div> Errors</span>
                </div>
              </div>
              <div className="h-64">
                <ConfusionMatrixHeatmap matrix={currentMetrics.confusionMatrix} isTraining={isTraining} />
              </div>
            </div>

            <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-xl">
              <h4 className="text-sm font-black text-slate-300 uppercase tracking-widest mb-6">Receiver Operating Characteristic</h4>
              <div className="h-64">
                {isTraining ? (
                  <div className="h-full flex items-center justify-center text-indigo-400"><Loader2 className="animate-spin w-8 h-8" /></div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentMetrics.rocCurve}>
                      <defs>
                        <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f472b6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#f472b6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={10} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#475569', fontSize: 9 }} />
                      <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#475569" fontSize={10} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 9 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', borderRadius: '8px' }} itemStyle={{ color: '#f472b6', fontSize: '12px' }} />
                      <Line type="monotone" dataKey="tpr" stroke="#f472b6" strokeWidth={4} dot={{ r: 4, fill: '#f472b6' }} activeDot={{ r: 6 }} animationDuration={1000} />
                      <Line dataKey="fpr" stroke="#334155" strokeDasharray="5 5" dot={false} strokeWidth={1} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          <div className="bg-indigo-900/10 border border-indigo-500/20 p-8 rounded-3xl relative overflow-hidden group">
             <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <ShieldCheck size={120} className="text-indigo-400" />
             </div>
             <h4 className="text-indigo-400 font-black text-sm uppercase tracking-widest mb-4 flex items-center gap-2">
                <Info size={18} /> Model Performance Insight
             </h4>
             <p className="text-slate-400 text-sm leading-relaxed max-w-3xl">
                Analysis for <strong>{selectedModel}</strong>:
                {selectedModel === MLModelType.LOGISTIC_REGRESSION && " This linear baseline provides high interpretability. It relies heavily on coefficients, making it easy to explain to medical staff why a patient was flagged (e.g., age and high cholesterol weightings)."}
                {selectedModel === MLModelType.RANDOM_FOREST && " The ensemble of trees captures non-linear interactions between symptoms. Its high precision ensures few healthy patients are wrongly given heart medication."}
                {selectedModel === MLModelType.SVM && " By maximizing the margin in a transformed feature space, it achieves strong generalization. It is particularly robust to outliers in blood pressure readings."}
                {selectedModel === MLModelType.KNN && " Classification is based on similarity to historical patients. This 'local' approach works well for clusters of patients with rare but similar symptom combinations."}
                {selectedModel === MLModelType.XGBOOST && " The gradient boosting approach iteratively corrects small errors. It is currently the top performer, capturing subtle risk patterns that simpler models miss."}
             </p>
          </div>
        </div>
      )}

      {activeTab === 'eda' && (
        <div className="space-y-8 animate-fade-in">
           <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-xl overflow-x-auto">
             <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                <Database size={20} className="text-indigo-400" /> Dataset Feature Samples
             </h3>
             <table className="w-full text-sm text-left">
               <thead className="text-[10px] text-slate-500 uppercase tracking-widest bg-slate-950">
                 <tr>
                   <th className="px-6 py-4 border-b border-slate-800">Age</th>
                   <th className="px-6 py-4 border-b border-slate-800">Chest Pain (CP)</th>
                   <th className="px-6 py-4 border-b border-slate-800">Cholesterol</th>
                   <th className="px-6 py-4 border-b border-slate-800">Max Heart Rate</th>
                   <th className="px-6 py-4 border-b border-slate-800 text-right">Target</th>
                 </tr>
               </thead>
               <tbody className="divide-y divide-slate-800">
                 {[
                   { age: 63, cp: 3, chol: 233, thalach: 150, target: 1 },
                   { age: 37, cp: 2, chol: 250, thalach: 187, target: 0 },
                   { age: 41, cp: 1, chol: 204, thalach: 172, target: 1 },
                   { age: 56, cp: 1, chol: 236, thalach: 178, target: 1 },
                 ].map((row, i) => (
                   <tr key={i} className="hover:bg-indigo-500/5 transition-colors group">
                     <td className="px-6 py-4 font-mono text-slate-300">{row.age}</td>
                     <td className="px-6 py-4 font-mono text-slate-300">{row.cp}</td>
                     <td className="px-6 py-4 font-mono text-slate-300">{row.chol}</td>
                     <td className="px-6 py-4 font-mono text-slate-300">{row.thalach}</td>
                     <td className={`px-6 py-4 text-right font-black ${row.target === 1 ? 'text-rose-500' : 'text-emerald-500'}`}>
                        {row.target === 1 ? 'POSITIVE' : 'NEGATIVE'}
                     </td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
        </div>
      )}

      {activeTab === 'code' && (
        <div className="animate-fade-in space-y-6">
          <div className="bg-slate-900 rounded-3xl border border-slate-800 overflow-hidden">
            <div className="bg-slate-800 px-6 py-3 flex items-center justify-between">
              <span className="text-xs font-black text-slate-400 uppercase tracking-widest">scikit-learn implementation</span>
              <div className="flex gap-1.5">
                 <div className="w-2.5 h-2.5 rounded-full bg-rose-500"></div>
                 <div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div>
                 <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
              </div>
            </div>
            <CodeBlock code={`import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv('heart_disease.csv')

# 2. Features/Target Split
X = df.drop('target', axis=1)
y = df['target']

# 3. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize XGBoost
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
accuracy = model.score(X_test_scaled, y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")`} />
          </div>
        </div>
      )}
    </div>
  );
};