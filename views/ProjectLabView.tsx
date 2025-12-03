import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { MLModelType, ModelMetrics } from '../types';
import { Loader2, Code, Activity, Database, ArrowRight } from 'lucide-react';
import { CodeBlock } from '../components/CodeBlock';

// Mock Data for the simulation
const MODEL_DATA: Record<MLModelType, ModelMetrics> = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    accuracy: 85.4,
    precision: 82.1,
    recall: 78.5,
    confusionMatrix: [
      { name: 'True Pos', value: 120 },
      { name: 'False Pos', value: 30 },
      { name: 'True Neg', value: 145 },
      { name: 'False Neg', value: 25 },
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
      { name: 'True Pos', value: 140 },
      { name: 'False Pos', value: 10 },
      { name: 'True Neg', value: 160 },
      { name: 'False Neg', value: 10 },
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
      { name: 'True Pos', value: 130 },
      { name: 'False Pos', value: 20 },
      { name: 'True Neg', value: 150 },
      { name: 'False Neg', value: 20 },
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
      { name: 'True Pos', value: 125 },
      { name: 'False Pos', value: 28 },
      { name: 'True Neg', value: 147 },
      { name: 'False Neg', value: 22 },
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
      { name: 'True Pos', value: 145 },
      { name: 'False Pos', value: 5 },
      { name: 'True Neg', value: 165 },
      { name: 'False Neg', value: 8 },
    ],
    rocCurve: [
      { fpr: 0, tpr: 0 }, { fpr: 0.02, tpr: 0.85 }, { fpr: 0.08, tpr: 0.95 }, { fpr: 0.15, tpr: 0.98 }, { fpr: 1, tpr: 1 }
    ]
  }
};

// EDA Data
const AGE_DISTRIBUTION = [
  { range: '20-30', count: 15 },
  { range: '30-40', count: 45 },
  { range: '40-50', count: 85 },
  { range: '50-60', count: 120 },
  { range: '60-70', count: 95 },
  { range: '70+', count: 40 },
];
const TARGET_BALANCE = [
  { name: 'Heart Disease', value: 45 },
  { name: 'Healthy', value: 55 },
];
const COLORS = ['#ef4444', '#10b981'];

const PipelineViz = () => (
  <div className="flex items-center justify-between w-full text-xs font-mono text-slate-400 py-6 overflow-x-auto select-none">
    <div className="flex flex-col items-center gap-2 min-w-[80px]">
      <div className="w-12 h-12 rounded-lg border border-slate-600 bg-slate-800 flex items-center justify-center text-slate-500 shadow-md">
        Raw
      </div>
      <span className="text-[10px] uppercase tracking-wider">Input</span>
    </div>
    
    <div className="flex-1 h-px bg-slate-700 mx-2 flex items-center justify-center relative">
        <div className="w-1 h-1 rounded-full bg-slate-600"></div>
    </div>

    <div className="flex flex-col items-center gap-2 min-w-[80px]">
       <div className="w-12 h-12 rounded-lg border border-indigo-500 bg-indigo-900/20 flex items-center justify-center text-indigo-400 font-bold shadow-[0_0_15px_rgba(99,102,241,0.2)]">
         NaN
       </div>
       <span className="text-[10px] uppercase tracking-wider text-indigo-400">Impute</span>
    </div>

    <div className="flex-1 h-px bg-slate-700 mx-2 flex items-center justify-center relative">
        <ArrowRight size={12} className="text-slate-600 absolute" />
    </div>

     <div className="flex flex-col items-center gap-2 min-w-[80px]">
       <div className="w-12 h-12 rounded-lg border border-emerald-500 bg-emerald-900/20 flex items-center justify-center text-emerald-400 font-bold shadow-[0_0_15px_rgba(16,185,129,0.2)]">
         0..1
       </div>
       <span className="text-[10px] uppercase tracking-wider text-emerald-400">Scale</span>
    </div>

    <div className="flex-1 h-px bg-slate-700 mx-2 flex items-center justify-center relative">
        <ArrowRight size={12} className="text-slate-600 absolute" />
    </div>

     <div className="flex flex-col items-center gap-2 min-w-[80px]">
       <div className="w-12 h-12 rounded-lg border border-fuchsia-500 bg-fuchsia-900/20 flex items-center justify-center text-fuchsia-400 font-bold shadow-[0_0_15px_rgba(217,70,239,0.2)]">
         0/1
       </div>
       <span className="text-[10px] uppercase tracking-wider text-fuchsia-400">Encode</span>
    </div>

    <div className="flex-1 h-px bg-slate-700 mx-2 flex items-center justify-center relative">
        <div className="w-1 h-1 rounded-full bg-slate-600"></div>
    </div>

    <div className="flex flex-col items-center gap-2 min-w-[80px]">
       <div className="w-12 h-12 rounded-lg border border-slate-600 bg-slate-800 flex items-center justify-center text-white shadow-md">
         Clean
       </div>
       <span className="text-[10px] uppercase tracking-wider">Output</span>
    </div>
  </div>
);

export const ProjectLabView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'eda' | 'code' | 'performance'>('performance');
  const [selectedModel, setSelectedModel] = useState<MLModelType>(MLModelType.LOGISTIC_REGRESSION);
  const [isTraining, setIsTraining] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics>(MODEL_DATA[MLModelType.LOGISTIC_REGRESSION]);

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const model = e.target.value as MLModelType;
    setSelectedModel(model);
    setIsTraining(true);
    // Simulate training delay
    setTimeout(() => {
      setCurrentMetrics(MODEL_DATA[model]);
      setIsTraining(false);
    }, 1000);
  };

  const renderTabContent = () => {
    if (activeTab === 'eda') {
      return (
        <div className="space-y-6 animate-fade-in">
           <div className="bg-slate-900 p-6 rounded-lg border border-slate-800 shadow-md">
             <h3 className="text-xl font-bold text-indigo-400 mb-4">Dataset Summary (Cleveland Heart Disease)</h3>
             <div className="overflow-x-auto">
               <table className="w-full text-sm text-left text-slate-400">
                 <thead className="text-xs text-slate-200 uppercase bg-slate-800">
                   <tr>
                     <th className="px-4 py-3">Age</th>
                     <th className="px-4 py-3">Sex</th>
                     <th className="px-4 py-3">CP (Chest Pain)</th>
                     <th className="px-4 py-3">Trestbps</th>
                     <th className="px-4 py-3">Chol</th>
                     <th className="px-4 py-3">Target</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr className="border-b border-slate-800 hover:bg-slate-800/50">
                     <td className="px-4 py-3 font-mono">63</td>
                     <td className="px-4 py-3 font-mono">1</td>
                     <td className="px-4 py-3 font-mono">3</td>
                     <td className="px-4 py-3 font-mono">145</td>
                     <td className="px-4 py-3 font-mono">233</td>
                     <td className="px-4 py-3 text-red-400 font-bold font-mono">1</td>
                   </tr>
                   <tr className="border-b border-slate-800 hover:bg-slate-800/50">
                     <td className="px-4 py-3 font-mono">37</td>
                     <td className="px-4 py-3 font-mono">1</td>
                     <td className="px-4 py-3 font-mono">2</td>
                     <td className="px-4 py-3 font-mono">130</td>
                     <td className="px-4 py-3 font-mono">250</td>
                     <td className="px-4 py-3 text-green-400 font-bold font-mono">0</td>
                   </tr>
                   <tr>
                     <td className="px-4 py-3 italic text-slate-600" colSpan={6}>... 301 more rows</td>
                   </tr>
                 </tbody>
               </table>
             </div>
           </div>

           <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
             <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 shadow-md">
               <h4 className="text-sm font-bold text-slate-300 mb-4">Age Distribution</h4>
               <div className="h-64">
                 <ResponsiveContainer width="100%" height="100%">
                   <BarChart data={AGE_DISTRIBUTION}>
                     <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                     <XAxis dataKey="range" stroke="#94a3b8" fontSize={12} />
                     <YAxis stroke="#94a3b8" />
                     <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                     <Bar dataKey="count" fill="#818cf8" radius={[4, 4, 0, 0]} />
                   </BarChart>
                 </ResponsiveContainer>
               </div>
             </div>
             <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 shadow-md">
               <h4 className="text-sm font-bold text-slate-300 mb-4">Target Class Balance</h4>
               <div className="h-64 flex items-center justify-center">
                 <ResponsiveContainer width="100%" height="100%">
                   <PieChart>
                     <Pie data={TARGET_BALANCE} cx="50%" cy="50%" outerRadius={80} fill="#8884d8" dataKey="value" label>
                       {TARGET_BALANCE.map((entry, index) => (
                         <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                       ))}
                     </Pie>
                     <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                     <Legend />
                   </PieChart>
                 </ResponsiveContainer>
               </div>
             </div>
           </div>

           {/* Data Cleaning Pipeline Section */}
           <div className="bg-slate-900 p-6 rounded-lg border border-slate-800 shadow-md">
             <h3 className="text-xl font-bold text-indigo-400 mb-6">Data Cleaning Pipeline</h3>
             
             {/* Visual Pipeline */}
             <div className="mb-8 bg-slate-950 p-4 rounded-lg border border-slate-800">
               <PipelineViz />
             </div>

             <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <div>
                   <h4 className="text-lg font-bold text-slate-200 mb-4">Essential Preprocessing Steps</h4>
                   <p className="text-slate-400 mb-6 text-sm leading-relaxed">
                     Raw data is rarely ready for modeling. To ensure high performance and avoid biases, we apply a standard cleaning pipeline before feeding data into algorithms.
                   </p>
                   <ul className="space-y-4">
                     <li className="flex gap-4">
                        <div className="mt-0.5 w-6 h-6 rounded bg-indigo-500/20 text-indigo-400 flex flex-shrink-0 items-center justify-center font-bold text-xs border border-indigo-500/50">1</div>
                        <div>
                          <strong className="text-slate-200 block text-sm">Handling Missing Values (Imputation)</strong>
                          <span className="text-sm text-slate-400">Imputing nulls with mean/median or dropping incomplete rows ensures we don't lose valuable data or introduce errors.</span>
                        </div>
                     </li>
                     <li className="flex gap-4">
                        <div className="mt-0.5 w-6 h-6 rounded bg-emerald-500/20 text-emerald-400 flex flex-shrink-0 items-center justify-center font-bold text-xs border border-emerald-500/50">2</div>
                        <div>
                          <strong className="text-slate-200 block text-sm">Feature Scaling (Standardization)</strong>
                          <span className="text-sm text-slate-400">Standardizing features (e.g., Age vs. Cholesterol) prevents variables with larger magnitudes from dominating distance-based algorithms.</span>
                        </div>
                     </li>
                     <li className="flex gap-4">
                        <div className="mt-0.5 w-6 h-6 rounded bg-fuchsia-500/20 text-fuchsia-400 flex flex-shrink-0 items-center justify-center font-bold text-xs border border-fuchsia-500/50">3</div>
                        <div>
                          <strong className="text-slate-200 block text-sm">Categorical Encoding</strong>
                          <span className="text-sm text-slate-400">Converting text categories (like 'Chest Pain Type') into numeric vectors (One-Hot Encoding) so the model can process them.</span>
                        </div>
                     </li>
                   </ul>
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-500 mb-2 uppercase tracking-wider">Preprocessing Snippet</div>
                  <CodeBlock code={`# 1. Impute Missing Values
df['chol'] = df['chol'].fillna(df['chol'].mean())

# 2. Scale Numerical Features (Standardization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = ['age', 'trestbps', 'chol', 'thalach']
df[cols] = scaler.fit_transform(df[cols])

# 3. Categorical Encoding (One-Hot)
# Converts 'cp' (0,1,2,3) into 'cp_0', 'cp_1', ...
df = pd.get_dummies(df, columns=['cp', 'thal'])`} />
                </div>
             </div>
           </div>
        </div>
      );
    }

    if (activeTab === 'code') {
      return (
        <div className="animate-fade-in">
          <CodeBlock code={`import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv('heart.csv')

# 2. Preprocessing
# Encode categorical variables
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope'])

# Separate Features and Target
X = df.drop('target', axis=1)
y = df['target']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model Initialization & Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Evaluation
print(f"Accuracy: {model.score(X_test, y_test):.4f}")`} />
        </div>
      );
    }

    return (
      <div className="space-y-6 animate-fade-in">
        {/* Controls Area */}
        <div className="bg-slate-900 p-6 rounded-lg border border-slate-800 flex flex-col md:flex-row gap-6 items-start md:items-center justify-between shadow-md">
          <div className="flex-1 w-full">
            <label className="block text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2">Select Algorithm</label>
            <div className="relative">
              <select 
                value={selectedModel}
                onChange={handleModelChange}
                disabled={isTraining}
                className="w-full bg-slate-800 text-white border border-slate-700 rounded p-3 focus:outline-none focus:border-indigo-500 disabled:opacity-50 appearance-none cursor-pointer"
              >
                <option value={MLModelType.LOGISTIC_REGRESSION}>Logistic Regression (Baseline)</option>
                <option value={MLModelType.KNN}>K-Nearest Neighbors</option>
                <option value={MLModelType.SVM}>Support Vector Machine</option>
                <option value={MLModelType.RANDOM_FOREST}>Random Forest</option>
                <option value={MLModelType.XGBOOST}>XGBoost (Gradient Boosting)</option>
              </select>
              <div className="absolute right-4 top-4 pointer-events-none text-slate-400">▼</div>
            </div>
          </div>
          
          <div className="flex gap-4 w-full md:w-auto">
            {['Accuracy', 'Precision', 'Recall'].map((metric) => (
              <div key={metric} className="bg-slate-800 p-4 rounded border border-slate-700 flex-1 min-w-[100px] text-center shadow-sm">
                <div className="text-xs text-slate-400 uppercase">{metric}</div>
                <div className={`text-2xl font-bold ${isTraining ? 'text-slate-600' : 'text-emerald-400'}`}>
                  {isTraining ? '--' : (currentMetrics[metric.toLowerCase() as keyof ModelMetrics] as number)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 shadow-lg">
            <h4 className="text-sm font-bold text-slate-300 mb-4 border-b border-slate-800 pb-2">Confusion Matrix</h4>
            <div className="h-64">
              {isTraining ? (
                  <div className="h-full flex items-center justify-center text-indigo-400"><Loader2 className="animate-spin w-8 h-8" /></div>
              ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={currentMetrics.confusionMatrix}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{fontSize: 12}} />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
                      <Bar dataKey="value" fill="#818cf8" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 shadow-lg">
            <h4 className="text-sm font-bold text-slate-300 mb-4 border-b border-slate-800 pb-2">ROC Curve</h4>
            <div className="h-64">
              {isTraining ? (
                  <div className="h-full flex items-center justify-center text-indigo-400"><Loader2 className="animate-spin w-8 h-8" /></div>
              ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentMetrics.rocCurve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 12 }} stroke="#94a3b8" />
                      <YAxis dataKey="tpr" type="number" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
                      <Line type="monotone" dataKey="tpr" stroke="#f472b6" strokeWidth={3} dot={false} />
                      <Line dataKey="fpr" stroke="#475569" strokeDasharray="5 5" dot={false} strokeWidth={1} />
                    </LineChart>
                  </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Inference Logic Panel */}
        <div className="bg-indigo-900/20 border border-indigo-500/30 p-6 rounded-lg">
          <h4 className="text-indigo-400 font-bold mb-2 flex items-center gap-2">
            <Activity size={18} /> Model Inference Logic
          </h4>
          <p className="text-slate-300 text-sm leading-relaxed">
            The <strong>{selectedModel}</strong> model achieves {currentMetrics.accuracy}% accuracy. 
            {selectedModel === MLModelType.LOGISTIC_REGRESSION && " Being a linear model, it applies weights to features like 'thalach' (heart rate) and 'cp' (chest pain). High positive weights on these features increase the log-odds of the disease."}
            {selectedModel === MLModelType.RANDOM_FOREST && " It aggregates decisions from 100+ uncorrelated trees. It identifies non-linear interactions, noticing that high 'cholesterol' combined with 'age > 50' drastically increases risk."}
            {selectedModel === MLModelType.SVM && " It found a hyperplane in high-dimensional space maximizing the margin. The Radial Basis Function (RBF) kernel allowed it to capture the circular decision boundary inherent in the data cluster."}
            {selectedModel === MLModelType.KNN && " It classified patients purely based on similarity. A patient is flagged as 'Risk' if the majority of their 5 closest neighbors in the feature space also have heart disease."}
            {selectedModel === MLModelType.XGBOOST && " By sequentially correcting errors of previous trees, it focused heavily on 'hard-to-classify' edge cases, resulting in superior precision."}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="pb-10">
      <header className="border-b border-slate-800 pb-6 mb-6">
        <h1 className="text-3xl font-serif font-bold text-white flex items-center gap-3">
          <span className="bg-indigo-600 px-3 py-1 rounded text-lg">Case Study</span>
          Heart Disease Classification
        </h1>
        <p className="text-slate-400 mt-2">
          End-to-end machine learning pipeline analysis.
        </p>
      </header>

      {/* Tab Navigation */}
      <div className="flex border-b border-slate-800 mb-6 overflow-x-auto">
         <button 
           onClick={() => setActiveTab('eda')}
           className={`px-6 py-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap ${activeTab === 'eda' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-white'}`}
         >
           <Database size={16} /> Data Analysis
         </button>
         <button 
           onClick={() => setActiveTab('code')}
           className={`px-6 py-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap ${activeTab === 'code' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-white'}`}
         >
           <Code size={16} /> The Code
         </button>
         <button 
           onClick={() => setActiveTab('performance')}
           className={`px-6 py-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap ${activeTab === 'performance' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-white'}`}
         >
           <Activity size={16} /> Model Performance
         </button>
      </div>

      {renderTabContent()}
    </div>
  );
};