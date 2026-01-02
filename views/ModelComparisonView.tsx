
import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceDot, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, Tooltip } from 'recharts';
import { MLModelType } from '../types';
import { Swords, Zap } from 'lucide-react';

const comparisonData = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    type: 'Linear',
    interpretability: 'High',
    trainingSpeed: 'Very Fast',
    overfittingRisk: 'Low',
    dataScale: 'Small/Med',
    metrics: [
      { subject: 'Speed', A: 100 },
      { subject: 'Accuracy', A: 70 },
      { subject: 'Complexity', A: 20 },
      { subject: 'Interpretability', A: 95 },
      { subject: 'Scalability', A: 85 },
    ]
  },
  [MLModelType.RANDOM_FOREST]: {
    type: 'Ensemble',
    interpretability: 'Medium',
    trainingSpeed: 'Medium',
    overfittingRisk: 'Low',
    dataScale: 'Large',
    metrics: [
      { subject: 'Speed', A: 60 },
      { subject: 'Accuracy', A: 90 },
      { subject: 'Complexity', A: 75 },
      { subject: 'Interpretability', A: 50 },
      { subject: 'Scalability', A: 70 },
    ]
  },
  [MLModelType.SVM]: {
    type: 'Geometric',
    interpretability: 'Low',
    trainingSpeed: 'Slow',
    overfittingRisk: 'Medium',
    dataScale: 'Small/Med',
    metrics: [
      { subject: 'Speed', A: 30 },
      { subject: 'Accuracy', A: 85 },
      { subject: 'Complexity', A: 90 },
      { subject: 'Interpretability', A: 40 },
      { subject: 'Scalability', A: 50 },
    ]
  },
  [MLModelType.KNN]: {
    type: 'Instance',
    interpretability: 'High',
    trainingSpeed: 'Instant',
    overfittingRisk: 'Medium',
    dataScale: 'Small',
    metrics: [
      { subject: 'Speed', A: 90 },
      { subject: 'Accuracy', A: 65 },
      { subject: 'Complexity', A: 10 },
      { subject: 'Interpretability', A: 90 },
      { subject: 'Scalability', A: 30 },
    ]
  },
  [MLModelType.XGBOOST]: {
    type: 'Boosting',
    interpretability: 'Low',
    trainingSpeed: 'Fast',
    overfittingRisk: 'High',
    dataScale: 'Very Large',
    metrics: [
      { subject: 'Speed', A: 80 },
      { subject: 'Accuracy', A: 98 },
      { subject: 'Complexity', A: 95 },
      { subject: 'Interpretability', A: 30 },
      { subject: 'Scalability', A: 100 },
    ]
  }
};

const MODEL_IMPACTS: Record<MLModelType, any[]> = {
  [MLModelType.LOGISTIC_REGRESSION]: [
    { paramName: 'C (Inverse Reg)', labelX: 'C Value', data: [{ x: '0.01', y: 0.78 }, { x: '0.1', y: 0.86 }, { x: '1.0', y: 0.85 }, { x: '10', y: 0.83 }] }
  ],
  [MLModelType.RANDOM_FOREST]: [
    { paramName: 'n_estimators', labelX: 'No. Trees', data: [{ x: 10, y: 0.78 }, { x: 100, y: 0.91 }, { x: 500, y: 0.925 }] }
  ],
  [MLModelType.SVM]: [
    { paramName: 'C (Regularization)', labelX: 'C Value', data: [{ x: '0.1', y: 0.75 }, { x: '1', y: 0.86 }, { x: '10', y: 0.84 }] }
  ],
  [MLModelType.KNN]: [
    { paramName: 'n_neighbors', labelX: 'k', data: [{ x: 1, y: 0.74 }, { x: 5, y: 0.85 }, { x: 20, y: 0.79 }] }
  ],
  [MLModelType.XGBOOST]: [
    { paramName: 'learning_rate', labelX: 'Alpha', data: [{ x: 0.01, y: 0.82 }, { x: 0.1, y: 0.93 }, { x: 0.5, y: 0.86 }] }
  ]
};

const SensitivityPanel = ({ model, impacts, color }: any) => {
  const [sliderIndex, setSliderIndex] = useState(0);
  const currentParam = impacts[0];
  const currentDataPoint = currentParam.data[sliderIndex];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden shadow-xl">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-800/20">
         <span className="text-xs font-black uppercase tracking-widest text-slate-400">{model} Sensitivity</span>
         <span className="text-[10px] bg-slate-950 px-2 py-1 rounded text-slate-500 font-mono">Tuning {currentParam.paramName}</span>
      </div>
      <div className="p-6">
        <div className="h-40 relative">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={currentParam.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="x" hide />
              <YAxis hide domain={[0, 1]} />
              <Line type="monotone" dataKey="y" stroke={color} strokeWidth={3} dot={{ r: 4, fill: color }} />
              <ReferenceDot x={currentDataPoint.x} y={currentDataPoint.y} r={6} fill="#fff" stroke={color} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 space-y-4">
           <input type="range" min={0} max={currentParam.data.length - 1} value={sliderIndex} onChange={(e) => setSliderIndex(Number(e.target.value))} className="w-full" style={{ accentColor: color }} />
           <div className="flex justify-between text-[10px] font-black font-mono">
              <span className="text-slate-500">PARAM: <span className="text-white">{currentDataPoint.x}</span></span>
              <span className="text-slate-500">ACCURACY: <span style={{ color }}>{(currentDataPoint.y * 100).toFixed(1)}%</span></span>
           </div>
        </div>
      </div>
    </div>
  );
};

export const ModelComparisonView: React.FC = () => {
  const [modelA, setModelA] = useState<MLModelType>(MLModelType.LOGISTIC_REGRESSION);
  const [modelB, setModelB] = useState<MLModelType>(MLModelType.XGBOOST);

  // useMemo hook used to calculate radarData based on selected models.
  const radarData = useMemo(() => {
    const metricsA = (comparisonData[modelA] as any).metrics;
    const metricsB = (comparisonData[modelB] as any).metrics;
    return metricsA.map((m: any, i: number) => ({
      subject: m.subject,
      A: m.A,
      B: metricsB[i].A
    }));
  }, [modelA, modelB]);

  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header>
        <h1 className="text-5xl font-serif font-bold text-white mb-4 flex items-center gap-4">
          <Swords className="text-rose-500" size={48} />
          The Battleground
        </h1>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed">
          Comparing algorithms side-by-side to understand the fundamental trade-offs in modern machine learning.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-1 bg-gradient-to-br from-indigo-500/20 to-transparent rounded-3xl">
           <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-[22px] border border-white/5">
             <label className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-2 block">Challenger Alpha</label>
             <select value={modelA} onChange={(e) => setModelA(e.target.value as MLModelType)} className="w-full bg-slate-950 text-white font-bold p-4 rounded-xl border border-slate-800 outline-none focus:ring-2 focus:ring-indigo-500/50">
                {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
             </select>
           </div>
        </div>
        <div className="p-1 bg-gradient-to-br from-emerald-500/20 to-transparent rounded-3xl">
           <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-[22px] border border-white/5">
             <label className="text-[10px] font-black text-emerald-400 uppercase tracking-widest mb-2 block">Challenger Beta</label>
             <select value={modelB} onChange={(e) => setModelB(e.target.value as MLModelType)} className="w-full bg-slate-950 text-white font-bold p-4 rounded-xl border border-slate-800 outline-none focus:ring-2 focus:ring-emerald-500/50">
                {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
             </select>
           </div>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-10 shadow-2xl overflow-hidden relative">
         <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-rose-500 to-emerald-500 opacity-30"></div>
         <div className="flex flex-col lg:flex-row items-center gap-12">
            <div className="flex-1 space-y-8 w-full">
               <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                  <Zap className="text-amber-400" /> Statistical Footprint
               </h3>
               <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {[
                    { label: 'Interpretability', key: 'interpretability' },
                    { label: 'Training Speed', key: 'trainingSpeed' },
                    { label: 'Risk of Overfit', key: 'overfittingRisk' },
                    { label: 'Data Appetite', key: 'dataScale' }
                  ].map(row => (
                    <div key={row.key} className="p-4 bg-slate-950 rounded-2xl border border-slate-800/50">
                       <span className="text-[9px] font-black text-slate-500 uppercase tracking-[0.2em] mb-3 block">{row.label}</span>
                       <div className="flex items-center justify-between text-xs font-bold">
                          <span className="text-indigo-400">{(comparisonData[modelA] as any)[row.key]}</span>
                          <span className="text-slate-700 text-[10px]">vs</span>
                          <span className="text-emerald-400">{(comparisonData[modelB] as any)[row.key]}</span>
                       </div>
                    </div>
                  ))}
               </div>
            </div>
            <div className="w-full lg:w-[450px] h-[350px] bg-slate-950 rounded-3xl border border-slate-800/50 p-4 shadow-inner">
               <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#1e293b" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#475569', fontSize: 10, fontWeight: 'bold' }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name={modelA} dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                    <Radar name={modelB} dataKey="B" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                    <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', color: '#fff' }} />
                    <Legend />
                  </RadarChart>
               </ResponsiveContainer>
            </div>
         </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
         <SensitivityPanel model={modelA} impacts={MODEL_IMPACTS[modelA]} color="#6366f1" />
         <SensitivityPanel model={modelB} impacts={MODEL_IMPACTS[modelB]} color="#10b981" />
      </div>
    </div>
  );
};
