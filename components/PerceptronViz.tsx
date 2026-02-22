import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ReferenceLine, Cell, Tooltip } from 'recharts';
import { RefreshCw, Play } from 'lucide-react';

const generateData = () => {
  const data = [];
  // Class 0 (Bottom Left)
  for (let i = 0; i < 10; i++) {
    data.push({ x: Math.random() * 4 + 1, y: Math.random() * 4 + 1, class: 0 });
  }
  // Class 1 (Top Right)
  for (let i = 0; i < 10; i++) {
    data.push({ x: Math.random() * 4 + 5, y: Math.random() * 4 + 5, class: 1 });
  }
  return data;
};

export const PerceptronViz: React.FC = () => {
  const [data, setData] = useState(generateData());
  const [w1, setW1] = useState(0.5);
  const [w2, setW2] = useState(0.5);
  const [bias, setBias] = useState(-5);
  const [isAnimating, setIsAnimating] = useState(false);

  // Calculate decision boundary line: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
  const lineData = useMemo(() => {
    if (Math.abs(w2) < 0.01) return []; // Avoid division by zero
    const x1 = 0;
    const y1 = -(w1 * x1 + bias) / w2;
    const x2 = 10;
    const y2 = -(w1 * x2 + bias) / w2;
    return [{ x: x1, y: y1 }, { x: x2, y: y2 }];
  }, [w1, w2, bias]);

  const classify = (x: number, y: number) => {
    return w1 * x + w2 * y + bias > 0 ? 1 : 0;
  };

  const accuracy = useMemo(() => {
    const correct = data.filter(p => classify(p.x, p.y) === p.class).length;
    return (correct / data.length) * 100;
  }, [data, w1, w2, bias]);

  const trainStep = () => {
    // Simple Perceptron Learning Rule Step
    const learningRate = 0.1;
    let newW1 = w1;
    let newW2 = w2;
    let newBias = bias;
    let misclassified = false;

    // Pick a random point or iterate
    // For visualization, let's just do one pass over data or a random batch
    for (const point of data) {
        const prediction = newW1 * point.x + newW2 * point.y + newBias > 0 ? 1 : 0;
        const error = point.class - prediction;
        
        if (error !== 0) {
            newW1 += learningRate * error * point.x;
            newW2 += learningRate * error * point.y;
            newBias += learningRate * error;
            misclassified = true;
        }
    }
    
    setW1(newW1);
    setW2(newW2);
    setBias(newBias);
    return misclassified;
  };

  React.useEffect(() => {
      let interval: any;
      if (isAnimating) {
          interval = setInterval(() => {
              const changed = trainStep();
              if (!changed || accuracy === 100) {
                  setIsAnimating(false);
              }
          }, 100);
      }
      return () => clearInterval(interval);
  }, [isAnimating, accuracy, w1, w2, bias]); // Dependencies for closure

  return (
    <div className="flex flex-col md:flex-row gap-6 bg-slate-950 rounded-2xl border border-slate-800/50 p-6">
      {/* Controls */}
      <div className="w-full md:w-1/3 flex flex-col gap-6">
        <div className="space-y-4">
            <div className="flex justify-between items-center">
                <label className="text-xs font-mono text-slate-400 uppercase">Weight 1 (X)</label>
                <span className="text-xs font-mono text-indigo-400">{w1.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-2" max="2" step="0.1" 
                value={w1} onChange={(e) => setW1(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
        </div>
        <div className="space-y-4">
            <div className="flex justify-between items-center">
                <label className="text-xs font-mono text-slate-400 uppercase">Weight 2 (Y)</label>
                <span className="text-xs font-mono text-indigo-400">{w2.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-2" max="2" step="0.1" 
                value={w2} onChange={(e) => setW2(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
        </div>
        <div className="space-y-4">
            <div className="flex justify-between items-center">
                <label className="text-xs font-mono text-slate-400 uppercase">Bias</label>
                <span className="text-xs font-mono text-indigo-400">{bias.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-10" max="10" step="0.5" 
                value={bias} onChange={(e) => setBias(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
        </div>

        <div className="pt-4 border-t border-slate-800 flex gap-2">
            <button 
                onClick={() => setIsAnimating(!isAnimating)}
                className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-colors ${isAnimating ? 'bg-rose-500/10 text-rose-400 border border-rose-500/50' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-500/20'}`}
            >
                <Play size={14} className={isAnimating ? "animate-pulse" : ""} />
                {isAnimating ? "Stop Training" : "Auto-Train"}
            </button>
            <button 
                onClick={() => { setData(generateData()); setW1(0.5); setW2(0.5); setBias(-5); setIsAnimating(false); }}
                className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
                title="Reset Data"
            >
                <RefreshCw size={14} />
            </button>
        </div>

        <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 text-center">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Accuracy</div>
            <div className={`text-2xl font-mono font-bold ${accuracy === 100 ? 'text-emerald-400' : 'text-white'}`}>
                {accuracy.toFixed(1)}%
            </div>
        </div>
      </div>

      {/* Visualization */}
      <div className="flex-1 h-64 bg-slate-900/50 rounded-xl border border-slate-800 relative overflow-hidden">
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="x" domain={[0, 10]} hide />
                <YAxis type="number" dataKey="y" domain={[0, 10]} hide />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '10px' }} />
                
                {/* Decision Boundary */}
                <ReferenceLine segment={[lineData[0], lineData[1]]} stroke="#6366f1" strokeWidth={2} strokeDasharray="5 5" />
                
                {/* Data Points */}
                <Scatter data={data}>
                    {data.map((entry, index) => (
                        <Cell 
                            key={`cell-${index}`} 
                            fill={entry.class === 1 ? '#f472b6' : '#818cf8'} 
                            stroke={classify(entry.x, entry.y) === entry.class ? 'none' : '#ef4444'}
                            strokeWidth={2}
                        />
                    ))}
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
        
        {/* Legend overlay */}
        <div className="absolute top-2 right-2 flex flex-col gap-1 text-[9px] font-mono bg-slate-950/80 p-2 rounded border border-slate-800">
            <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#f472b6]"></div> Class 1
            </div>
            <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#818cf8]"></div> Class 0
            </div>
            <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-indigo-500 border-t border-dashed border-indigo-300"></div> Boundary
            </div>
        </div>
      </div>
    </div>
  );
};
