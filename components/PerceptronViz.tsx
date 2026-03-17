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
    <div className="flex flex-col md:flex-row gap-8 bg-app border border-border-strong p-8 relative overflow-hidden">
      <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle, #1e293b 1px, transparent 1px)', backgroundSize: '30px 30px' }} />
      
      {/* Controls */}
      <div className="w-full md:w-1/3 flex flex-col gap-8 relative z-10">
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">WEIGHT_1 (X)</label>
                <span className="text-[10px] font-mono font-black text-brand">{w1.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-2" max="2" step="0.1" 
                value={w1} onChange={(e) => setW1(parseFloat(e.target.value))}
                className="w-full h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand"
            />
        </div>
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">WEIGHT_2 (Y)</label>
                <span className="text-[10px] font-mono font-black text-brand">{w2.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-2" max="2" step="0.1" 
                value={w2} onChange={(e) => setW2(parseFloat(e.target.value))}
                className="w-full h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand"
            />
        </div>
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">BIAS_OFFSET</label>
                <span className="text-[10px] font-mono font-black text-brand">{bias.toFixed(2)}</span>
            </div>
            <input 
                type="range" min="-10" max="10" step="0.5" 
                value={bias} onChange={(e) => setBias(parseFloat(e.target.value))}
                className="w-full h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand"
            />
        </div>

        <div className="pt-6 border-t border-border-strong flex gap-3">
            <button 
                onClick={() => setIsAnimating(!isAnimating)}
                className={`flex-1 py-2 border font-mono text-[10px] font-black uppercase tracking-[0.2em] flex items-center justify-center gap-2 transition-all ${isAnimating ? 'bg-rose-500/10 text-rose-500 border-rose-500/50' : 'bg-surface text-text-muted border-border-strong hover:text-brand hover:border-brand'}`}
            >
                <Play size={12} className={isAnimating ? "animate-pulse" : ""} />
                {isAnimating ? "HALT_TRAIN" : "AUTO_OPTIMIZE"}
            </button>
            <button 
                onClick={() => { setData(generateData()); setW1(0.5); setW2(0.5); setBias(-5); setIsAnimating(false); }}
                className="p-2 bg-surface border border-border-strong text-text-muted hover:text-brand transition-all"
                title="RESET_SYSTEM"
            >
                <RefreshCw size={12} />
            </button>
        </div>

        <div className="bg-surface p-4 border border-border-strong text-center">
            <div className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-2">SYSTEM_ACCURACY</div>
            <div className={`text-3xl font-mono font-black ${accuracy === 100 ? 'text-brand' : 'text-text-primary'}`}>
                {accuracy.toFixed(1)}%
            </div>
        </div>
      </div>

      {/* Visualization */}
      <div className="flex-1 h-80 bg-surface/30 border border-border-strong relative overflow-hidden z-10">
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-strong)" />
                <XAxis type="number" dataKey="x" domain={[0, 10]} hide />
                <YAxis type="number" dataKey="y" domain={[0, 10]} hide />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', fontSize: '9px', fontFamily: 'monospace', fontWeight: '900', borderRadius: '0' }} />
                
                {/* Decision Boundary */}
                <ReferenceLine segment={[lineData[0], lineData[1]]} stroke="var(--brand)" strokeWidth={2} strokeDasharray="4 4" />
                
                {/* Data Points */}
                <Scatter data={data}>
                    {data.map((entry, index) => (
                        <Cell 
                            key={`cell-${index}`} 
                            fill={entry.class === 1 ? '#f43f5e' : 'var(--brand)'} 
                            stroke={classify(entry.x, entry.y) === entry.class ? 'none' : '#fbbf24'}
                            strokeWidth={2}
                        />
                    ))}
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
        
        {/* Legend overlay */}
        <div className="absolute top-4 right-4 flex flex-col gap-2 text-[8px] font-mono font-black bg-surface border border-border-strong p-3 uppercase tracking-widest">
            <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#f43f5e]"></div> CLASS_ALPHA
            </div>
            <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-brand"></div> CLASS_BETA
            </div>
            <div className="flex items-center gap-2">
                <div className="w-4 h-[1px] bg-brand border-t border-dashed border-brand/50"></div> BOUNDARY
            </div>
        </div>
      </div>
    </div>
  );
};
