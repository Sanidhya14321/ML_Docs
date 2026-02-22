
import React, { useState, useEffect, useMemo } from 'react';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, ScatterChart, Scatter, ReferenceLine, LabelList } from 'recharts';
import { Database, ArrowRight, Search, FileText, Bot, Play, Pause } from 'lucide-react';
import { PerceptronViz } from '../components/PerceptronViz';
import { ActivationFunctionsViz } from '../components/ActivationFunctionsViz';

// Data for Embeddings Viz
const embeddingData = [
    { x: 2, y: 2, label: 'Man', fill: '#818cf8' },
    { x: 2, y: 6, label: 'Woman', fill: '#f472b6' },
    { x: 6, y: 2, label: 'King', fill: '#818cf8' },
    { x: 6, y: 6, label: 'Queen', fill: '#f472b6' }
];

const NeuralNetworkViz = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
        interval = setInterval(() => {
            setActiveStep(prev => (prev + 1) % 3);
        }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const width = 600;
  const height = 300;
  const layerX = [80, 240, 400, 540]; 
  
  const inputNodes = [80, 150, 220];
  const h1Nodes = [50, 100, 150, 200, 250];
  const h2Nodes = [70, 130, 190, 250];
  const outputNodes = [110, 190];

  const createConnections = (nodesLeft: number[], nodesRight: number[], xLeft: number, xRight: number) => {
    const paths = [];
    for (let i = 0; i < nodesLeft.length; i++) {
        for (let j = 0; j < nodesRight.length; j++) {
            const weight = 0.2 + Math.random() * 0.8;
            paths.push({
                d: `M ${xLeft} ${nodesLeft[i]} C ${xLeft + (xRight - xLeft)/2} ${nodesLeft[i]}, ${xLeft + (xRight - xLeft)/2} ${nodesRight[j]}, ${xRight} ${nodesRight[j]}`,
                weight,
                key: `${xLeft}-${i}-${j}`
            });
        }
    }
    return paths;
  };

  const c1 = useMemo(() => createConnections(inputNodes, h1Nodes, layerX[0], layerX[1]), []);
  const c2 = useMemo(() => createConnections(h1Nodes, h2Nodes, layerX[1], layerX[2]), []);
  const c3 = useMemo(() => createConnections(h2Nodes, outputNodes, layerX[2], layerX[3]), []);

  return (
    <div className="w-full h-80 bg-slate-950/50 rounded-3xl border border-slate-800 flex items-center justify-center relative overflow-hidden shadow-inner select-none backdrop-blur-sm group">
       <button 
         onClick={() => setIsPlaying(!isPlaying)}
         className="absolute top-4 right-4 z-20 p-2 rounded-lg bg-slate-800/80 text-slate-400 hover:text-white hover:bg-indigo-600 transition-all opacity-0 group-hover:opacity-100"
       >
         {isPlaying ? <Pause size={16} /> : <Play size={16} />}
       </button>

       <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="w-full h-full p-4">
          <defs>
            <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <linearGradient id="forwardGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#818cf8" />
            </linearGradient>
            <linearGradient id="backGrad" x1="100%" y1="0%" x2="0%" y2="0%">
                <stop offset="0%" stopColor="#f43f5e" />
                <stop offset="100%" stopColor="#fb7185" />
            </linearGradient>
          </defs>

          {/* Lines */}
          {[c1, c2, c3].map((layer, lIdx) => (
              <g key={lIdx}>
                  {layer.map((conn) => (
                      <path 
                        key={conn.key} 
                        d={conn.d} 
                        stroke="#1e293b" 
                        strokeWidth={conn.weight * 1.5} 
                        fill="none" 
                        className="transition-all duration-700"
                        strokeOpacity={activeStep === 0 ? 0.6 : 0.2}
                      />
                  ))}
                  {activeStep === 0 && layer.map((conn) => (
                      <path 
                        key={`${conn.key}-active`}
                        d={conn.d} 
                        stroke="url(#forwardGrad)" 
                        strokeWidth={conn.weight * 2} 
                        fill="none" 
                        strokeDasharray="5 10"
                        className="animate-[flow_1.5s_linear_infinite]"
                        strokeOpacity={0.5}
                      />
                  ))}
                  {activeStep === 2 && layer.map((conn) => (
                      <path 
                        key={`${conn.key}-back`}
                        d={conn.d} 
                        stroke="url(#backGrad)" 
                        strokeWidth={conn.weight * 1.5} 
                        fill="none" 
                        strokeDasharray="5 10"
                        className="animate-[flow_1.5s_linear_infinite]"
                        style={{ animationDirection: 'reverse' }}
                        strokeOpacity={0.4}
                      />
                  ))}
              </g>
          ))}

          {/* Nodes */}
          {[inputNodes, h1Nodes, h2Nodes, outputNodes].map((layer, lIdx) => (
              <g key={lIdx}>
                  {layer.map((y, nIdx) => (
                      <circle 
                        key={nIdx} 
                        cx={layerX[lIdx]} 
                        cy={y} 
                        r={lIdx === 0 || lIdx === 3 ? 9 : 7} 
                        fill="#020617" 
                        stroke={activeStep === 0 ? (lIdx === 0 ? '#6366f1' : '#1e293b') : activeStep === 2 ? '#f43f5e' : '#1e293b'} 
                        strokeWidth="2" 
                        filter={activeStep === 0 && lIdx === 0 ? "url(#nodeGlow)" : ""}
                        className="transition-colors duration-1000"
                      />
                  ))}
              </g>
          ))}

          {/* Legend */}
          <g transform="translate(20, 20)">
             <rect width="140" height="42" rx="10" fill="#020617" stroke="#1e293b" />
             <text x="14" y="26" fill="#94a3b8" fontSize="9" fontWeight="bold" className="font-mono uppercase tracking-widest">
                {activeStep === 0 ? "Forwarding" : activeStep === 1 ? "Error: 0.042" : "Backpropping"}
             </text>
             <circle cx="122" cy="23" r="4" fill={activeStep === 0 ? "#6366f1" : activeStep === 1 ? "#fbbf24" : "#f43f5e"} className="animate-pulse" />
          </g>
       </svg>
    </div>
  );
};

const ConvolutionViz = () => (
    <div className="flex items-center justify-center gap-4 py-8 select-none">
        <div className="grid grid-cols-4 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 16}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${[5,6,9,10].includes(i) ? 'bg-indigo-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
        <div className="text-slate-500 font-mono text-xl">×</div>
        <div className="grid grid-cols-2 gap-1 p-1 bg-indigo-900 border border-indigo-500">
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
        </div>
        <div className="text-slate-500 font-mono text-xl">=</div>
        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 9}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${i === 4 ? 'bg-emerald-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
    </div>
);

const AttentionViz = () => {
    const words = ['The', 'cat', 'sat', 'on'];
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    // Dynamic attention weights based on hover
    // In a real model, this is computed. Here we simulate the focus.
    const getWeights = (idx: number) => {
        const base = [0.1, 0.1, 0.1, 0.1];
        base[idx] = 0.7; // Self focus
        if (idx === 1) base[2] = 0.2; // Cat -> Sat
        if (idx === 2) base[1] = 0.2; // Sat -> Cat
        return base;
    };

    const currentWeights = hoverIndex !== null ? getWeights(hoverIndex) : [0.5, 0.3, 0.1, 0.1];

    return (
        <div className="flex flex-col items-center py-6 gap-6">
            <div className="flex gap-4">
                {words.map((word, i) => (
                    <div 
                        key={i} 
                        onMouseEnter={() => setHoverIndex(i)}
                        onMouseLeave={() => setHoverIndex(null)}
                        className={`
                            px-3 py-2 rounded-lg border font-mono text-sm cursor-default transition-all duration-300
                            ${hoverIndex === i ? 'bg-indigo-500 border-indigo-400 text-white scale-110' : 'bg-slate-900 border-slate-700 text-slate-400'}
                        `}
                    >
                        {word}
                    </div>
                ))}
            </div>
            
            <div className="flex flex-col items-center gap-2">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-[0.2em]">Context Weighting</div>
                <div className="flex gap-4 items-end h-24">
                    {currentWeights.map((w, i) => (
                        <div key={i} className="flex flex-col items-center gap-1">
                            <div 
                                className="w-10 bg-indigo-500/40 border border-indigo-500/60 rounded-t transition-all duration-500 shadow-[0_0_15px_rgba(99,102,241,0.2)]" 
                                style={{ height: `${w * 100}%` }}
                            ></div>
                            <span className="text-[8px] text-slate-600 font-mono">{(w * 100).toFixed(0)}%</span>
                        </div>
                    ))}
                </div>
            </div>
            
            <p className="text-[9px] text-slate-600 font-mono text-center max-w-xs uppercase tracking-widest leading-relaxed">
                {hoverIndex !== null 
                    ? `Showing how the model focuses on other words when processing "${words[hoverIndex]}"`
                    : "Hover over a word to see its attention map"}
            </p>
        </div>
    );
};

const EmbeddingsViz = () => (
    <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" hide domain={[0, 8]} />
                <YAxis type="number" dataKey="y" hide domain={[0, 8]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: '8px' }} />
                
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 2, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" />
                <ReferenceLine segment={[{x: 6, y: 2}, {x: 6, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" />
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 6, y: 2}]} stroke="#475569" strokeDasharray="2 2" />

                <Scatter data={embeddingData} shape="circle">
                    <LabelList dataKey="label" position="top" fill="#cbd5e1" fontSize={11} offset={10} fontWeight="bold" />
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
    </div>
);

const BackpropViz = () => (
    <div className="flex flex-col gap-4 p-5 bg-slate-900/50 rounded-2xl border border-slate-800/50">
        <div className="flex justify-between items-center text-[8px] font-black uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-3">
             <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div> Forward</span>
             <span className="flex items-center gap-1">Backward <div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div></span>
        </div>
        <div className="flex justify-between items-center gap-2 py-2">
            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-indigo-500 flex items-center justify-center bg-indigo-900/20 text-indigo-300 font-bold shadow-lg shadow-indigo-900/20">x</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">In</span>
            </div>
            
            <div className="flex-1 h-[2px] bg-slate-800 relative group overflow-hidden">
                <div className="absolute inset-0 bg-indigo-500/20 animate-pulse"></div>
            </div>

            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-slate-600 flex items-center justify-center bg-slate-800 text-slate-300 font-bold">h</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">Hidden</span>
            </div>

             <div className="flex-1 h-[2px] bg-slate-800 relative group overflow-hidden">
                <div className="absolute inset-0 bg-rose-500/20 animate-pulse"></div>
            </div>

            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-emerald-500 flex items-center justify-center bg-emerald-900/20 text-emerald-300 font-bold">y&#770;</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">Pred</span>
            </div>
        </div>
        <div className="bg-slate-950 p-4 rounded-xl text-[10px] font-mono text-slate-500 border border-slate-800 leading-relaxed italic">
            <span className="text-rose-400 font-bold not-italic">Backprop Rule:</span> <br/>
            Chain gradients backwards to update weights using calculus.
        </div>
    </div>
);

const BPTTViz = () => {
    const [phase, setPhase] = useState<'forward' | 'backward'>('forward');
    const [step, setStep] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setStep(prev => {
                if (phase === 'forward') {
                    if (prev >= 3) { 
                        setPhase('backward');
                        return 2;
                    }
                    return prev + 1;
                } else {
                    if (prev <= 0) {
                        setPhase('forward');
                        return 0;
                    }
                    return prev - 1;
                }
            });
        }, 1000);
        return () => clearInterval(interval);
    }, [phase]);

    return (
        <div className="w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-6 relative overflow-hidden">
            <div className="absolute top-3 right-4 text-[9px] font-mono font-black uppercase tracking-widest bg-slate-900 px-2 py-1 rounded border border-slate-800">
                Mode: <span className={phase === 'forward' ? 'text-indigo-400' : 'text-rose-400'}>{phase === 'forward' ? 'Inference' : 'Learning (BPTT)'}</span>
            </div>
            
            <div className="flex justify-around items-center mt-8">
                {[0, 1, 2].map((t) => {
                    const isActive = phase === 'forward' ? step >= t : step <= t;
                    const isGradient = phase === 'backward' && step <= t;

                    return (
                        <div key={t} className="flex flex-col items-center gap-2 relative group">
                             <div className="absolute -top-7 text-[8px] text-slate-600 font-mono font-bold tracking-widest">T={t}</div>
                             <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-all duration-500 ${isGradient ? 'border-rose-500 bg-rose-900/30 text-rose-300 scale-110' : isActive ? 'border-emerald-500 bg-emerald-900/30 text-emerald-300' : 'border-slate-800 bg-slate-900 text-slate-700'}`}>
                                Y
                             </div>
                             <div className={`w-[2px] h-6 transition-colors duration-500 ${isGradient ? 'bg-rose-500' : isActive ? 'bg-slate-700' : 'bg-slate-900'}`}></div>
                             <div className={`w-10 h-10 rounded-xl border-2 flex items-center justify-center text-xs font-bold transition-all duration-500 z-10 ${isGradient ? 'border-rose-500 bg-rose-900/30 text-rose-300 scale-110 shadow-[0_0_20px_rgba(244,63,94,0.3)]' : isActive ? 'border-indigo-500 bg-indigo-900/20 text-indigo-300' : 'border-slate-800 bg-slate-900 text-slate-700'}`}>
                                H
                             </div>
                             <div className={`w-[2px] h-6 transition-colors duration-500 ${isActive ? 'bg-slate-700' : 'bg-slate-900'}`}></div>
                             <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-all duration-500 ${isActive ? 'border-slate-500 bg-slate-800 text-white' : 'border-slate-900 bg-slate-950 text-slate-800'}`}>
                                X
                             </div>
                             {t < 2 && (
                                 <div className="absolute top-[3.5rem] left-9 w-12 h-8 flex items-center justify-center">
                                     <div className={`absolute w-full h-[2px] transition-colors duration-500 ${phase === 'forward' && step > t ? 'bg-indigo-500' : 'bg-slate-900'}`}></div>
                                     <div className={`absolute w-full h-[2px] transition-all duration-500 ${phase === 'backward' && step <= t ? 'bg-rose-500 translate-y-1 opacity-100' : 'opacity-0'}`}></div>
                                 </div>
                             )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const RAGViz = () => (
    <div className="flex flex-col gap-6 p-6 bg-slate-900/50 rounded-2xl border border-slate-800 items-center justify-center relative overflow-hidden select-none">
        {/* Connection Dashed Line */}
        <div className="absolute top-1/2 left-4 right-4 h-0 border-t-2 border-dashed border-slate-800/50 -z-0 hidden md:block"></div>

        <div className="flex flex-col md:flex-row items-center justify-between w-full max-w-lg gap-4 z-10">
            {/* Step 1: Query */}
            <div className="flex flex-col items-center gap-3">
                <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center shadow-lg group">
                    <Search size={22} className="text-indigo-400 group-hover:scale-110 transition-transform" />
                </div>
                <div className="text-center">
                    <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Query</div>
                    <div className="text-[9px] text-slate-600 font-mono">Input</div>
                </div>
            </div>

            <ArrowRight size={16} className="text-slate-700 rotate-90 md:rotate-0" />

            {/* Step 2: Vector DB */}
            <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-2xl bg-emerald-900/10 border border-emerald-500/20 flex flex-col items-center justify-center shadow-lg relative overflow-hidden group">
                    <div className="absolute inset-0 bg-emerald-500/5 group-hover:bg-emerald-500/10 transition-colors"></div>
                    <Database size={24} className="text-emerald-500 z-10" />
                    <div className="absolute bottom-1 w-8 h-0.5 bg-emerald-500/50 rounded-full animate-pulse"></div>
                </div>
                <div className="text-center">
                    <div className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">Vector DB</div>
                    <div className="text-[9px] text-emerald-500/60 font-mono">Similarity Search</div>
                </div>
            </div>

            <ArrowRight size={16} className="text-slate-700 rotate-90 md:rotate-0" />

            {/* Step 3: Context */}
            <div className="flex flex-col items-center gap-3">
                <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center shadow-lg relative group">
                    <FileText size={22} className="text-amber-400 z-10 group-hover:-translate-y-1 transition-transform" />
                    <FileText size={22} className="text-amber-400/50 absolute top-4 left-5 -z-0" />
                </div>
                <div className="text-center">
                    <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Context</div>
                    <div className="text-[9px] text-slate-600 font-mono">Top-K Chunks</div>
                </div>
            </div>

            <ArrowRight size={16} className="text-slate-700 rotate-90 md:rotate-0" />

            {/* Step 4: LLM */}
            <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 flex items-center justify-center shadow-xl shadow-indigo-600/20">
                    <Bot size={28} className="text-white" />
                </div>
                <div className="text-center">
                    <div className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">LLM</div>
                    <div className="text-[9px] text-indigo-400/60 font-mono">Generation</div>
                </div>
            </div>
        </div>
        
        <div className="bg-slate-950 px-4 py-1.5 rounded-full border border-slate-800 text-[9px] text-slate-500 font-mono z-20 mt-4 md:mt-0">
            Augmented Prompt = Context + Query
        </div>
    </div>
);

import { motion } from 'framer-motion';

export const DeepLearningView: React.FC = () => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-12 pb-20"
    >
      <header className="mb-12 border-b border-slate-800 pb-8">
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-serif font-bold text-white mb-4"
        >
          Deep Learning
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light"
        >
          Harnessing the power of multi-layered artificial neural networks. Deep learning identifies complex hierarchies of features within massive datasets to solve tasks once thought impossible for machines.
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
        id="perceptron"
        title="The Perceptron"
        complexity="Fundamental"
        theory={`The simplest form of a neural network. A single neuron that takes multiple inputs, weights them, sums them up, adds a bias, and passes the result through an activation function (originally a step function) to produce a binary output. It forms the building block of all deep learning.
        
### Mathematical Model
1. **Weighted Sum:** z = w₁x₁ + w₂x₂ + ... + b
2. **Activation:** y = step(z)`}
        math={<span>y = &phi;(&Sigma; w<sub>i</sub>x<sub>i</sub> + b)</span>}
        mathLabel="Perceptron Formula"
        code={`class Perceptron:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = 0

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear > 0, 1, 0)`}
        pros={['Simple and interpretable', 'Guaranteed to converge for linearly separable data', 'Foundation of modern AI']}
        cons={['Can ONLY solve linearly separable problems', 'Fails on XOR problem', 'Step function is not differentiable (cannot use backprop directly)']}
        steps={[
            "Initialize weights and bias to small random numbers.",
            "For each training example:",
            "1. Calculate prediction: y_pred = step(w·x + b)",
            "2. Calculate error: error = y_true - y_pred",
            "3. Update weights: w = w + lr * error * x",
            "4. Update bias: b = b + lr * error",
            "Repeat until convergence."
        ]}
      >
        <PerceptronViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="activation-functions"
        title="Activation Functions"
        complexity="Fundamental"
        theory={`Activation functions introduce non-linearity into the network. Without them, a neural network of any depth is mathematically equivalent to a single linear layer. They determine whether a neuron should "fire" or not.
        
### Common Functions
* **Sigmoid:** Smooth step, outputs (0,1). Good for probability.
* **ReLU:** max(0,x). Efficient, solves vanishing gradient.
* **Tanh:** Zero-centered (-1,1).`}
        math={<span>ReLU(x) = max(0, x)</span>}
        mathLabel="Rectified Linear Unit"
        code={`# Common Activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)`}
        pros={['Enables learning of complex non-linear patterns', 'Differentiability allows backpropagation', 'Control over output range']}
        cons={['Sigmoid/Tanh suffer from vanishing gradients', 'ReLU can suffer from "dead neurons"', 'Choice depends heavily on architecture']}
        steps={[
            "Hidden Layers: Default to **ReLU** (Rectified Linear Unit). It's fast and effective.",
            "Output Layer (Binary): Use **Sigmoid** to get a probability between 0 and 1.",
            "Output Layer (Multi-class): Use **Softmax** to get a probability distribution summing to 1.",
            "RNNs: Often use **Tanh** or **Sigmoid** for gating mechanisms."
        ]}
      >
        <ActivationFunctionsViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="mlp"
        title="Multilayer Perceptrons"
        complexity="Intermediate"
        theory={`The fundamental neural architecture. It stacks fully-connected layers, applying non-linear activation functions (like ReLU) at each stage. Through the backpropagation algorithm, it learns to map raw inputs into sophisticated high-dimensional feature representations.

### Forward Pass
[Input x] -> [Weights W1] -> [ReLU] -> [Hidden h] -> [Weights W2] -> [Softmax] -> [Output y]`}
        math={<span>a<sup>[l]</sup> = &sigma;(W<sup>[l]</sup> a<sup>[l-1]</sup> + b<sup>[l]</sup>)</span>}
        mathLabel="Forward Pass"
        code={`from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])`}
        pros={['Universal function approximator', 'Scales with data', 'Foundational for all deep models']}
        cons={['Requires vast amounts of labeled data', 'Hyperparameter tuning is difficult', 'Difficult to explain decision logic']}
        steps={[
            "Open Google Colab. Import `tensorflow` or `torch`.",
            "Preprocess data. Normalize inputs to [0,1] or [-1,1]. One-hot encode targets if categorical.",
            "Build model: Use `keras.Sequential` or `nn.Module`.",
            "Add Dense layers with 'relu' activation. Use 'softmax' (multi-class) or 'sigmoid' (binary) for the output layer.",
            "Compile: Choose optimizer (Adam) and loss function (CrossEntropy).",
            "Train: `model.fit(x, y, epochs=10, batch_size=32)`."
        ]}
      >
        <NeuralNetworkViz />
        <div className="mt-12">
            <h4 className="text-[10px] font-black text-rose-400 uppercase tracking-[0.3em] mb-6">Learning Process: Backpropagation</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                <p className="text-slate-500 text-sm leading-relaxed font-light">
                    Networks learn by minimizing error. <strong className="text-slate-300">Backpropagation</strong> utilizes the Chain Rule to distribute error gradients from the output layer back through every weight in the network, informing precisely how to nudge each parameter to improve future predictions.
                </p>
                <BackpropViz />
            </div>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="cnn"
        title="Convolutional Networks"
        complexity="Intermediate"
        theory={`Specialized for grid-structured data like images. CNNs use convolutional kernels to extract spatial features while preserving local relationships. They are highly efficient due to parameter sharing and spatial pooling mechanisms.

### CNN Pipeline
[Image] -> [Conv] -> [ReLU] -> [Pool] -> [Conv] -> [ReLU] -> [Pool] -> [FC] -> [Class]`}
        math={<span>(I * K)(i, j) = &Sigma; &Sigma; I(m, n) K(i-m, j-n)</span>}
        mathLabel="Convolution Operation"
        code={`model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])`}
        pros={['Translation invariance', 'Automatic hierarchical feature extraction', 'Highly efficient for visual data']}
        cons={['High compute requirement', 'Loss of fine-grained spatial information through pooling', 'Adversarial vulnerability']}
        steps={[
            "Use Colab with GPU runtime (`Runtime > Change runtime type > GPU`).",
            "Load image dataset (e.g., MNIST/CIFAR-10). Normalize pixel values.",
            "Build: `Conv2D` layers (filters, kernel_size) followed by `MaxPooling2D`.",
            "Flatten the output of convolutions.",
            "Add Dense layers for classification.",
            "Compile and Train. Watch accuracy increase as filters learn edges and shapes."
        ]}
      >
        <ConvolutionViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="rnn"
        title="Recurrent Networks (LSTM)"
        complexity="Advanced"
        theory={`Designed for sequential data processing. RNNs maintain an internal hidden state that acts as memory across time-steps. LSTMs (Long Short-Term Memory) refine this with specialized 'gates' that control information flow, solving long-range dependency issues.

### Unrolled RNN
h(t-1) --> [Cell A] --> h(t) --> [Cell A] --> h(t+1)
              ^                    ^
              |                    |
            x(t)                 x(t+1)`}
        math={<span>h<sub>t</sub> = &sigma;(W<sub>hh</sub> h<sub>t-1</sub> + W<sub>xh</sub> x<sub>t</sub>)</span>}
        mathLabel="Hidden Update"
        code={`model = models.Sequential([
    layers.LSTM(128, input_shape=(None, 10))
])`}
        pros={['Handles variable-length inputs', 'Captures temporal dependencies', 'Standard for audio/time-series']}
        cons={['Slow to train (non-parallelizable)', 'Vanishing gradient issues in vanilla versions', 'Memory limitations for very long sequences']}
        steps={[
            "Prepare sequence data (Time Series or Text). Pad sequences to same length.",
            "Use `Embedding` layer if input is text.",
            "Add `LSTM` or `GRU` layer. Set `return_sequences=True` if stacking layers.",
            "End with Dense layer.",
            "Compile with Adam optimizer. Train."
        ]}
      >
         <BPTTViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="embeddings"
        title="Embeddings"
        complexity="Intermediate"
        theory={`Learned dense representations of discrete items. Unlike one-hot encoding, embeddings place similar items closer together in a continuous vector space, allowing models to learn semantic relationships mathematically.

### Word2Vec Concept
"King" - "Man" + "Woman" ≈ "Queen"`}
        math={<span>L = E[ log P(w<sub>context</sub> | w<sub>target</sub>) ]</span>}
        mathLabel="Word2Vec Objective"
        code={`from tensorflow.keras.layers import Embedding
layer = Embedding(input_dim=10000, output_dim=300)`}
        pros={['Dramatic dimensionality reduction', 'Captures semantic meaning', 'Transferable via pre-trained models']}
        cons={['Learns biases from training corpora', 'Static embeddings ignore context (e.g., "bank")']}
        steps={[
            "Tokenize text data (convert words to integers).",
            "Define vocab size and embedding dimension (e.g., 300).",
            "Use `layers.Embedding(vocab_size, embedding_dim)` as the first layer.",
            "Train as part of a larger network (e.g., classifier).",
            "Extract weights to visualize in 2D using PCA/t-SNE."
        ]}
      >
        <EmbeddingsViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="transformers"
        title="Transformers"
        complexity="Advanced"
        theory={`The modern standard for large-scale modeling. It utilizes a Self-Attention mechanism to draw global dependencies regardless of distance in the sequence. This enables massive parallelization and state-of-the-art performance in LLMs.

### Transformer Block
[Input] -> [Attention] -> [Add & Norm] -> [Feed Forward] -> [Add & Norm] -> [Output]`}
        math={<span>Attn(Q, K, V) = softmax(<sup>QK<sup>T</sup></sup>&frasl;<sub>&radic;d<sub>k</sub></sub>)V</span>}
        mathLabel="Scaled Dot-Product Attention"
        code={`# Attention mechanism logic
def attention(q, k, v):
    scores = matmul(q, k.T) / sqrt(dk)
    return matmul(softmax(scores), v)`}
        pros={['Massive parallel training capability', 'State-of-the-art across nearly all NLP benchmarks', 'Excellent long-range memory']}
        cons={['Quadratic compute cost with sequence length', 'Extreme data appetite', 'Complex implementation']}
        steps={[
            "Recommended: Use `Hugging Face Transformers` library in Colab.",
            "Import `AutoTokenizer` and `AutoModel`.",
            "Load a pre-trained model (e.g., 'bert-base-uncased').",
            "Fine-tune on your specific dataset using `Trainer` API.",
            "Or implement from scratch: Define Scaled Dot-Product Attention and Multi-Head layers manually in PyTorch/Keras."
        ]}
      >
        <AttentionViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="rag"
        title="Retrieval Augmented Generation (RAG)"
        complexity="Advanced"
        theory="RAG enhances Large Language Models by retrieving relevant data from external knowledge bases (Vector Databases) before generating a response. This grounds the model on specific, up-to-date, or private information, reducing hallucinations."
        math={<span>p(y|x) &asymp; &Sigma;<sub>z &isin; TopK(x)</sub> p(y|x,z)p(z|x)</span>}
        mathLabel="RAG Probability Approximation"
        code={`import chromadb

# 1. Initialize Client
client = chromadb.Client()
collection = client.create_collection("docs")

# 2. Add Documents
collection.add(
    documents=["Product A manual...", "Company policy..."],
    ids=["id1", "id2"]
)

# 3. Query at Runtime
results = collection.query(
    query_texts=["How do I reset Product A?"],
    n_results=2
)

# 4. Augment Prompt (Pseudo)
prompt = f"Context: {results['documents']}\nQuery: {user_query}"`}
        pros={['Access to private/proprietary data', 'Reduces model hallucinations', 'Easy to update knowledge (just update DB)']}
        cons={['Increased latency due to retrieval step', 'Depends heavily on retrieval quality', 'Context window limitations']}
        steps={[
            "Chunk your documents (e.g., 512 tokens with overlap).",
            "Embed chunks using a model like OpenAI's `text-embedding-3-small` or HuggingFace models.",
            "Store embeddings in a Vector Database (Pinecone, Chroma, Milvus).",
            "At runtime, embed user query and perform Cosine Similarity search.",
            "Retrieve Top-K chunks and inject them into the system prompt.",
            "Send augmented prompt to LLM for final answer generation."
        ]}
      >
        <RAGViz />
      </AlgorithmCard>
      </motion.div>
    </motion.div>
  );
};
