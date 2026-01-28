
import React, { useState, useEffect, useMemo } from 'react';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { LatexRenderer } from '../components/LatexRenderer';
import { BrainCircuit, Play, RotateCcw, Target, Shuffle, Activity, Database, Coins } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

// --- VISUALIZATIONS ---

const RLLoopViz = () => {
  const [phase, setPhase] = useState(0); 

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(p => (p + 1) % 4);
    }, 2000); // Slower cycle for clarity
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-8 gap-8 select-none w-full max-w-2xl mx-auto">
      {/* Top Path: Action */}
      <div className="relative w-full h-16 flex items-center justify-center">
        <div className="absolute left-[20%] right-[20%] h-1.5 bg-slate-800/50 rounded-full overflow-hidden">
             {/* Flow Animation */}
             <motion.div 
               className="h-full bg-gradient-to-r from-transparent via-indigo-500 to-transparent w-1/2"
               animate={{ x: phase === 1 || phase === 2 ? "200%" : "-100%" }}
               transition={{ duration: 1.5, ease: "easeInOut" }}
             />
        </div>
        <motion.div 
            className="absolute top-0 flex flex-col items-center"
            animate={{ 
                opacity: phase === 1 || phase === 2 ? 1 : 0.3,
                scale: phase === 1 || phase === 2 ? 1.1 : 1
            }}
        >
            <span className="text-xs font-mono font-bold text-indigo-400 tracking-widest bg-slate-900 px-2 py-1 rounded border border-indigo-500/30">ACTION (Aₜ)</span>
        </motion.div>
      </div>

      <div className="flex justify-between w-full items-center gap-12 px-8">
        {/* Agent */}
        <motion.div 
            animate={{ 
                borderColor: phase === 0 || phase === 1 ? 'rgba(99,102,241,0.6)' : 'rgba(30,41,59,0.5)',
                boxShadow: phase === 0 || phase === 1 ? '0 0 30px rgba(99,102,241,0.15)' : 'none',
                scale: phase === 1 ? 1.05 : 1
            }}
            className="relative z-10 p-8 rounded-3xl border-2 bg-slate-900 flex flex-col items-center w-48 h-48 justify-center transition-colors duration-500"
        >
           <BrainCircuit size={56} className={`mb-4 transition-colors duration-500 ${phase === 0 || phase === 1 ? 'text-indigo-400' : 'text-slate-600'}`} />
           <span className="text-sm font-bold text-slate-200">AGENT</span>
           <span className="text-[10px] text-slate-500 font-mono mt-1">POLICY &pi;(a|s)</span>
           
           {/* Internal Thought Process */}
           <AnimatePresence>
             {phase === 0 && (
                <motion.div 
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute -bottom-8 bg-indigo-500/10 text-indigo-300 text-[10px] px-2 py-1 rounded border border-indigo-500/20 whitespace-nowrap"
                >
                    Updating Weights...
                </motion.div>
             )}
           </AnimatePresence>
        </motion.div>

        {/* Env */}
        <motion.div 
            animate={{ 
                borderColor: phase === 2 || phase === 3 ? 'rgba(16,185,129,0.6)' : 'rgba(30,41,59,0.5)',
                boxShadow: phase === 2 || phase === 3 ? '0 0 30px rgba(16,185,129,0.15)' : 'none',
                scale: phase === 3 ? 1.05 : 1
            }}
            className="relative z-10 p-8 rounded-3xl border-2 bg-slate-900 flex flex-col items-center w-48 h-48 justify-center transition-colors duration-500"
        >
           <Target size={56} className={`mb-4 transition-colors duration-500 ${phase === 2 || phase === 3 ? 'text-emerald-400' : 'text-slate-600'}`} />
           <span className="text-sm font-bold text-slate-200">ENVIRONMENT</span>
           <span className="text-[10px] text-slate-500 font-mono mt-1">DYNAMICS P(s',r|s,a)</span>
        </motion.div>
      </div>

      {/* Bottom Path: Reward/State */}
      <div className="relative w-full h-16 flex items-center justify-center">
        <div className="absolute left-[20%] right-[20%] h-1.5 bg-slate-800/50 rounded-full overflow-hidden">
             {/* Flow Animation (Reverse Direction) */}
             <motion.div 
               className="h-full bg-gradient-to-l from-transparent via-emerald-500 to-transparent w-1/2"
               initial={{ x: "200%" }}
               animate={{ x: phase === 3 || phase === 0 ? "-100%" : "200%" }}
               transition={{ duration: 1.5, ease: "easeInOut" }}
             />
        </div>
        <motion.div 
            className="absolute bottom-0 flex flex-col items-center"
            animate={{ 
                opacity: phase === 3 || phase === 0 ? 1 : 0.3,
                scale: phase === 3 || phase === 0 ? 1.1 : 1
            }}
        >
            <span className="text-xs font-mono font-bold text-emerald-400 tracking-widest bg-slate-900 px-2 py-1 rounded border border-emerald-500/30">STATE, REWARD (S', R)</span>
        </motion.div>
      </div>
    </div>
  );
};

const BanditViz = () => {
  // True probabilities of winning for 3 arms (Hidden from agent)
  const TRUE_PROBS = [0.3, 0.7, 0.5];
  
  const [history, setHistory] = useState<{step: number, avgReward: number}[]>([]);
  const [armStats, setArmStats] = useState([
    { id: 0, pulls: 0, wins: 0, estimatedVal: 0.5 },
    { id: 1, pulls: 0, wins: 0, estimatedVal: 0.5 },
    { id: 2, pulls: 0, wins: 0, estimatedVal: 0.5 },
  ]);
  const [epsilon, setEpsilon] = useState(0.1);
  const [totalScore, setTotalScore] = useState(0);
  const [lastAction, setLastAction] = useState<{arm: number, result: 'win' | 'loss'} | null>(null);
  const [autoPlaying, setAutoPlaying] = useState(false);

  const pullArm = (armIndex: number) => {
    // Simulate environment
    const isWin = Math.random() < TRUE_PROBS[armIndex];
    const reward = isWin ? 1 : 0;

    setLastAction({ arm: armIndex, result: isWin ? 'win' : 'loss' });
    setTotalScore(prev => prev + reward);

    // Update agent knowledge (Q-value update)
    setArmStats(prev => {
      const newStats = [...prev];
      const arm = newStats[armIndex];
      arm.pulls += 1;
      arm.wins += reward;
      // Q_new = Q_old + (1/N) * (Reward - Q_old)  [Incremental Average]
      arm.estimatedVal = arm.estimatedVal + (1 / arm.pulls) * (reward - arm.estimatedVal);
      return newStats;
    });

    setHistory(prev => {
      const step = prev.length + 1;
      const currentTotal = prev.length > 0 ? prev[prev.length - 1].avgReward * (step - 1) : 0;
      const newAvg = (currentTotal + reward) / step;
      // Keep history manageable
      const newHist = [...prev, { step, avgReward: newAvg }];
      return newHist.slice(-50); 
    });
  };

  const agentStep = () => {
    let armToPull;
    // Epsilon-Greedy Logic
    if (Math.random() < epsilon) {
      // Explore
      armToPull = Math.floor(Math.random() * 3);
    } else {
      // Exploit
      let bestVal = -1;
      let bestArms: number[] = [];
      armStats.forEach((arm, idx) => {
        if (arm.estimatedVal > bestVal) {
          bestVal = arm.estimatedVal;
          bestArms = [idx];
        } else if (arm.estimatedVal === bestVal) {
          bestArms.push(idx);
        }
      });
      armToPull = bestArms[Math.floor(Math.random() * bestArms.length)];
    }
    pullArm(armToPull);
  };

  useEffect(() => {
    let interval: any;
    if (autoPlaying) {
      interval = setInterval(agentStep, 200);
    }
    return () => clearInterval(interval);
  }, [autoPlaying, armStats, epsilon]);

  const reset = () => {
    setHistory([]);
    setArmStats([
      { id: 0, pulls: 0, wins: 0, estimatedVal: 0.5 },
      { id: 1, pulls: 0, wins: 0, estimatedVal: 0.5 },
      { id: 2, pulls: 0, wins: 0, estimatedVal: 0.5 },
    ]);
    setTotalScore(0);
    setLastAction(null);
    setAutoPlaying(false);
  };

  return (
    <div className="space-y-8">
       {/* Controls */}
       <div className="flex flex-col md:flex-row justify-between items-center bg-slate-900/50 p-4 rounded-xl border border-slate-800 gap-6">
          <div className="flex-1 w-full md:w-auto">
             <div className="flex justify-between mb-2">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Exploration Rate (ε)</label>
                <span className="text-xs font-mono text-indigo-400">{epsilon.toFixed(2)}</span>
             </div>
             <input 
                type="range" min="0" max="1" step="0.05" 
                value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
             />
             <div className="flex justify-between text-[9px] text-slate-600 mt-1 uppercase">
                <span>Pure Greed</span>
                <span>Random</span>
             </div>
          </div>
          <div className="flex gap-3">
             <button onClick={() => setAutoPlaying(!autoPlaying)} className={`px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-all ${autoPlaying ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50' : 'bg-indigo-600 text-white hover:bg-indigo-500'}`}>
                {autoPlaying ? <span className="animate-pulse">Running...</span> : <><Play size={14} /> Auto-Run Agent</>}
             </button>
             <button onClick={reset} className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition-colors">
                <RotateCcw size={16} />
             </button>
          </div>
       </div>

       <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Bandits Interface */}
          <div className="bg-slate-950 rounded-2xl border border-slate-800 p-6 flex flex-col items-center justify-center gap-8 relative overflow-hidden">
             {/* Score Counter */}
             <div className="absolute top-4 right-4 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800 flex items-center gap-2">
                <Coins size={14} className="text-amber-400" />
                <span className="font-mono font-bold text-white">{totalScore}</span>
             </div>

             <div className="flex gap-4">
               {armStats.map((arm, i) => (
                  <motion.button 
                    key={i}
                    onClick={() => pullArm(i)}
                    whileTap={{ scale: 0.95 }}
                    className={`
                      relative group flex flex-col items-center gap-2 w-24 p-2 rounded-xl border-2 transition-all duration-100
                      ${lastAction?.arm === i ? (lastAction.result === 'win' ? 'border-emerald-500 bg-emerald-900/10' : 'border-rose-500 bg-rose-900/10') : 'border-slate-800 bg-slate-900 hover:border-slate-600'}
                    `}
                  >
                     <div className="w-16 h-20 bg-slate-800 rounded-lg flex items-center justify-center relative shadow-inner overflow-hidden">
                        <div className={`text-2xl font-bold transition-all ${lastAction?.arm === i ? 'scale-125' : 'scale-100'} ${lastAction?.arm === i ? (lastAction.result === 'win' ? 'text-emerald-400' : 'text-rose-400') : 'text-slate-600'}`}>
                           {lastAction?.arm === i ? (lastAction.result === 'win' ? '$' : 'X') : '?'}
                        </div>
                        {/* Probability "Peek" for learning viz */}
                        <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-700">
                           <motion.div 
                             className="h-full bg-indigo-500" 
                             initial={{ width: "50%" }}
                             animate={{ width: `${arm.estimatedVal * 100}%` }}
                             transition={{ duration: 0.5 }}
                           />
                        </div>
                     </div>
                     <span className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Arm {i+1}</span>
                     <div className="text-[9px] font-mono text-slate-400">Est: {arm.estimatedVal.toFixed(2)}</div>
                  </motion.button>
               ))}
             </div>
             <p className="text-xs text-slate-500 text-center max-w-xs">
                Click an arm to "Explore", or use Auto-Run to see Epsilon-Greedy in action. The bars show the agent's <span className="text-indigo-400">estimated value</span>.
             </p>
          </div>

          {/* Performance Chart */}
          <div className="bg-slate-950 rounded-2xl border border-slate-800 p-4 flex flex-col">
             <span className="text-[10px] font-black uppercase text-slate-500 tracking-widest mb-4">Average Reward over Time</span>
             <div className="flex-1 w-full min-h-[160px]">
                <ResponsiveContainer width="100%" height="100%">
                   <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis hide />
                      <YAxis domain={[0, 1]} tick={{fontSize: 10}} stroke="#475569" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '8px' }} itemStyle={{ fontSize: '12px' }} />
                      <ReferenceLine y={Math.max(...TRUE_PROBS)} stroke="#10b981" strokeDasharray="3 3" label={{ value: 'Optimal', fill: '#10b981', fontSize: 10 }} />
                      <Line type="monotone" dataKey="avgReward" stroke="#6366f1" strokeWidth={2} dot={false} isAnimationActive={false} />
                   </LineChart>
                </ResponsiveContainer>
             </div>
          </div>
       </div>
    </div>
  );
};

const GridWorldViz = () => {
    const grid = [
        ['S', 0, 0, 0],
        [0, 'X', 0, 'X'],
        [0, 0, 0, 'X'],
        [0, 'X', 0, 'G']
    ];
    
    const values = [
        [0.4, 0.5, 0.6, 0.7],
        [0.3, 0.0, 0.7, 0.0],
        [0.4, 0.6, 0.8, 0.0],
        [0.3, 0.0, 0.9, 1.0]
    ];

    const optimalPath = [
        {r:0, c:0}, {r:0, c:1}, {r:0, c:2}, {r:1, c:2}, {r:2, c:2}, {r:3, c:2}, {r:3, c:3}
    ];

    const [step, setStep] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    useEffect(() => {
        let timer: any;
        if (isRunning) {
            timer = setInterval(() => {
                setStep(prev => (prev >= optimalPath.length - 1 ? 0 : prev + 1));
            }, 800);
        }
        return () => clearInterval(timer);
    }, [isRunning]);

    const pos = optimalPath[step];

    return (
        <div className="flex flex-col md:flex-row gap-12 items-center justify-center p-6 bg-slate-950 rounded-2xl border border-slate-800">
            <div className="relative p-2 bg-slate-900 border-4 border-slate-800 rounded-xl shadow-2xl w-72 h-72">
                <div className="grid grid-cols-4 grid-rows-4 w-full h-full gap-2">
                    {grid.map((row, r) => row.map((cell, c) => (
                        <div key={`${r}-${c}`} className={`
                            relative flex items-center justify-center rounded-lg border overflow-hidden transition-colors duration-500
                            ${cell === 'X' ? 'bg-rose-950/20 border-rose-900/40' : 'border-slate-800/50'}
                            ${r === pos.r && c === pos.c ? 'bg-indigo-500/10 border-indigo-500/30' : ''}
                        `}>
                            {/* Heatmap Background */}
                            {cell !== 'X' && (
                                <div 
                                    className="absolute inset-0 bg-emerald-500/20 transition-opacity duration-500" 
                                    style={{ opacity: values[r][c] * 0.5 }} 
                                />
                            )}
                            
                            {cell !== 0 && <span className={`relative z-10 font-mono text-[10px] font-bold ${cell === 'X' ? 'text-rose-600' : 'text-slate-500'}`}>{cell}</span>}
                            {!cell && <span className="relative z-10 text-[8px] text-slate-600 font-mono opacity-50">{values[r][c].toFixed(1)}</span>}
                        </div>
                    )))}
                </div>

                {/* Smoothly Moving Agent */}
                <motion.div 
                    className="absolute w-[20%] h-[20%] bg-indigo-500 rounded-xl shadow-[0_0_25px_rgba(99,102,241,0.6)] flex items-center justify-center z-20 border-2 border-white/20"
                    animate={{ 
                        top: `${pos.r * 25 + 2.5}%`, 
                        left: `${pos.c * 25 + 2.5}%` 
                    }}
                    transition={{ 
                        type: "spring", 
                        stiffness: 200, 
                        damping: 25 
                    }}
                >
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                </motion.div>
            </div>

            <div className="flex flex-col gap-6 w-56">
                <div className="bg-slate-900/50 p-5 rounded-xl border border-slate-800 shadow-xl">
                    <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                        <Database size={12} className="text-indigo-400" /> State Analysis
                    </h4>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center text-xs font-mono border-b border-slate-800 pb-2">
                            <span className="text-slate-500">Coordinate</span>
                            <span className="text-indigo-400 font-bold">({pos.r}, {pos.c})</span>
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between items-center text-xs font-mono">
                                <span className="text-slate-500">Current Q(s,a)</span>
                                <span className="text-slate-300">{(values[pos.r][pos.c] * 0.8).toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs font-mono">
                                <span className="text-slate-500">Max Q(s')</span>
                                <span className="text-emerald-400 font-bold">{(values[pos.r][pos.c]).toFixed(2)}</span>
                            </div>
                        </div>
                        
                        <div className="pt-2">
                             <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                <motion.div 
                                    className="h-full bg-gradient-to-r from-indigo-500 to-emerald-500"
                                    animate={{ width: `${values[pos.r][pos.c] * 100}%` }}
                                />
                             </div>
                             <div className="text-[9px] text-right mt-1 text-slate-600">Value Confidence</div>
                        </div>
                    </div>
                </div>

                <div className="flex gap-3">
                    <button onClick={() => setIsRunning(!isRunning)} className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white py-3 rounded-lg shadow-lg shadow-indigo-900/40 font-bold flex items-center justify-center gap-2 transition-all active:scale-95">
                        {isRunning ? <span className="w-3 h-3 bg-white animate-pulse"></span> : <Play size={16} fill="white" />} {isRunning ? "Running" : "Start"}
                    </button>
                    <button onClick={() => { setStep(0); setIsRunning(false); }} className="p-3 bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-700 transition-all">
                        <RotateCcw size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

const ActorCriticViz = () => {
    return (
        <div className="w-full h-80 bg-slate-950 border border-slate-800 rounded-3xl overflow-hidden relative shadow-2xl select-none flex items-center justify-center">
            <svg width="600" height="300" viewBox="0 0 600 300" className="w-full h-full">
                <defs>
                    <marker id="arrow_head" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                        <path d="M0,0 L0,6 L6,3 z" fill="#475569" />
                    </marker>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#6366f1" stopOpacity="0" />
                        <stop offset="50%" stopColor="#6366f1" stopOpacity="1" />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
                    </linearGradient>
                </defs>

                {/* --- Static Connections --- */}
                {/* Env to Agent Split */}
                <path d="M 150 150 L 220 150" stroke="#1e293b" strokeWidth="2" markerEnd="url(#arrow_head)" />
                <path d="M 220 150 C 260 150, 260 80, 400 80" stroke="#1e293b" strokeWidth="2" fill="none" />
                <path d="M 220 150 C 260 150, 260 220, 400 220" stroke="#1e293b" strokeWidth="2" fill="none" />
                
                {/* Feedback Loops */}
                <path d="M 400 220 C 350 220, 400 80, 400 110" stroke="#f43f5e" strokeWidth="2" strokeDasharray="4 4" fill="none" opacity="0.3" />

                {/* --- Nodes --- */}
                {/* Environment Node */}
                <g transform="translate(100, 150)">
                    <circle r="40" fill="#0f172a" stroke="#1e293b" strokeWidth="2" />
                    <Target size={28} x="-14" y="-14" className="text-emerald-500" />
                    <text y="55" textAnchor="middle" fill="#475569" fontSize="10" fontWeight="bold" letterSpacing="1px">ENVIRONMENT</text>
                </g>

                {/* Actor Node */}
                <g transform="translate(450, 80)">
                    <circle r="40" fill="#0f172a" stroke="#6366f1" strokeWidth="2" className="animate-pulse-slow" style={{ filter: 'url(#glow)' }} />
                    <Shuffle size={28} x="-14" y="-14" className="text-indigo-400" />
                    <text y="55" textAnchor="middle" fill="#818cf8" fontSize="10" fontWeight="bold" letterSpacing="1px">ACTOR (&pi;)</text>
                    <text x="50" y="5" fill="#6366f1" fontSize="10" fontWeight="bold">Action</text>
                </g>

                {/* Critic Node */}
                <g transform="translate(450, 220)">
                    <circle r="40" fill="#0f172a" stroke="#f43f5e" strokeWidth="2" />
                    <Activity size={28} x="-14" y="-14" className="text-rose-400" />
                    <text y="55" textAnchor="middle" fill="#f43f5e" fontSize="10" fontWeight="bold" letterSpacing="1px">CRITIC (V)</text>
                    <text x="50" y="5" fill="#f43f5e" fontSize="10" fontWeight="bold">TD Error</text>
                </g>

                {/* --- Dynamic Data Packets (Smooth Framer Motion logic simulated via CSS for infinite loops) --- */}
                
                {/* State Signal to Actor */}
                <circle r="4" fill="#6366f1">
                    <animateMotion dur="2s" repeatCount="indefinite" path="M 140 150 L 220 150 C 260 150, 260 80, 410 80" keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.4 0 0.2 1" />
                </circle>

                {/* State Signal to Critic */}
                <circle r="4" fill="#10b981">
                    <animateMotion dur="2s" repeatCount="indefinite" path="M 140 150 L 220 150 C 260 150, 260 220, 410 220" keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.4 0 0.2 1" />
                </circle>

                {/* Critique (Error) Signal Feedback */}
                <circle r="3" fill="#f43f5e">
                    <animateMotion dur="2s" begin="1s" repeatCount="indefinite" path="M 450 180 L 450 120" />
                    <animate attributeName="opacity" values="0;1;0" dur="2s" begin="1s" repeatCount="indefinite" />
                </circle>

            </svg>

            {/* Labels overlay */}
            <div className="absolute top-[45%] left-[30%] bg-slate-900/90 px-2 py-1 rounded border border-slate-800 backdrop-blur-sm">
                <span className="text-[10px] font-mono text-emerald-400 font-bold">State (Sₜ)</span>
            </div>
        </div>
    );
};

export const ReinforcementView: React.FC = () => {
  return (
    <div className="space-y-16 animate-fade-in pb-20">
      <header>
        <h1 className="text-6xl font-serif font-bold text-white mb-6">Reinforcement Learning</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed">
          The computational study of goal-directed learning. Reinforcement Learning (RL) involves agents that learn through experience—maximizing rewards while navigating uncertain environments.
        </p>
      </header>

      <section id="rl-foundations" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">01. The RL Cycle</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         <AlgorithmCard
            id="mdp"
            title="Markov Decision Processes"
            theory={`The formal framework for decision making. An MDP is defined by its states, actions, transition probabilities, and rewards. It assumes the Markov Property: 'The future depends only on the present'.

### The Cycle
[State S] -> [Agent] -> [Action A] -> [Environment] -> [Reward R, Next State S']`}
            math={<span>&langle; S, A, P, R, &gamma; &rangle;</span>}
            mathLabel="The RL Quintuple"
            code={`# The fundamental RL loop
state = env.reset()
for _ in range(steps):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    agent.learn(state, action, reward, next_state)
    state = next_state`}
            pros={['Provides mathematical guarantees', 'Handles long-term planning', 'Generalizes across domains']}
            cons={['Computationally hard for large state spaces', 'Sensitive to sparse rewards']}
            steps={[
                "Open Google Colab. Install `gym` environment: `!pip install gym`.",
                "Import dependencies: `import gym`, `import numpy as np`.",
                "Initialize environment: `env = gym.make('CartPole-v1')`.",
                "Inspect Action and Observation spaces: `env.action_space`, `env.observation_space`.",
                "Run a random agent loop: `state = env.reset()`, `action = env.action_space.sample()`, `env.step(action)`."
            ]}
         >
            <RLLoopViz />
         </AlgorithmCard>
      </section>

      <section id="exploration-exploitation" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">02. Exploration vs. Exploitation</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         <AlgorithmCard
            id="bandits"
            title="Multi-Armed Bandits"
            complexity="Fundamental"
            theory={`Before tackling complex environments, agents must solve the fundamental dilemma: Do I 'Exploit' my current knowledge to get the best known reward, or 'Explore' new options to potentially find something better?

### Epsilon-Greedy Strategy
With probability ε, choose a random action (Explore).
With probability 1-ε, choose the best known action (Exploit).`}
            math={<LatexRenderer formula="Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)" />}
            mathLabel="Incremental Value Update"
            code={`# Epsilon-Greedy Logic
if random.random() < epsilon:
    action = random.choice(actions) # Explore
else:
    action = argmax(Q_values)       # Exploit`}
            pros={['Simple but effective baseline', 'Guarantees exploration', 'Tunable via Epsilon']}
            cons={['Exploration is random (inefficient)', 'Performance drops if Epsilon is not decayed']}
            hyperparameters={[
              { name: 'epsilon', description: 'Probability of choosing a random action (Exploration Rate).', default: '0.1' }
            ]}
            steps={[
                "Define a Bandit class in Colab with true probabilities for 'arms'.",
                "Initialize Q-values array to zeros.",
                "Run a loop for N trials.",
                "Generate random number `r`. If `r < epsilon`, select random arm. Else, select `argmax(Q)`.",
                "Simulate pulling the arm (return 1 or 0 based on probability).",
                "Update Q-value for the chosen arm: `Q[a] = Q[a] + alpha * (reward - Q[a])`."
            ]}
         >
            <BanditViz />
         </AlgorithmCard>
      </section>

      <section id="value-based" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">03. Value-Based Methods</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
        <AlgorithmCard
            id="q-learning"
            title="Q-Learning"
            theory={`A model-free algorithm that learns the value of every action in every state. It stores these estimates in a Q-Table, iteratively updating them based on the Bellman equation.

### Update Logic
Q(s,a) <-- Q(s,a) + alpha * [Target - Q(s,a)]`}
            math={<span>Q(s,a) &larr; Q(s,a) + &alpha; [r + &gamma; max Q(s',a') - Q(s,a)]</span>}
            mathLabel="Temporal Difference Update"
            code={`# Q-Table Update Logic
target = reward + gamma * np.max(Q[next_state])
error = target - Q[state, action]
Q[state, action] += alpha * error`}
            pros={['Guaranteed convergence', 'Off-policy (learns from any data)', 'Simple to implement']}
            cons={['Scales poorly to high dimensions', 'Requires discrete actions']}
            steps={[
                "Use `gym.make('FrozenLake-v1')` in Colab.",
                "Initialize `Q_table = np.zeros((state_space, action_space))`.",
                "Set hyperparameters: `alpha` (learning rate), `gamma` (discount), `epsilon`.",
                "Loop through episodes. Reset environment.",
                "Step in environment. Apply formula: `Q[s,a] = Q[s,a] + alpha * (R + gamma * max(Q[s']) - Q[s,a])`.",
                "Decay epsilon over time to shift from exploration to exploitation."
            ]}
        >
            <GridWorldViz />
        </AlgorithmCard>
      </section>

      <section id="actor-critic" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">04. Advanced Paradigms</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         <div className="grid grid-cols-1 md:grid-cols-2 gap-12 bg-slate-900 border border-slate-800 rounded-3xl p-10 shadow-2xl">
             <div className="space-y-6">
                 <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                    <span className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg">AC</span> Actor-Critic Methods
                 </h3>
                 <p className="text-slate-400 leading-relaxed">
                     Actor-Critic methods solve the high variance problem of pure policy gradients by combining two networks:
                 </p>
                 <div className="space-y-4">
                     <div className="flex gap-4 items-start bg-slate-950/50 p-4 rounded-xl border border-slate-800">
                         <div className="w-8 h-8 rounded bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold flex-shrink-0">A</div>
                         <div>
                            <p className="text-sm font-bold text-slate-200 mb-1">The Actor (Policy)</p>
                            <p className="text-xs text-slate-500">Proposes actions based on the current state. It tries to maximize the value estimated by the Critic.</p>
                         </div>
                     </div>
                     <div className="flex gap-4 items-start bg-slate-950/50 p-4 rounded-xl border border-slate-800">
                         <div className="w-8 h-8 rounded bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold flex-shrink-0">C</div>
                         <div>
                            <p className="text-sm font-bold text-slate-200 mb-1">The Critic (Value)</p>
                            <p className="text-xs text-slate-500">Evaluates the action by computing the TD Error (Surprise). This error signal is used to update both networks.</p>
                         </div>
                     </div>
                 </div>
             </div>
             <ActorCriticViz />
         </div>
      </section>
    </div>
  );
};
