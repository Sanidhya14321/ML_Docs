import React, { useState, useEffect, useMemo } from 'react';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { BrainCircuit, Play, RotateCcw, Target, Shuffle, Activity, Database } from 'lucide-react';

// --- VISUALIZATIONS ---

const RLLoopViz = () => {
  const [phase, setPhase] = useState(0); 

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(p => (p + 1) % 4);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-12 gap-6 select-none w-full max-w-2xl mx-auto">
      <div className="relative w-full h-12 flex items-center">
        <div className="absolute left-[20%] right-[20%] h-1 bg-slate-800 rounded-full">
             <div 
               className={`absolute top-[-2px] w-6 h-6 bg-indigo-500 rounded-full shadow-[0_0_15px_rgba(99,102,241,0.8)] transition-all duration-1000 ease-in-out ${phase === 2 ? 'left-[95%] opacity-100' : 'left-[5%] opacity-0'}`}
             ></div>
        </div>
        <span className={`absolute top-[-20px] left-1/2 -translate-x-1/2 text-[10px] font-mono tracking-widest transition-colors ${phase === 2 ? 'text-indigo-400 font-bold' : 'text-slate-600'}`}>ACTION (Aₜ)</span>
      </div>

      <div className="flex justify-between w-full items-center gap-12">
        <div className={`relative z-10 p-8 rounded-3xl border-2 transition-all duration-700 flex flex-col items-center w-40 h-40 justify-center bg-slate-900 ${phase === 1 ? 'border-indigo-500 shadow-[0_0_30px_rgba(99,102,241,0.2)] scale-110' : 'border-slate-800'}`}>
           <BrainCircuit size={48} className={`transition-colors duration-500 ${phase === 1 ? 'text-indigo-400' : 'text-slate-700'}`} />
           <span className="mt-3 font-bold text-slate-300">AGENT</span>
           <span className="text-[8px] text-slate-500 font-mono">POLICY &pi;(a|s)</span>
        </div>

        <div className={`relative z-10 p-8 rounded-3xl border-2 transition-all duration-700 flex flex-col items-center w-40 h-40 justify-center bg-slate-900 ${phase === 3 ? 'border-emerald-500 shadow-[0_0_30px_rgba(16,185,129,0.2)] scale-110' : 'border-slate-800'}`}>
           <Target size={48} className={`transition-colors duration-500 ${phase === 3 ? 'text-emerald-400' : 'text-slate-700'}`} />
           <span className="mt-3 font-bold text-slate-300">ENV</span>
           <span className="text-[8px] text-slate-500 font-mono">DYNAMICS P(s',r|s,a)</span>
        </div>
      </div>

      <div className="relative w-full h-12 flex items-center">
        <div className="absolute left-[20%] right-[20%] h-1 bg-slate-800 rounded-full">
             <div 
               className={`absolute top-[-2px] w-12 h-6 bg-gradient-to-r from-emerald-500 to-amber-500 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.8)] transition-all duration-1000 ease-in-out ${phase === 0 ? 'right-[95%] opacity-100' : 'right-[5%] opacity-0'}`}
             ></div>
        </div>
        <span className={`absolute bottom-[-20px] left-1/2 -translate-x-1/2 text-[10px] font-mono tracking-widest transition-colors ${phase === 0 ? 'text-emerald-400 font-bold' : 'text-slate-600'}`}>STATE/REWARD (Sₜ₊₁, Rₜ)</span>
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
    
    // Theoretical Value Function V(s) to show heatmap
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
            }, 600);
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
                            relative flex items-center justify-center rounded-lg border overflow-hidden
                            ${cell === 'X' ? 'bg-rose-950/20 border-rose-900/40' : 'border-slate-800/50'}
                        `} style={{ backgroundColor: cell === 'X' ? undefined : `rgba(16, 185, 129, ${values[r][c] * 0.15})` }}>
                            {cell !== 0 && <span className={`font-mono text-[10px] font-bold ${cell === 'X' ? 'text-rose-600' : 'text-slate-500'}`}>{cell}</span>}
                            {!cell && <span className="text-[8px] text-slate-600 font-mono">{values[r][c].toFixed(1)}</span>}
                        </div>
                    )))}
                </div>

                <div 
                    className="absolute w-[20%] h-[20%] bg-indigo-500 rounded-xl shadow-[0_0_25px_rgba(99,102,241,0.6)] flex items-center justify-center transition-all duration-500 ease-in-out z-10 border-2 border-white/20"
                    style={{ top: `${pos.r * 25 + 2.5}%`, left: `${pos.c * 25 + 2.5}%` }}
                >
                    <div className="w-2 h-2 bg-white rounded-full animate-ping"></div>
                </div>
            </div>

            <div className="flex flex-col gap-6 w-56">
                <div className="bg-slate-900/50 p-5 rounded-xl border border-slate-800 shadow-xl">
                    <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                        <Database size={12} className="text-indigo-400" /> V-State Estimation
                    </h4>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center text-xs font-mono">
                            <span className="text-slate-500">Coordinate:</span>
                            <span className="text-indigo-400 font-bold">({pos.r}, {pos.c})</span>
                        </div>
                        <div className="flex justify-between items-center text-xs font-mono">
                            <span className="text-slate-500">Value V(s):</span>
                            <span className="text-emerald-400 font-bold">{values[pos.r][pos.c].toFixed(2)}</span>
                        </div>
                        <div className="pt-3 border-t border-slate-800">
                             <div className="flex items-center gap-2 text-[10px] text-slate-500">
                                <div className="w-2 h-2 rounded bg-emerald-500 opacity-20"></div> Learned Heatmap
                             </div>
                        </div>
                    </div>
                </div>

                <div className="flex gap-3">
                    <button onClick={() => setIsRunning(!isRunning)} className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white py-3 rounded-lg shadow-lg shadow-indigo-900/40 font-bold flex items-center justify-center gap-2 transition-all active:scale-95">
                        {isRunning ? <span className="w-3 h-3 bg-white"></span> : <Play size={16} fill="white" />} {isRunning ? "Pause" : "Learn"}
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
        <div className="w-full h-72 bg-slate-950 border border-slate-800 rounded-3xl overflow-hidden relative shadow-2xl select-none">
            <svg width="100%" height="100%" viewBox="0 0 500 250">
                <defs>
                    <marker id="arrow_head" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
                    </marker>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>

                {/* Paths */}
                <path d="M 120 125 L 200 125" stroke="#1e293b" strokeWidth="4" markerEnd="url(#arrow_head)" />
                <path d="M 200 125 C 250 125, 300 70, 360 70" stroke="#1e293b" strokeWidth="4" markerEnd="url(#arrow_head)" />
                <path d="M 200 125 C 250 125, 300 180, 360 180" stroke="#1e293b" strokeWidth="4" markerEnd="url(#arrow_head)" />
                <path d="M 400 150 L 400 100" stroke="#f43f5e" strokeWidth="2" strokeDasharray="5 5" opacity="0.4" />

                {/* Environment */}
                <g transform="translate(80, 125)">
                    <circle r="40" fill="#0f172a" stroke="#1e293b" strokeWidth="3" />
                    <Target size={32} x="-16" y="-16" className="text-emerald-500" />
                    <text y="55" textAnchor="middle" fill="#475569" fontSize="10" fontWeight="bold">ENV</text>
                </g>

                {/* Actor */}
                <g transform="translate(400, 70)">
                    <circle r="40" fill="#0f172a" stroke="#6366f1" strokeWidth="3" className="animate-pulse-slow" style={{ filter: 'url(#glow)' }} />
                    <Shuffle size={32} x="-16" y="-16" className="text-indigo-400" />
                    <text y="55" textAnchor="middle" fill="#818cf8" fontSize="10" fontWeight="bold">ACTOR (&pi;)</text>
                </g>

                {/* Critic */}
                <g transform="translate(400, 180)">
                    <circle r="40" fill="#0f172a" stroke="#f43f5e" strokeWidth="3" />
                    <Activity size={32} x="-16" y="-16" className="text-rose-400" />
                    <text y="55" textAnchor="middle" fill="#f43f5e" fontSize="10" fontWeight="bold">CRITIC (V)</text>
                </g>

                {/* Flow Particles */}
                <circle r="4" fill="#6366f1">
                    <animateMotion dur="2.5s" repeatCount="indefinite" path="M 120 125 L 200 125 C 250 125, 300 70, 360 70" />
                </circle>
                <circle r="4" fill="#10b981">
                    <animateMotion dur="2.5s" repeatCount="indefinite" path="M 120 125 L 200 125 C 250 125, 300 180, 360 180" />
                </circle>
                <circle r="3" fill="#f43f5e">
                    <animateMotion dur="1.2s" begin="0.8s" repeatCount="indefinite" path="M 400 150 L 400 100" />
                </circle>
            </svg>
            <div className="absolute top-1/2 left-44 -translate-y-1/2 flex flex-col gap-1">
                <span className="text-[8px] font-mono text-indigo-400 tracking-widest bg-slate-900/80 px-1">STATE</span>
                <span className="text-[8px] font-mono text-emerald-400 tracking-widest bg-slate-900/80 px-1">REWARD</span>
            </div>
             <div className="absolute top-1/2 right-12 -translate-y-1/2 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-500/20">
                <span className="text-[8px] font-mono text-rose-300 font-bold uppercase">TD-Error</span>
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
            theory="The formal framework for decision making. An MDP is defined by its states, actions, transition probabilities, and rewards. It assumes the Markov Property: 'The future depends only on the present'."
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
         >
            <RLLoopViz />
         </AlgorithmCard>
      </section>

      <section id="value-based" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">02. Value-Based Methods</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
        <AlgorithmCard
            id="q-learning"
            title="Q-Learning"
            theory="A model-free algorithm that learns the value of every action in every state. It stores these estimates in a Q-Table, iteratively updating them based on the Bellman equation."
            math={<span>Q(s,a) &larr; Q(s,a) + &alpha; [r + &gamma; max Q(s',a') - Q(s,a)]</span>}
            mathLabel="Temporal Difference Update"
            code={`# Q-Table Update Logic
target = reward + gamma * np.max(Q[next_state])
error = target - Q[state, action]
Q[state, action] += alpha * error`}
            pros={['Guaranteed convergence', 'Off-policy (learns from any data)', 'Simple to implement']}
            cons={['Scales poorly to high dimensions', 'Requires discrete actions']}
        >
            <GridWorldViz />
        </AlgorithmCard>
      </section>

      <section id="actor-critic" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white">03. Advanced Paradigms</h2>
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
                     <div className="flex gap-4 items-start">
                         <div className="w-8 h-8 rounded bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold flex-shrink-0">A</div>
                         <p className="text-sm text-slate-500"><strong className="text-slate-300">The Actor</strong> proposes actions based on the current policy. It aims to maximize the reward feedback from the critic.</p>
                     </div>
                     <div className="flex gap-4 items-start">
                         <div className="w-8 h-8 rounded bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold flex-shrink-0">C</div>
                         <p className="text-sm text-slate-500"><strong className="text-slate-300">The Critic</strong> evaluates the action by estimating the value function. It calculates the 'Advantage'—how much better an action was compared to the average.</p>
                     </div>
                 </div>
             </div>
             <ActorCriticViz />
         </div>
      </section>
    </div>
  );
};