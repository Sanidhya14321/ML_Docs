import React, { useState, useEffect, useRef } from 'react';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { BrainCircuit, Play, RotateCcw, Target, Zap, Trophy, Shuffle, Cpu, Activity, Database, ArrowRight, TrendingUp } from 'lucide-react';

// --- VISUALIZATIONS ---

// 1. The RL Loop (Agent <-> Environment)
const RLLoopViz = () => {
  const [phase, setPhase] = useState(0); // 0: State -> Agent, 1: Processing, 2: Action -> Env, 3: Processing

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(p => (p + 1) % 4);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-8 gap-4 select-none w-full max-w-2xl mx-auto">
      {/* Top Channel (Action) */}
      <div className="relative w-full h-12 flex items-center">
        <div className="absolute left-[20%] right-[20%] h-2 bg-slate-800 rounded-full overflow-hidden">
             {/* Flowing Particle */}
             <div 
               className={`absolute top-0 bottom-0 w-8 bg-indigo-500 rounded-full shadow-[0_0_10px_rgba(99,102,241,0.8)] transition-all duration-1000 ease-in-out ${phase === 2 ? 'left-[80%] opacity-100' : 'left-[10%] opacity-0'}`}
             ></div>
        </div>
        <span className={`absolute top-[-10px] left-1/2 -translate-x-1/2 text-xs font-mono transition-colors ${phase === 2 ? 'text-indigo-400 font-bold' : 'text-slate-600'}`}>Action (a<sub>t</sub>) &rarr;</span>
      </div>

      <div className="flex justify-between w-full items-center gap-12">
        {/* Agent Node */}
        <div className={`relative z-10 p-6 rounded-2xl border-2 transition-all duration-500 flex flex-col items-center w-32 h-32 justify-center bg-slate-900 ${phase === 1 ? 'border-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.4)] scale-105' : 'border-slate-700'}`}>
           <BrainCircuit size={40} className={`transition-colors duration-500 ${phase === 1 ? 'text-indigo-400' : 'text-slate-600'}`} />
           <span className="mt-2 font-bold text-slate-200">Agent</span>
           <span className="text-[10px] text-slate-500">Policy &pi;</span>
        </div>

        {/* Environment Node */}
        <div className={`relative z-10 p-6 rounded-2xl border-2 transition-all duration-500 flex flex-col items-center w-32 h-32 justify-center bg-slate-900 ${phase === 3 ? 'border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.4)] scale-105' : 'border-slate-700'}`}>
           <Target size={40} className={`transition-colors duration-500 ${phase === 3 ? 'text-emerald-400' : 'text-slate-600'}`} />
           <span className="mt-2 font-bold text-slate-200">Env</span>
           <span className="text-[10px] text-slate-500">Physics</span>
        </div>
      </div>

      {/* Bottom Channel (State/Reward) */}
      <div className="relative w-full h-12 flex items-center">
        <div className="absolute left-[20%] right-[20%] h-2 bg-slate-800 rounded-full overflow-hidden">
             {/* Flowing Particle */}
             <div 
               className={`absolute top-0 bottom-0 w-16 bg-gradient-to-r from-emerald-500 to-amber-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.8)] transition-all duration-1000 ease-in-out ${phase === 0 ? 'right-[80%] opacity-100' : 'right-[10%] opacity-0'}`}
             ></div>
        </div>
        <span className={`absolute bottom-[-10px] left-1/2 -translate-x-1/2 text-xs font-mono transition-colors ${phase === 0 ? 'text-emerald-400 font-bold' : 'text-slate-600'}`}>&larr; State (s<sub>t+1</sub>), Reward (r<sub>t</sub>)</span>
      </div>
    </div>
  );
};

// 2. Multi-Armed Bandit
const BanditViz = () => {
  const [machines, setMachines] = useState([
    { id: 0, trueWinRate: 0.3, estimated: 0.5, pulls: 0 },
    { id: 1, trueWinRate: 0.8, estimated: 0.5, pulls: 0 },
    { id: 2, trueWinRate: 0.4, estimated: 0.5, pulls: 0 }
  ]);
  const [activeMachine, setActiveMachine] = useState<number | null>(null);
  const [lastReward, setLastReward] = useState<number | null>(null);
  const [epsilon, setEpsilon] = useState(0.2);

  const pullLever = () => {
    if (activeMachine !== null) return; // Prevent double click

    // Epsilon-Greedy Choice
    let choice;
    if (Math.random() < epsilon) {
      choice = Math.floor(Math.random() * machines.length); // Explore
    } else {
      choice = machines.reduce((maxIdx, m, i, arr) => m.estimated > arr[maxIdx].estimated ? i : maxIdx, 0); // Exploit
    }

    setActiveMachine(choice);

    // Delay for animation
    setTimeout(() => {
        const win = Math.random() < machines[choice].trueWinRate;
        const reward = win ? 1 : 0;
        setLastReward(reward);
        
        // Update Stats
        setMachines(prev => prev.map((m, i) => {
            if (i !== choice) return m;
            const newPulls = m.pulls + 1;
            const newEst = m.estimated + (1/newPulls) * (reward - m.estimated);
            return { ...m, pulls: newPulls, estimated: newEst };
        }));

        // Reset animation state
        setTimeout(() => {
            setActiveMachine(null);
            setLastReward(null);
        }, 1000);
    }, 500);
  };

  return (
    <div className="flex flex-col gap-8 items-center">
       <div className="flex justify-between items-center w-full bg-slate-900 p-4 rounded-lg border border-slate-800">
          <div className="w-1/2">
             <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                Exploration Rate (&epsilon;): <span className="text-indigo-400">{epsilon}</span>
             </label>
             <input 
               type="range" min="0" max="1" step="0.1" 
               value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))}
               className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
             />
             <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                <span>0% (Greedy)</span>
                <span>100% (Random)</span>
             </div>
          </div>
          <button 
            onClick={pullLever}
            disabled={activeMachine !== null} 
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-3 rounded shadow-lg shadow-indigo-900/50 font-bold flex items-center gap-2 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play size={18} fill="currentColor" /> Pull Arm
          </button>
       </div>

       <div className="grid grid-cols-3 gap-8">
          {machines.map((m) => (
             <div key={m.id} className="relative group">
                {/* Machine Body */}
                <div className={`relative w-24 h-32 bg-slate-800 rounded-t-xl border-4 transition-colors duration-300 flex flex-col items-center justify-center overflow-hidden ${activeMachine === m.id ? 'border-yellow-400 bg-slate-800' : 'border-slate-600'}`}>
                    <div className="text-4xl">🎰</div>
                    
                    {/* Screen Flash */}
                    <div className={`absolute inset-0 bg-yellow-400/20 transition-opacity duration-200 ${activeMachine === m.id && lastReward === 1 ? 'opacity-100' : 'opacity-0'}`}></div>

                    {/* Stats */}
                    <div className="absolute bottom-1 w-full text-center">
                         <div className="text-[10px] text-slate-400 font-mono">Est: {(m.estimated * 100).toFixed(0)}%</div>
                    </div>
                </div>

                {/* Lever Base */}
                <div className="absolute top-8 -right-3 w-4 h-8 bg-slate-700 rounded-r-md border border-l-0 border-slate-600"></div>
                
                {/* Lever Arm (Animated) */}
                <div className="absolute top-8 -right-3 w-4 h-24 origin-top transition-transform duration-500 ease-in-out" style={{ 
                    transform: activeMachine === m.id ? 'rotate(180deg)' : 'rotate(0deg)'
                }}>
                    <div className="w-1.5 h-full bg-slate-400 mx-auto rounded-full relative">
                        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4 bg-rose-500 rounded-full shadow-sm"></div>
                    </div>
                </div>

                {/* Coin Animation */}
                {activeMachine === m.id && lastReward === 1 && (
                    <div className="absolute -top-12 left-1/2 -translate-x-1/2 text-2xl animate-bounce drop-shadow-[0_0_10px_rgba(250,204,21,0.8)]">
                        🪙
                    </div>
                )}
             </div>
          ))}
       </div>
    </div>
  );
};

// 3. Smooth Grid World Viz
const GridWorldViz = () => {
    // 4x4 Grid Definition
    const grid = [
        ['S', 0, 0, 0],
        [0, 'X', 0, 'X'],
        [0, 0, 0, 'X'],
        [0, 'X', 0, 'G']
    ];
    
    // Policy Arrows for Visualization (approximate solution)
    // 0: Up, 1: Right, 2: Down, 3: Left
    const policy = [
        ['→', '→', '↓', '↓'],
        ['↓', '', '↓', ''],
        ['↓', '→', '↓', ''],
        ['→', '', '→', '★']
    ];

    // Learned Path
    const optimalPath = [
        {r:0, c:0}, {r:0, c:1}, {r:0, c:2}, // Right, Right, Right
        {r:1, c:2}, {r:2, c:2},             // Down, Down
        {r:3, c:2}, {r:3, c:3}              // Down, Right (Goal)
    ];

    const [stepIndex, setStepIndex] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isRunning) {
            interval = setInterval(() => {
                setStepIndex(prev => {
                    if (prev >= optimalPath.length - 1) {
                        setIsRunning(false); // Stop at goal
                        return prev;
                    }
                    return prev + 1;
                });
            }, 800);
        }
        return () => clearInterval(interval);
    }, [isRunning]);

    const reset = () => {
        setIsRunning(false);
        setStepIndex(0);
    };

    const currentPos = optimalPath[stepIndex];
    const currentAction = stepIndex < optimalPath.length - 1 
        ? (optimalPath[stepIndex+1].r > currentPos.r ? "DOWN" : "RIGHT") 
        : "DONE";

    return (
        <div className="flex gap-8 items-start">
            <div className="relative p-1 bg-slate-900 border border-slate-800 rounded-lg shadow-xl w-64 h-64">
                {/* The Grid */}
                <div className="grid grid-cols-4 grid-rows-4 w-full h-full gap-1">
                    {grid.map((row, r) => row.map((cell, c) => (
                        <div key={`${r}-${c}`} className={`
                            relative flex items-center justify-center rounded border
                            ${cell === 'S' ? 'bg-indigo-900/20 border-indigo-500/30' : 
                              cell === 'G' ? 'bg-emerald-900/20 border-emerald-500/30' :
                              cell === 'X' ? 'bg-rose-900/20 border-rose-500/30' : 
                              'bg-slate-800/50 border-slate-700/50'}
                        `}>
                            {/* Cell Label */}
                            {cell !== 0 && (
                                <span className={`font-bold text-xs ${cell === 'X' ? 'text-rose-500' : 'text-slate-500'}`}>{cell}</span>
                            )}
                            
                            {/* Policy Arrow (Faint) */}
                            {cell !== 'X' && (
                                <span className="absolute text-slate-600/30 text-lg font-bold select-none">{policy[r][c]}</span>
                            )}
                        </div>
                    )))}
                </div>

                {/* The Agent (Absolute overlay for smooth movement) */}
                <div 
                    className="absolute w-[23%] h-[23%] bg-indigo-500 rounded-lg shadow-[0_0_15px_rgba(99,102,241,0.6)] flex items-center justify-center transition-all duration-700 ease-in-out z-10"
                    style={{
                        top: `${currentPos.r * 25 + 1}%`,
                        left: `${currentPos.c * 25 + 1}%`
                    }}
                >
                    <div className="w-2 h-2 bg-white rounded-full animate-ping"></div>
                </div>
            </div>

            <div className="flex flex-col gap-4">
                <div className="bg-slate-900 p-4 rounded border border-slate-800 w-48 shadow-lg">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                        <Database size={12} /> Q-Table Lookup
                    </h4>
                    <div className="text-sm font-mono text-slate-300">
                        State: <span className="text-indigo-400">({currentPos.r}, {currentPos.c})</span>
                    </div>
                    <div className="text-sm font-mono text-slate-300 mt-1">
                        Best Action: <span className="text-emerald-400 font-bold">{currentAction}</span>
                    </div>
                    <div className="mt-3 pt-3 border-t border-slate-800">
                         <div className="flex justify-between text-[10px] text-slate-500">
                            <span>Q(s, &uarr;): 0.1</span>
                            <span>Q(s, &darr;): <span className="text-emerald-500 font-bold">0.8</span></span>
                         </div>
                         <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                            <span>Q(s, &larr;): 0.0</span>
                            <span>Q(s, &rarr;): 0.2</span>
                         </div>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button onClick={() => setIsRunning(true)} disabled={isRunning || stepIndex === 6} className="bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded disabled:opacity-50 shadow-lg shadow-indigo-900/50">
                        <Play size={16} />
                    </button>
                    <button onClick={reset} className="bg-slate-700 hover:bg-slate-600 text-white p-2 rounded border border-slate-600">
                        <RotateCcw size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
};

// 4. Sliding Replay Buffer Viz
const ReplayBufferViz = () => {
    const [experiences, setExperiences] = useState([1, 2, 3, 4, 5]);
    const [sampling, setSampling] = useState<number[]>([]);

    useEffect(() => {
        const interval = setInterval(() => {
            // Add new experience
            setExperiences(prev => {
                const next = [Date.now(), ...prev];
                if (next.length > 6) next.pop(); // Keep size fixed
                return next;
            });
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            // Randomly sample indices
            const s1 = Math.floor(Math.random() * 5);
            const s2 = (s1 + 1 + Math.floor(Math.random() * 3)) % 5;
            setSampling([s1, s2]);
            setTimeout(() => setSampling([]), 800);
        }, 2500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full flex flex-col gap-2 overflow-hidden p-2">
             <div className="flex gap-2 transition-all duration-500 ease-in-out transform">
                {experiences.map((exp, i) => (
                    <div key={exp} className={`
                        w-16 h-12 flex-shrink-0 rounded border flex items-center justify-center text-[10px] font-mono transition-all duration-500
                        ${i === 0 ? 'animate-slide-in bg-emerald-900/40 border-emerald-500/50 text-emerald-300' : 'bg-slate-800 border-slate-700 text-slate-500'}
                        ${sampling.includes(i) ? 'ring-2 ring-indigo-500 bg-indigo-900/50 text-indigo-200 scale-105 shadow-[0_0_15px_rgba(99,102,241,0.5)]' : ''}
                    `}>
                        {i === 0 ? 'New' : `Exp ${i}`}
                    </div>
                ))}
             </div>
             <div className="flex justify-between text-[10px] text-slate-500 px-1">
                 <span>&uarr; Enqueue (Latest)</span>
                 <span>Dequeue (Oldest) &uarr;</span>
             </div>
        </div>
    );
};

// 5. SVG Based Actor-Critic Data Flow
const ActorCriticViz = () => {
    const [phase, setPhase] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setPhase(prev => (prev + 1) % 150); // Loop 0 to 150 for steps
        }, 30);
        return () => clearInterval(interval);
    }, []);
    
    // Geometry
    const width = 500;
    const height = 250;
    
    // Nodes
    const envPos = { x: 80, y: 125 };
    const actorPos = { x: 400, y: 70 };
    const criticPos = { x: 400, y: 180 };

    return (
        <div className="w-full h-64 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden relative shadow-inner select-none">
            <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
                <defs>
                    <linearGradient id="gradFlow" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#6366f1" stopOpacity="1" />
                        <stop offset="100%" stopColor="#f43f5e" stopOpacity="1" />
                    </linearGradient>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
                    </marker>
                </defs>

                {/* --- PATHS --- */}

                {/* Env -> Splitter */}
                <path d={`M ${envPos.x + 30} ${envPos.y} L ${envPos.x + 100} ${envPos.y}`} stroke="#334155" strokeWidth="2" />
                
                {/* Splitter -> Actor (State) */}
                <path d={`M ${envPos.x + 100} ${envPos.y} C ${envPos.x + 150} ${envPos.y}, ${actorPos.x - 100} ${actorPos.y}, ${actorPos.x - 40} ${actorPos.y}`} stroke="#334155" strokeWidth="2" markerEnd="url(#arrowhead)" />
                
                {/* Splitter -> Critic (State + Reward) */}
                <path d={`M ${envPos.x + 100} ${envPos.y} C ${envPos.x + 150} ${envPos.y}, ${criticPos.x - 100} ${criticPos.y}, ${criticPos.x - 40} ${criticPos.y}`} stroke="#334155" strokeWidth="2" markerEnd="url(#arrowhead)" />

                {/* Critic -> Actor (Advantage) - Dashed verticalish */}
                <path d={`M ${criticPos.x} ${criticPos.y - 40} L ${actorPos.x} ${actorPos.y + 40}`} stroke="#475569" strokeWidth="2" strokeDasharray="5 5" />

                {/* Actor -> Env (Action) - Loop back top */}
                <path d={`M ${actorPos.x} ${actorPos.y - 40} C ${actorPos.x} 20, ${envPos.x} 20, ${envPos.x} ${envPos.y - 40}`} stroke="#334155" strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="5 5" />


                {/* --- NODES --- */}

                {/* Environment */}
                <g transform={`translate(${envPos.x}, ${envPos.y})`}>
                    <circle r="35" fill="#1e293b" stroke="#334155" strokeWidth="2" />
                    <Target size={24} x="-12" y="-12" className="text-emerald-400" />
                    <text y="50" textAnchor="middle" fill="#94a3b8" fontSize="12" fontWeight="bold">Environment</text>
                </g>

                {/* Actor */}
                <g transform={`translate(${actorPos.x}, ${actorPos.y})`}>
                    <circle r="35" fill="#1e293b" stroke="#6366f1" strokeWidth="2" />
                    <Shuffle size={24} x="-12" y="-12" className="text-indigo-400" />
                    <text y="50" textAnchor="middle" fill="#818cf8" fontSize="12" fontWeight="bold">Actor (&pi;)</text>
                </g>

                {/* Critic */}
                <g transform={`translate(${criticPos.x}, ${criticPos.y})`}>
                    <circle r="35" fill="#1e293b" stroke="#f43f5e" strokeWidth="2" />
                    <Activity size={24} x="-12" y="-12" className="text-rose-400" />
                    <text y="50" textAnchor="middle" fill="#f472b6" fontSize="12" fontWeight="bold">Critic (V)</text>
                </g>


                {/* --- ANIMATIONS --- */}

                {/* Particle: State (Blue) -> Actor */}
                <circle r="4" fill="#6366f1">
                    <animateMotion 
                        dur="2s" 
                        repeatCount="indefinite"
                        path={`M ${envPos.x + 35} ${envPos.y} L ${envPos.x + 100} ${envPos.y} C ${envPos.x + 150} ${envPos.y}, ${actorPos.x - 100} ${actorPos.y}, ${actorPos.x - 40} ${actorPos.y}`}
                    />
                </circle>

                {/* Particle: State+Reward (Green) -> Critic */}
                <circle r="4" fill="#10b981">
                    <animateMotion 
                        dur="2s" 
                        repeatCount="indefinite"
                        path={`M ${envPos.x + 35} ${envPos.y} L ${envPos.x + 100} ${envPos.y} C ${envPos.x + 150} ${envPos.y}, ${criticPos.x - 100} ${criticPos.y}, ${criticPos.x - 40} ${criticPos.y}`}
                    />
                </circle>

                 {/* Particle: Advantage (Red) -> Actor */}
                 <circle r="4" fill="#f43f5e" opacity="0.8">
                    <animateMotion 
                        dur="1s"
                        begin="1s"
                        repeatCount="indefinite"
                        path={`M ${criticPos.x} ${criticPos.y - 40} L ${actorPos.x} ${actorPos.y + 40}`}
                    />
                </circle>
            </svg>
            
            {/* Labels overlay */}
            <div className="absolute top-24 left-44 text-[10px] text-indigo-300 font-mono">State</div>
            <div className="absolute bottom-24 left-44 text-[10px] text-emerald-300 font-mono">Reward</div>
            <div className="absolute top-1/2 right-24 -translate-y-1/2 text-[10px] text-rose-300 font-mono bg-slate-900 px-1">Advantage</div>
        </div>
    );
};


export const ReinforcementView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in">
      <header>
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Reinforcement Learning</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed">
          The science of decision making. Unlike supervised learning, RL agents learn by interacting with an environment, receiving feedback in the form of rewards or penalties, and optimizing a policy to maximize cumulative reward over time.
        </p>
      </header>

      {/* 1. FOUNDATIONS */}
      <section id="rl-foundations" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">01</span>
            <h2 className="text-2xl font-bold text-indigo-400 uppercase tracking-widest">The Cycle of Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         
         <AlgorithmCard
            id="mdp"
            title="Markov Decision Processes (MDP)"
            theory="The mathematical framework for RL. It assumes the environment satisfies the Markov Property: the future depends only on the current state, not the history."
            math={<span>(S, A, P, R, <span className="math-serif">&gamma;</span>)</span>}
            mathLabel="The MDP Tuple"
            code={`# The Interaction Loop
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.learn(state, action, reward, next_state)
    state = next_state`}
            pros={['General framework for decision problems', 'Can handle stochastic environments', 'Basis for optimal control']}
            cons={['Assumes perfect state information (usually)', 'Computationally hard to solve exactly for large S']}
         >
            <RLLoopViz />
         </AlgorithmCard>
      </section>

      {/* 2. EXPLORATION */}
      <section id="exploration" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">02</span>
            <h2 className="text-2xl font-bold text-amber-400 uppercase tracking-widest">Exploration vs. Exploitation</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>
         
         <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg p-6">
            <div className="mb-6">
                <h3 className="text-xl font-bold text-white mb-2">The Multi-Armed Bandit Problem</h3>
                <p className="text-slate-400">
                    Should the agent stick to the machine it <em>thinks</em> pays out the most (Exploit), or try a risky new machine to see if it's even better (Explore)? 
                    The <strong>Epsilon-Greedy</strong> strategy is a simple solution: explore with probability &epsilon;, exploit otherwise.
                </p>
            </div>
            <BanditViz />
         </div>
      </section>

      {/* 3. VALUE-BASED METHODS */}
      <section id="value-based" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">03</span>
            <h2 className="text-2xl font-bold text-emerald-400 uppercase tracking-widest">Value-Based Methods</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>

        <AlgorithmCard
            id="q-learning"
            title="Q-Learning (Tabular)"
            theory="A model-free algorithm that learns the quality (Q-value) of actions. It maintains a table Q(s,a) estimating the expected future reward for taking action 'a' in state 's'."
            math={<span>Q(s,a) &larr; Q(s,a) + <span className="math-serif">&alpha;</span>[r + <span className="math-serif">&gamma;</span> max<sub>a'</sub>Q(s',a') - Q(s,a)]</span>}
            mathLabel="Bellman Optimality Update"
            code={`import numpy as np
# Update Rule
td_target = reward + gamma * np.max(Q[next_state])
td_error = td_target - Q[state, action]
Q[state, action] += alpha * td_error`}
            pros={['Guaranteed convergence (tabular)', 'Off-policy (learns from observed data)', 'Simple to implement']}
            cons={['Impossible for large state spaces (Curse of Dimensionality)', 'Slow propagation of rewards']}
        >
            <GridWorldViz />
        </AlgorithmCard>

        <AlgorithmCard
            id="dqn"
            title="Deep Q-Networks (DQN)"
            theory="Solving the storage problem of Q-tables by using a Neural Network to approximate the Q-function. It introduced 'Experience Replay' to stabilize training by breaking correlation between consecutive samples."
            math={<span>L(<span className="math-serif">&theta;</span>) = E[(r + <span className="math-serif">&gamma;</span> max<sub>a'</sub>Q(s',a'; <span className="math-serif">&theta;</span><sup>-</sup>) - Q(s,a; <span className="math-serif">&theta;</span>))<sup>2</sup>]</span>}
            mathLabel="MSE Loss with Target Network"
            code={`import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inputs, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )
        
    def forward(self, x):
        return self.fc(x) # Returns Q-values for all actions`}
            pros={['Handles high-dimensional inputs (e.g., pixels)', 'Superhuman performance on Atari', 'Sample efficient (via replay)']}
            cons={['Training can be unstable', 'Overestimation bias of Q-values', 'Sensitive to hyperparameters']}
            hyperparameters={[
                { name: 'buffer_size', description: 'Size of experience replay memory.', default: '10000' },
                { name: 'batch_size', description: 'Number of experiences sampled per update.', default: '64' },
                { name: 'gamma', description: 'Discount factor for future rewards.', default: '0.99' }
            ]}
        >
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                    <Database size={16} className="text-indigo-400" />
                    Experience Replay Buffer
                </h4>
                <ReplayBufferViz />
            </div>
        </AlgorithmCard>
      </section>

      {/* 4. POLICY-BASED METHODS */}
      <section id="policy-based" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">04</span>
            <h2 className="text-2xl font-bold text-rose-400 uppercase tracking-widest">Policy-Based Methods</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>

         <AlgorithmCard
            id="policy-gradient"
            title="Policy Gradients (REINFORCE)"
            theory="Instead of learning values (Q), these methods learn the Policy function &pi;(a|s) directly. They adjust the probabilities of actions based on how much reward they yielded. Think of it as 'increasing the likelihood of actions that led to a win'."
            math={<span><span className="math-serif">&nabla;</span>J(<span className="math-serif">&theta;</span>) = E [ <span className="math-serif">&nabla;</span> log <span className="math-serif">&pi;</span>(a|s) G<sub>t</sub> ]</span>}
            mathLabel="Policy Gradient Theorem"
            code={`# REINFORCE pseudocode
for episode in range(max_episodes):
    trajectory = run_episode(policy)
    returns = calculate_discounted_returns(trajectory)
    
    loss = 0
    for log_prob, G in zip(saved_log_probs, returns):
        loss += -log_prob * G # Negative for gradient ascent
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
            pros={['Effective in high-dimensional/continuous action spaces', 'Can learn stochastic policies', 'Convergence to local optimum']}
            cons={['High variance in gradients', 'Sample inefficient (on-policy)', 'Local optima']}
         >
            <div className="h-48 w-full bg-slate-900 border border-slate-800 rounded flex items-center justify-center relative overflow-hidden">
                {/* Visualizing Probability Distribution Shift */}
                <div className="absolute inset-0 flex items-end justify-center gap-1 opacity-20">
                     {[10, 20, 30, 40, 30, 20, 10].map((h, i) => (
                         <div key={i} className="w-8 bg-slate-500" style={{ height: `${h}%` }}></div>
                     ))}
                </div>
                <div className="absolute inset-0 flex items-end justify-center gap-1">
                     {[5, 15, 25, 60, 25, 10, 5].map((h, i) => (
                         <div key={i} className="w-8 bg-rose-500/80 transition-all duration-1000" style={{ height: `${h}%` }}></div>
                     ))}
                </div>
                <div className="absolute top-4 text-xs text-rose-300 font-mono bg-slate-900/80 px-2 py-1 rounded">
                   &pi;(a|s) probability mass shifts towards high-reward actions
                </div>
            </div>
         </AlgorithmCard>
      </section>

      {/* 5. ACTOR-CRITIC */}
      <section id="actor-critic" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-8">
            <span className="text-sm font-mono text-slate-500">05</span>
            <h2 className="text-2xl font-bold text-fuchsia-400 uppercase tracking-widest">Actor-Critic</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
         </div>

         <div className="grid grid-cols-1 md:grid-cols-2 gap-8 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
             <div>
                 <h3 className="text-xl font-bold text-white mb-4">The Best of Both Worlds</h3>
                 <p className="text-slate-400 text-sm leading-relaxed mb-4">
                     Actor-Critic methods combine Policy-Based (Actor) and Value-Based (Critic) approaches.
                 </p>
                 <ul className="space-y-4">
                     <li className="flex gap-4">
                         <div className="w-10 h-10 rounded bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold border border-rose-500/50">A</div>
                         <div>
                             <strong className="text-slate-200 block text-sm">The Actor (Policy)</strong>
                             <span className="text-xs text-slate-500">Decides which action to take. Updates distribution based on Critic's feedback.</span>
                         </div>
                     </li>
                     <li className="flex gap-4">
                         <div className="w-10 h-10 rounded bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold border border-emerald-500/50">C</div>
                         <div>
                             <strong className="text-slate-200 block text-sm">The Critic (Value)</strong>
                             <span className="text-xs text-slate-500">Evaluates the action. Estimates the Value Function V(s) or Q(s,a) to reduce variance.</span>
                         </div>
                     </li>
                 </ul>
             </div>
             
             <ActorCriticViz />
         </div>
      </section>
    </div>
  );
};