
import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { LatexRenderer } from '../components/LatexRenderer';
import { MOTION_VARIANTS } from '../constants';
import { 
  ResponsiveContainer, 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  CartesianGrid, 
  Tooltip, 
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts';
import { Brain, Target, Zap, Layers, Activity, Award } from 'lucide-react';

// --- VISUALIZATIONS ---

const SupervisedViz = () => {
    const data = [
        { x: 10, y: 30, class: 'A' }, { x: 20, y: 50, class: 'A' }, { x: 15, y: 40, class: 'A' },
        { x: 25, y: 60, class: 'A' }, { x: 30, y: 70, class: 'A' }, { x: 35, y: 80, class: 'A' },
        { x: 60, y: 20, class: 'B' }, { x: 70, y: 30, class: 'B' }, { x: 80, y: 40, class: 'B' },
        { x: 65, y: 15, class: 'B' }, { x: 75, y: 25, class: 'B' }, { x: 85, y: 35, class: 'B' },
    ];

    return (
        <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-4 relative overflow-hidden">
            <div className="absolute inset-0 opacity-20">
                <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-indigo-500/20 via-transparent to-rose-500/20" />
                <div className="absolute top-1/2 left-0 w-full h-px bg-slate-800 -rotate-45 transform origin-center scale-150" />
            </div>
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" name="Feature 1" hide />
                    <YAxis type="number" dataKey="y" name="Feature 2" hide />
                    <ZAxis type="number" range={[100, 100]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                    <Scatter name="Data Points" data={data}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.class === 'A' ? '#6366f1' : '#f43f5e'} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
            <div className="absolute bottom-4 left-4 flex gap-4">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-indigo-500" />
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Class A (Labeled)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-rose-500" />
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Class B (Labeled)</span>
                </div>
            </div>
        </div>
    );
};

const UnsupervisedViz = () => {
    const [step, setStep] = useState(0);
    
    const clusters = useMemo(() => {
        if (step === 0) {
            return [
                { x: 20, y: 20 }, { x: 25, y: 25 }, { x: 15, y: 30 },
                { x: 70, y: 70 }, { x: 75, y: 75 }, { x: 80, y: 65 },
                { x: 20, y: 70 }, { x: 25, y: 75 }, { x: 15, y: 80 },
            ].map(p => ({ ...p, cluster: 'none' }));
        }
        return [
            { x: 20, y: 20, cluster: 0 }, { x: 25, y: 25, cluster: 0 }, { x: 15, y: 30, cluster: 0 },
            { x: 70, y: 70, cluster: 1 }, { x: 75, y: 75, cluster: 1 }, { x: 80, y: 65, cluster: 1 },
            { x: 20, y: 70, cluster: 2 }, { x: 25, y: 75, cluster: 2 }, { x: 15, y: 80, cluster: 2 },
        ];
    }, [step]);

    return (
        <div className="space-y-4">
            <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-4 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" hide />
                        <YAxis type="number" dataKey="y" hide />
                        <ZAxis type="number" range={[100, 100]} />
                        <Scatter data={clusters}>
                            {clusters.map((entry, index) => (
                                <Cell 
                                    key={`cell-${index}`} 
                                    fill={entry.cluster === 'none' ? '#475569' : ['#10b981', '#f59e0b', '#8b5cf6'][entry.cluster as number]} 
                                />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
                <button 
                    onClick={() => setStep(step === 0 ? 1 : 0)}
                    className="absolute top-4 right-4 px-3 py-1 bg-slate-800 hover:bg-slate-700 text-white text-[10px] font-bold rounded-lg transition-colors uppercase tracking-widest"
                >
                    {step === 0 ? 'Find Patterns' : 'Reset'}
                </button>
            </div>
            <p className="text-[10px] text-center text-slate-500 uppercase tracking-widest font-mono">
                Unsupervised learning discovers hidden structures in unlabeled data.
            </p>
        </div>
    );
};

const ReinforcementViz = () => {
    const [score, setScore] = useState(0);
    const [pos, setPos] = useState({ x: 0, y: 0 });
    
    const moveAgent = () => {
        const nextX = Math.min(3, Math.max(0, pos.x + (Math.random() > 0.5 ? 1 : -1)));
        const nextY = Math.min(3, Math.max(0, pos.y + (Math.random() > 0.5 ? 1 : -1)));
        setPos({ x: nextX, y: nextY });
        if (nextX === 3 && nextY === 3) {
            setScore(s => s + 10);
            setTimeout(() => setPos({ x: 0, y: 0 }), 500);
        } else {
            setScore(s => Math.max(0, s - 1));
        }
    };

    return (
        <div className="flex flex-col md:flex-row gap-8 items-center">
            <div className="grid grid-cols-4 gap-2 bg-slate-900 p-4 rounded-2xl border border-slate-800">
                {Array.from({ length: 16 }).map((_, i) => {
                    const x = i % 4;
                    const y = Math.floor(i / 4);
                    const isAgent = pos.x === x && pos.y === y;
                    const isGoal = x === 3 && y === 3;
                    return (
                        <div 
                            key={i} 
                            className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${
                                isAgent ? 'bg-indigo-500 shadow-lg shadow-indigo-500/50 scale-110' : 
                                isGoal ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-500' : 
                                'bg-slate-800 border border-slate-700'
                            }`}
                        >
                            {isAgent && <Brain size={16} className="text-white animate-pulse" />}
                            {isGoal && !isAgent && <Target size={16} />}
                        </div>
                    );
                })}
            </div>
            <div className="flex-1 space-y-4 text-center md:text-left">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase font-mono mb-1">Cumulative Reward</div>
                    <div className="text-3xl font-bold text-white font-mono">{score}</div>
                </div>
                <button 
                    onClick={moveAgent}
                    className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2"
                >
                    <Zap size={16} />
                    Take Action
                </button>
                <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono leading-relaxed">
                    RL agents learn by interacting with an environment to maximize rewards.
                </p>
            </div>
        </div>
    );
};

// --- MAIN VIEW ---

export const MLParadigmsView: React.FC = () => {
  return (
    <motion.div 
      variants={MOTION_VARIANTS.container}
      initial="hidden"
      animate="show"
      className="space-y-24 pb-20"
    >
      <motion.header variants={MOTION_VARIANTS.item} className="border-b border-slate-800 pb-12">
        <div className="flex items-center gap-4 mb-6">
            <div className="p-3 rounded-2xl bg-brand/10 border border-brand/20 text-brand">
                <Brain size={32} />
            </div>
            <h1 className="text-6xl font-serif font-bold text-white">ML Paradigms</h1>
        </div>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed font-light">
          Understanding the three fundamental ways machines learn from data, experience, and feedback loops.
        </p>
      </motion.header>

      {/* SECTION 1: SUPERVISED LEARNING */}
      <motion.section variants={MOTION_VARIANTS.item} id="supervised" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">01. Supervised Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="supervised-learning" title="Learning with a Teacher" complexity="Fundamental"
              theory="Supervised learning is the most common paradigm. The algorithm is trained on a labeled dataset, where each input is paired with the correct output. The goal is to learn a mapping function that can predict outputs for new, unseen data."
              math={<LatexRenderer formula="y = f(x; \theta) + \epsilon" />} mathLabel="General Mapping Function"
              code={`from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train) # Training with labels
predictions = model.predict(X_test)`}
              pros={['Highly predictable results', 'Clear evaluation metrics', 'Wide range of proven algorithms']}
              cons={['Requires expensive labeled data', 'Prone to overfitting', 'Limited by the quality of labels']}
              steps={[
                "Collect a dataset with known ground truth (labels).",
                "Split data into training and testing sets.",
                "Choose a model (e.g., Linear Regression, SVM, Neural Net).",
                "Train the model to minimize the error between predictions and true labels.",
                "Evaluate performance on the test set using metrics like Accuracy or MSE."
              ]}
          >
              <SupervisedViz />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 2: UNSUPERVISED LEARNING */}
      <motion.section variants={MOTION_VARIANTS.item} id="unsupervised" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">02. Unsupervised Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="unsupervised-learning" title="Discovering Hidden Structure" complexity="Intermediate"
              theory="In unsupervised learning, the algorithm works with unlabeled data. There is no 'teacher' providing the correct answers. Instead, the model tries to find inherent patterns, structures, or groupings within the data itself."
              math={<LatexRenderer formula="J = \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2" />} mathLabel="K-Means Objective Function"
              code={`from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X) # No labels provided`}
              pros={['No labeling cost', 'Discovers unexpected patterns', 'Great for data exploration']}
              cons={['Hard to evaluate performance', 'Results can be ambiguous', 'Computationally intensive for large datasets']}
              steps={[
                "Gather raw, unlabeled data.",
                "Perform feature scaling and normalization.",
                "Select a clustering or dimensionality reduction algorithm.",
                "Run the algorithm to identify groups or latent features.",
                "Interpret the results to gain insights into the data's structure."
              ]}
          >
              <UnsupervisedViz />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 3: REINFORCEMENT LEARNING */}
      <motion.section variants={MOTION_VARIANTS.item} id="reinforcement" className="scroll-mt-24">
        <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">03. Reinforcement Learning</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        <AlgorithmCard
              id="reinforcement-learning" title="Learning through Interaction" complexity="Advanced"
              theory="Reinforcement Learning (RL) is about an agent taking actions in an environment to maximize a cumulative reward. It's a trial-and-error process where the agent learns which actions lead to the best long-term outcomes."
              math={<LatexRenderer formula="Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]" />} mathLabel="Q-Learning Update Rule"
              code={`import gym
env = gym.make('CartPole-v1')
state = env.reset()
for _ in range(1000):
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.update(state, action, reward, next_state)`}
              pros={['Can solve complex sequential tasks', 'No dataset required (learns from environment)', 'Mimics human learning']}
              cons={['Extremely sample inefficient', 'Unstable training process', 'Requires a well-defined reward function']}
              steps={[
                "Define the Environment and the Agent.",
                "Specify the State space and Action space.",
                "Design a Reward function to guide the agent.",
                "Let the agent interact with the environment (Exploration vs. Exploitation).",
                "Update the agent's policy based on the rewards received."
              ]}
          >
              <ReinforcementViz />
          </AlgorithmCard>
      </motion.section>

      {/* SECTION 4: COMPARISON & QUIZ */}
      <motion.section variants={MOTION_VARIANTS.item} id="comparison" className="scroll-mt-24">
         <div className="flex items-center gap-3 mb-10">
            <h2 className="text-3xl font-bold text-white tracking-tight">04. Knowledge Check</h2>
            <div className="h-px bg-slate-800 flex-1"></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {[
                { title: 'Supervised', icon: Target, desc: 'Predicting labels from features.', color: 'text-indigo-400' },
                { title: 'Unsupervised', icon: Layers, desc: 'Finding patterns in raw data.', color: 'text-emerald-400' },
                { title: 'Reinforcement', icon: Zap, desc: 'Learning through trial and error.', color: 'text-rose-400' },
            ].map((item, i) => (
                <div key={i} className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl hover:border-slate-700 transition-colors">
                    <item.icon className={`mb-4 ${item.color}`} size={24} />
                    <h3 className="text-lg font-bold text-white mb-2">{item.title}</h3>
                    <p className="text-xs text-slate-500 leading-relaxed">{item.desc}</p>
                </div>
            ))}
        </div>

        <div className="bg-brand/5 border border-brand/20 rounded-3xl p-8 md:p-12 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
                <Award size={120} className="text-brand" />
            </div>
            <div className="relative z-10 max-w-2xl">
                <h3 className="text-2xl font-bold text-white mb-4">Paradigm Checkpoint</h3>
                <p className="text-slate-400 mb-8">
                    Ready to test your understanding of these fundamental learning styles? Complete the module quiz to earn your progress badge.
                </p>
                <div className="flex flex-wrap gap-4">
                    <button 
                        onClick={() => window.location.hash = '#/quiz-ml-paradigms'}
                        className="px-6 py-3 bg-brand text-white font-bold rounded-xl hover:bg-brand/90 transition-all shadow-lg shadow-brand/20"
                    >
                        Start Quiz
                    </button>
                    <button 
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="px-6 py-3 bg-slate-800 text-white font-bold rounded-xl hover:bg-slate-700 transition-all"
                    >
                        Review Theory
                    </button>
                </div>
            </div>
        </div>
      </motion.section>
    </motion.div>
  );
};
