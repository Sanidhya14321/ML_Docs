import React from 'react';
import { AlgorithmCard } from '../components/AlgorithmCard';

const GridWorldViz = () => (
    <div className="flex flex-col items-center justify-center p-4">
        <div className="grid grid-cols-4 gap-2 mb-2">
            {/* Row 1 */}
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center font-mono text-slate-500">S</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.1</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.2</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.3</div>
            
            {/* Row 2 */}
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.1</div>
            <div className="w-12 h-12 bg-slate-900 border border-slate-800 flex items-center justify-center">🧱</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.4</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-rose-400">-1</div>
            
            {/* Row 3 */}
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.1</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.2</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.5</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.8</div>

            {/* Row 4 */}
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.0</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.0</div>
            <div className="w-12 h-12 bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-indigo-400">0.0</div>
            <div className="w-12 h-12 bg-emerald-900/40 border border-emerald-500 flex items-center justify-center font-bold text-emerald-400">G</div>
        </div>
        <div className="text-xs text-slate-500 text-center">
            Agent learns Q-values (numbers) to navigate from Start (S) to Goal (G) avoiding Walls (🧱) and Traps (-1).
        </div>
    </div>
);

export const ReinforcementView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Reinforcement Learning</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          A type of machine learning where an agent learns to make decisions by performing actions in an environment and receiving rewards or penalties.
        </p>
      </header>

      <AlgorithmCard
        id="q-learning"
        title="Q-Learning"
        theory="A model-free reinforcement learning algorithm to learn the value of an action in a particular state. It learns a policy that finds the best action to take given the current state to maximize the total expected reward."
        math={<span>Q(s,a) &larr; Q(s,a) + &alpha; [r + &gamma; max<sub>a'</sub> Q(s',a') - Q(s,a)]</span>}
        mathLabel="Bellman Equation Update"
        code={`import numpy as np
# Initialize Q-table
Q = np.zeros([state_space_size, action_space_size])

# Update loop
Q[state, action] = (1 - alpha) * Q[state, action] + \
    alpha * (reward + gamma * np.max(Q[next_state]))`}
        pros={['Model-free (no environment knowledge needed)', 'Guaranteed convergence to optimal policy', 'Simple to implement']}
        cons={['Slow convergence', 'Q-Table scales poorly with state space', 'Exploration vs Exploitation tradeoff']}
        hyperparameters={[
          {
            name: 'alpha (learning_rate)',
            description: 'Determines to what extent newly acquired information overrides old information.',
            default: '0.1',
            range: '(0, 1]'
          },
          {
            name: 'gamma (discount_factor)',
            description: 'Determines the importance of future rewards. A factor of 0 will make the agent opportunistic.',
            default: '0.99',
            range: '[0, 1]'
          },
          {
            name: 'epsilon (exploration_rate)',
            description: 'Probability of choosing a random action (explore) instead of the best known action (exploit).',
            default: '0.1',
            range: '[0, 1]'
          }
        ]}
      >
        <GridWorldViz />
      </AlgorithmCard>
    </div>
  );
};