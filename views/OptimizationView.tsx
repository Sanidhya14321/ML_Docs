import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

const data = Array.from({ length: 20 }, (_, i) => ({
  epoch: i,
  loss: 10 * Math.exp(-0.3 * i) + Math.random() * 0.5
}));

export const OptimizationView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header>
        <h1 className="text-4xl font-serif font-bold text-white mb-2">Optimization</h1>
        <p className="text-slate-400 text-lg">The engine of modern machine learning that iteratively minimizes error.</p>
      </header>

      <AlgorithmCard
        id="gradient-descent"
        title="Gradient Descent"
        theory="Gradient Descent is an iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point."
        math={<span>&theta;<sub>new</sub> = &theta;<sub>old</sub> - &alpha; &nabla;J(&theta;)</span>}
        mathLabel="Update Rule"
        code={`def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1/m) * learning_rate * (X.T.dot(prediction - y))
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history`}
        pros={['Simple to implement', 'Computationally efficient for simple convex problems', 'Basis for advanced optimizers (Adam, RMSprop)']}
        cons={['Can get stuck in local minima', 'Sensitive to learning rate choice', 'Slow convergence near minimum']}
        hyperparameters={[
          {
            name: 'learning_rate (alpha)',
            description: 'Determines the step size at each iteration while moving toward a minimum of a loss function. Too small: slow convergence. Too large: overshooting.',
            default: '0.01',
            range: '(0, 1)'
          },
          {
            name: 'epochs',
            description: 'The number of times the algorithm will run through the entire training dataset.',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'batch_size',
            description: 'Number of training examples used in one iteration. 1 = Stochastic GD, M = Mini-batch, N = Batch GD.',
            default: '32',
            range: 'Integer'
          }
        ]}
      >
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b' }} />
                <YAxis stroke="#94a3b8" label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }}
                  itemStyle={{ color: '#818cf8' }}
                />
                <Line type="monotone" dataKey="loss" stroke="#818cf8" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-center text-slate-500 mt-2">Loss decreasing over epochs.</p>
        </div>
      </AlgorithmCard>
    </div>
  );
};