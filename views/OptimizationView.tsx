import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { Play, RotateCcw, FastForward } from 'lucide-react';

const data = Array.from({ length: 20 }, (_, i) => ({
  epoch: i,
  loss: 10 * Math.exp(-0.3 * i) + Math.random() * 0.5
}));

// --- SUDOKU LOGIC & VIZ ---

const INITIAL_BOARD = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9]
];

const SudokuViz: React.FC = () => {
  const [board, setBoard] = useState<number[][]>(JSON.parse(JSON.stringify(INITIAL_BOARD)));
  const [solving, setSolving] = useState(false);
  const [speed, setSpeed] = useState(10); // ms delay
  const stopRef = useRef(false);

  const resetBoard = () => {
    stopRef.current = true;
    setSolving(false);
    setBoard(JSON.parse(JSON.stringify(INITIAL_BOARD)));
  };

  const isValid = (board: number[][], row: number, col: number, num: number) => {
    for (let x = 0; x < 9; x++) {
      if (board[row][x] === num || board[x][col] === num) return false;
      const subRow = 3 * Math.floor(row / 3) + Math.floor(x / 3);
      const subCol = 3 * Math.floor(col / 3) + (x % 3);
      if (board[subRow][subCol] === num) return false;
    }
    return true;
  };

  const solve = async () => {
    if (solving) return;
    setSolving(true);
    stopRef.current = false;
    
    const b = JSON.parse(JSON.stringify(board));
    await solveStep(b);
    setSolving(false);
  };

  const solveStep = async (b: number[][]): Promise<boolean> => {
    if (stopRef.current) return false;

    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (b[row][col] === 0) {
          for (let num = 1; num <= 9; num++) {
            if (isValid(b, row, col, num)) {
              b[row][col] = num;
              setBoard([...b.map(r => [...r])]); // Update UI
              
              if (speed > 0) await new Promise(r => setTimeout(r, speed));
              
              if (await solveStep(b)) return true;
              
              b[row][col] = 0; // Backtrack
              setBoard([...b.map(r => [...r])]);
            }
          }
          return false;
        }
      }
    }
    return true;
  };

  return (
    <div className="flex flex-col items-center w-full">
      {/* Responsive Controls */}
      <div className="flex flex-wrap gap-4 mb-6 justify-center w-full">
        <button 
          onClick={solve} 
          disabled={solving}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded font-bold text-sm transition-colors shadow-lg shadow-indigo-900/50"
        >
          <Play size={16} /> Start Backtracking
        </button>
        <button 
          onClick={resetBoard} 
          className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded font-bold text-sm transition-colors border border-slate-600"
        >
          <RotateCcw size={16} /> Reset
        </button>
        <div className="flex items-center gap-2 bg-slate-900 px-3 py-2 rounded border border-slate-800">
           <FastForward size={16} className="text-slate-400" />
           <select 
             value={speed} 
             onChange={(e) => setSpeed(Number(e.target.value))}
             disabled={solving}
             className="bg-transparent text-slate-300 text-xs font-mono outline-none cursor-pointer"
           >
             <option value={100}>Slow</option>
             <option value={10}>Fast</option>
             <option value={0}>Instant</option>
           </select>
        </div>
      </div>

      {/* Responsive Sudoku Grid Container */}
      <div className="w-full max-w-[400px] aspect-square bg-slate-800 p-1 md:p-2 rounded-lg shadow-2xl border border-slate-700">
        <div className="w-full h-full grid grid-cols-9 grid-rows-9 gap-px bg-slate-600 border-2 border-slate-900">
          {board.map((row, rIdx) => (
            row.map((cell, cIdx) => {
              // Calculate borders for 3x3 subgrids
              const borderRight = (cIdx + 1) % 3 === 0 && cIdx !== 8 ? 'border-r-2 border-r-slate-900' : '';
              const borderBottom = (rIdx + 1) % 3 === 0 && rIdx !== 8 ? 'border-b-2 border-b-slate-900' : '';
              
              const isInitial = INITIAL_BOARD[rIdx][cIdx] !== 0;

              return (
                <div 
                  key={`${rIdx}-${cIdx}`}
                  className={`
                    relative flex items-center justify-center 
                    bg-slate-900 
                    ${borderRight} ${borderBottom}
                    hover:bg-slate-800 transition-colors duration-75
                  `}
                >
                  <span className={`
                    font-mono font-bold select-none
                    ${isInitial ? 'text-slate-400' : 'text-indigo-400 animate-pulse'}
                    text-base sm:text-lg md:text-xl lg:text-2xl
                  `}>
                    {cell !== 0 ? cell : ''}
                  </span>
                </div>
              );
            })
          ))}
        </div>
      </div>
      <p className="text-xs text-slate-500 mt-4 text-center max-w-md">
        The algorithm tries a number, moves to the next cell, and <strong className="text-indigo-400">backtracks</strong> (returns to 0) if it hits a dead end.
      </p>
    </div>
  );
};

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
        math={<span>&theta;<sub>new</sub> = <span className="not-italic">&theta;<sub>old</sub></span> - &alpha; &nabla;J(&theta;)</span>}
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

      <AlgorithmCard
        id="constraint-satisfaction"
        title="Constraint Satisfaction (Backtracking)"
        theory="Backtracking is a general algorithm for finding all (or some) solutions to computational problems, notably Constraint Satisfaction Problems (CSPs), that incrementally builds candidates to the solutions, and abandons a candidate ('backtracks') as soon as it determines that the candidate cannot possibly be completed to a valid solution."
        math={<span>Solve(P): if P is full return true; else &forall; v &in; Domain: if valid(v) &rarr; Solve(P+v)</span>}
        mathLabel="Recursive Search Logic"
        code={`def solve(board):
    row, col = find_empty(board)
    if row is None: return True # Solved

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num
            
            if solve(board): return True
            
            board[row][col] = 0 # Backtrack

    return False`}
        pros={['Guaranteed to find a solution (if one exists)', 'More efficient than Brute Force', 'Simple recursive implementation']}
        cons={['Exponential time complexity O(m^n)', 'Can be very slow for large/complex inputs', 'Not an optimization algorithm (just satisfiability)']}
        hyperparameters={[
          {
            name: 'heuristic',
            description: 'Strategy to choose which variable to assign next (e.g., Minimum Remaining Values).',
            default: 'None',
            range: 'MRV, Degree, LCV'
          }
        ]}
      >
        <SudokuViz />
      </AlgorithmCard>
    </div>
  );
};
