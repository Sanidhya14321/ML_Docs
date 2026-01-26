
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { Play, RotateCcw, Cpu, AlertCircle } from 'lucide-react';

const data = Array.from({ length: 20 }, (_, i) => ({
  epoch: i,
  loss: 10 * Math.exp(-0.3 * i) + Math.random() * 0.5
}));

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
  const [speed, setSpeed] = useState(10);
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef(false);

  const resetBoard = () => {
    stopRef.current = true;
    setSolving(false);
    setError(null);
    setBoard(JSON.parse(JSON.stringify(INITIAL_BOARD)));
  };

  const isValid = (board: number[][], row: number, col: number, num: number) => {
    for (let x = 0; x < 9; x++) {
      // Check row and column
      if (board[row][x] === num && x !== col) return false;
      if (board[x][col] === num && x !== row) return false;
      
      // Check subgrid
      const subRow = 3 * Math.floor(row / 3) + Math.floor(x / 3);
      const subCol = 3 * Math.floor(col / 3) + (x % 3);
      if (board[subRow][subCol] === num && (subRow !== row || subCol !== col)) return false;
    }
    return true;
  };

  const checkBoardValidity = (b: number[][]) => {
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        if (b[r][c] !== 0) {
          if (!isValid(b, r, c, b[r][c])) return false;
        }
      }
    }
    return true;
  };

  const solve = async () => {
    if (solving) return;
    setError(null);
    setSolving(true);
    stopRef.current = false;

    try {
      const b = JSON.parse(JSON.stringify(board));
      
      if (!checkBoardValidity(b)) {
        throw new Error("Board state is invalid (duplicates found).");
      }

      const success = await solveStep(b);
      if (!success && !stopRef.current) {
         setError("No solution exists for this configuration.");
      }
    } catch (e: any) {
      console.error(e);
      setError(e.message || "An unexpected error occurred.");
      stopRef.current = true;
    } finally {
      setSolving(false);
    }
  };

  const solveStep = async (b: number[][]): Promise<boolean> => {
    if (stopRef.current) return false;
    
    // Find empty cell
    let row = -1;
    let col = -1;
    let isEmpty = false;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        if (b[r][c] === 0) {
          row = r;
          col = c;
          isEmpty = true;
          break;
        }
      }
      if (isEmpty) break;
    }

    // If no empty cell, we are done
    if (!isEmpty) return true;

    for (let num = 1; num <= 9; num++) {
      if (isValid(b, row, col, num)) {
        b[row][col] = num;
        setBoard([...b.map(r => [...r])]);
        
        if (speed > 0) await new Promise(r => setTimeout(r, speed));
        
        if (await solveStep(b)) return true;
        
        // Backtrack
        if (stopRef.current) return false;
        b[row][col] = 0;
        setBoard([...b.map(r => [...r])]);
      }
    }
    return false;
  };

  return (
    <div className="flex flex-col items-center w-full">
      <div className="flex flex-wrap gap-4 mb-6 justify-center w-full">
        <button onClick={solve} disabled={solving} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg font-bold text-sm transition-all shadow-lg shadow-indigo-900/50"><Play size={14} /> Start Solver</button>
        <button onClick={resetBoard} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-bold text-sm transition-all border border-slate-700"><RotateCcw size={14} /> Reset</button>
      </div>
      
      {error && (
        <div className="mb-4 px-4 py-2 bg-rose-500/10 border border-rose-500/20 rounded-lg flex items-center gap-2 text-rose-400 text-xs font-bold animate-pulse">
           <AlertCircle size={14} /> {error}
        </div>
      )}

      <div className={`w-full max-w-[360px] aspect-square bg-slate-900 p-2 rounded-2xl border ${error ? 'border-rose-500/50' : 'border-slate-800'} shadow-2xl transition-colors duration-300`}>
        <div className="w-full h-full grid grid-cols-9 grid-rows-9 gap-px bg-slate-800">
          {board.map((row, rIdx) => row.map((cell, cIdx) => (
            <div key={`${rIdx}-${cIdx}`} className={`relative flex items-center justify-center bg-slate-950 ${(cIdx + 1) % 3 === 0 && cIdx !== 8 ? 'border-r-2 border-r-slate-800' : ''} ${(rIdx + 1) % 3 === 0 && rIdx !== 8 ? 'border-b-2 border-b-slate-800' : ''}`}>
              <span className={`font-mono text-sm font-bold ${INITIAL_BOARD[rIdx][cIdx] !== 0 ? 'text-slate-600' : 'text-indigo-400'}`}>{cell !== 0 ? cell : ''}</span>
            </div>
          )))}
        </div>
      </div>
    </div>
  );
};

export const OptimizationView: React.FC = () => {
  return (
    <div className="space-y-16">
      <header id="optimization-header" className="border-b border-slate-800 pb-12">
        <h1 className="text-6xl font-serif font-bold text-white mb-6">Optimization</h1>
        <p className="text-slate-400 text-xl font-light leading-relaxed max-w-2xl">
          The mathematical core of machine learning. Optimization algorithms find the minimal error state, transforming "training" into a rigorous search for truth.
        </p>
      </header>

      <section id="calculus-engine">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 rounded-xl bg-indigo-600/10 flex items-center justify-center text-indigo-400">
             <Cpu size={20} />
          </div>
          <h2 id="gradient-descent-title" className="text-3xl font-bold text-white tracking-tight">The Descent Engine</h2>
        </div>
        
        <AlgorithmCard
          id="gradient-descent"
          title="Gradient Descent"
          complexity="Fundamental"
          theory="A first-order iterative optimization algorithm for finding the minimum of a cost function. It moves downhill in the direction of the steepest decrease—calculated via the negative of the gradient."
          math={<span>&theta;<sub>t+1</sub> = &theta;<sub>t</sub> - &eta; &nabla;J(&theta;<sub>t</sub>)</span>}
          mathLabel="Parameter Update Vector"
          code={`def update_weights(w, g, lr):
    return w - lr * g`}
          pros={['Globally optimal for convex spaces', 'Scalable to millions of parameters', 'The engine of modern AI']}
          cons={['Sensitive to learning rate', 'Can oscillate in ravines', 'Vanishing gradients in deep paths']}
          hyperparameters={[
            { name: 'Learning Rate (η)', description: 'Controls the step size at each iteration. Large values can overshoot the minimum; small values converge slowly.', default: '0.01' },
            { name: 'Iterations (Epochs)', description: 'The number of passes through the entire training dataset.', default: '1000' }
          ]}
          steps={[
            "Initialize a Google Colab notebook for the experiment.",
            "Define your cost function J(θ) (e.g., Mean Squared Error) and its gradient ∇J(θ).",
            "Initialize parameters θ randomly (e.g., weights and bias