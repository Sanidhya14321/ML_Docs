import React from 'react';

export const NeuralNetworkViz: React.FC = () => {
  // Simple representation of layers
  const inputNodes = [1, 2, 3];
  const hiddenNodes = [1, 2, 3, 4];
  const outputNodes = [1, 2];

  return (
    <div className="w-full h-64 bg-slate-900 rounded-lg border border-slate-700 flex items-center justify-around relative overflow-hidden p-8">
      {/* Input Layer */}
      <div className="flex flex-col justify-center space-y-6 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono">Input</div>
        {inputNodes.map((n) => (
          <div key={`in-${n}`} className="w-8 h-8 rounded-full bg-indigo-500 border-2 border-white shadow-[0_0_15px_rgba(99,102,241,0.5)] z-10" />
        ))}
      </div>

      {/* Hidden Layer */}
      <div className="flex flex-col justify-center space-y-4 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono">Hidden</div>
        {hiddenNodes.map((n) => (
          <div key={`hid-${n}`} className="w-8 h-8 rounded-full bg-slate-700 border-2 border-indigo-400 z-10" />
        ))}
      </div>

      {/* Output Layer */}
      <div className="flex flex-col justify-center space-y-8 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono">Output</div>
        {outputNodes.map((n) => (
          <div key={`out-${n}`} className="w-8 h-8 rounded-full bg-emerald-500 border-2 border-white shadow-[0_0_15px_rgba(16,185,129,0.5)] z-10" />
        ))}
      </div>

      {/* CSS Connectivity Lines (Simulated with absolute background generic lines) */}
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        {/* We use a repeating gradient to simulate connections without SVG */}
        <div className="w-full h-full bg-[linear-gradient(45deg,transparent_49%,#fff_50%,transparent_51%)] bg-[length:20px_20px] opacity-10"></div>
      </div>
    </div>
  );
};