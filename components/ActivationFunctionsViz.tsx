import React, { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';

const generateData = (func: (x: number) => number, range: [number, number]) => {
    const data = [];
    const step = (range[1] - range[0]) / 50;
    for (let x = range[0]; x <= range[1]; x += step) {
        data.push({ x: parseFloat(x.toFixed(2)), y: func(x) });
    }
    return data;
};

const functions = {
    sigmoid: {
        name: 'Sigmoid',
        fn: (x: number) => 1 / (1 + Math.exp(-x)),
        range: [-6, 6] as [number, number],
        desc: 'S-shaped curve. Maps input to (0, 1). Used for probabilities.',
        color: '#818cf8'
    },
    tanh: {
        name: 'Tanh',
        fn: (x: number) => Math.tanh(x),
        range: [-6, 6] as [number, number],
        desc: 'Hyperbolic Tangent. Maps input to (-1, 1). Zero-centered.',
        color: '#2dd4bf'
    },
    relu: {
        name: 'ReLU',
        fn: (x: number) => Math.max(0, x),
        range: [-6, 6] as [number, number],
        desc: 'Rectified Linear Unit. f(x) = max(0, x). Solves vanishing gradient.',
        color: '#f472b6'
    },
    leakyRelu: {
        name: 'Leaky ReLU',
        fn: (x: number) => x > 0 ? x : 0.1 * x,
        range: [-6, 6] as [number, number],
        desc: 'Allows a small gradient when x < 0 to prevent "dead neurons".',
        color: '#fbbf24'
    }
};

export const ActivationFunctionsViz: React.FC = () => {
    const [activeFunc, setActiveFunc] = useState<keyof typeof functions>('sigmoid');
    const current = functions[activeFunc];
    const data = generateData(current.fn, current.range);

    return (
        <div className="flex flex-col md:flex-row gap-6 bg-slate-950 rounded-2xl border border-slate-800/50 p-6">
            <div className="w-full md:w-1/3 flex flex-col gap-2">
                <label className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-2">Select Function</label>
                {(Object.keys(functions) as Array<keyof typeof functions>).map((key) => (
                    <button
                        key={key}
                        onClick={() => setActiveFunc(key)}
                        className={`px-4 py-3 rounded-lg text-left text-sm font-bold transition-all border ${activeFunc === key ? 'bg-slate-800 border-indigo-500 text-white shadow-lg shadow-indigo-500/10' : 'bg-transparent border-transparent text-slate-500 hover:bg-slate-900 hover:text-slate-300'}`}
                    >
                        {functions[key].name}
                    </button>
                ))}
                
                <div className="mt-6 p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                    <p className="text-xs text-slate-400 leading-relaxed">
                        {current.desc}
                    </p>
                </div>
            </div>

            <div className="flex-1 h-64 bg-slate-900/30 rounded-xl border border-slate-800/50 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="x" type="number" stroke="#475569" fontSize={10} tickCount={7} />
                        <YAxis stroke="#475569" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '10px' }} itemStyle={{ color: current.color }} />
                        <ReferenceLine x={0} stroke="#334155" />
                        <ReferenceLine y={0} stroke="#334155" />
                        <Line type="monotone" dataKey="y" stroke={current.color} strokeWidth={3} dot={false} animationDuration={500} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
