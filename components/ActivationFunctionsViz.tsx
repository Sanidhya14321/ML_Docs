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
        <div className="flex flex-col md:flex-row gap-8 bg-app border border-border-strong p-8 relative overflow-hidden">
            <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'linear-gradient(#1e293b 1px, transparent 1px)', backgroundSize: '100% 30px' }} />
            
            <div className="w-full md:w-1/3 flex flex-col gap-3 relative z-10">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-4">SELECT_FUNCTION</label>
                {(Object.keys(functions) as Array<keyof typeof functions>).map((key) => (
                    <button
                        key={key}
                        onClick={() => setActiveFunc(key)}
                        className={`px-4 py-3 border font-mono text-[10px] font-black uppercase tracking-widest text-left transition-all ${activeFunc === key ? 'bg-brand border-brand text-app shadow-[0_0_20px_rgba(16,185,129,0.2)]' : 'bg-surface border-border-strong text-text-muted hover:text-brand hover:border-brand'}`}
                    >
                        {functions[key].name}
                    </button>
                ))}
                
                <div className="mt-8 p-4 bg-surface border border-border-strong">
                    <p className="text-[10px] font-mono text-text-muted leading-relaxed uppercase tracking-widest">
                        {current.desc}
                    </p>
                </div>
            </div>

            <div className="flex-1 h-80 bg-surface/30 border border-border-strong relative z-10">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 30, right: 30, bottom: 30, left: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-strong)" />
                        <XAxis dataKey="x" type="number" stroke="var(--text-muted)" fontSize={9} tickCount={7} fontFamily="monospace" fontWeight="900" />
                        <YAxis stroke="var(--text-muted)" fontSize={9} fontFamily="monospace" fontWeight="900" />
                        <Tooltip contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', fontSize: '9px', fontFamily: 'monospace', fontWeight: '900', borderRadius: '0' }} itemStyle={{ color: 'var(--brand)' }} />
                        <ReferenceLine x={0} stroke="var(--border-strong)" />
                        <ReferenceLine y={0} stroke="var(--border-strong)" />
                        <Line type="monotone" dataKey="y" stroke="var(--brand)" strokeWidth={3} dot={false} animationDuration={500} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
