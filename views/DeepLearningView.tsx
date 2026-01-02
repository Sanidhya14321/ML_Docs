
import React, { useState, useEffect } from 'react';
import { NeuralNetworkViz } from '../components/NeuralNetworkViz';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, ScatterChart, Scatter, ReferenceLine, LabelList } from 'recharts';

// Data for RNN
const timeSeriesData = Array.from({ length: 20 }, (_, i) => ({
    time: i,
    actual: Math.sin(i * 0.5),
    predicted: Math.sin(i * 0.5 - 0.5) 
}));

// Data for Embeddings Viz
const embeddingData = [
    { x: 2, y: 2, label: 'Man', fill: '#818cf8' },
    { x: 2, y: 6, label: 'Woman', fill: '#f472b6' },
    { x: 6, y: 2, label: 'King', fill: '#818cf8' },
    { x: 6, y: 6, label: 'Queen', fill: '#f472b6' }
];

const ConvolutionViz = () => (
    <div className="flex items-center justify-center gap-4 py-8 select-none">
        <div className="grid grid-cols-4 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 16}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${[5,6,9,10].includes(i) ? 'bg-indigo-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
        <div className="text-slate-500 font-mono text-xl">×</div>
        <div className="grid grid-cols-2 gap-1 p-1 bg-indigo-900 border border-indigo-500">
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
        </div>
        <div className="text-slate-500 font-mono text-xl">=</div>
        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 9}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${i === 4 ? 'bg-emerald-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
    </div>
);

const AttentionViz = () => {
    const words = ['The', 'cat', 'sat', 'on'];
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    // Dynamic attention weights based on hover
    // In a real model, this is computed. Here we simulate the focus.
    const getWeights = (idx: number) => {
        const base = [0.1, 0.1, 0.1, 0.1];
        base[idx] = 0.7; // Self focus
        if (idx === 1) base[2] = 0.2; // Cat -> Sat
        if (idx === 2) base[1] = 0.2; // Sat -> Cat
        return base;
    };

    const currentWeights = hoverIndex !== null ? getWeights(hoverIndex) : [0.5, 0.3, 0.1, 0.1];

    return (
        <div className="flex flex-col items-center py-6 gap-6">
            <div className="flex gap-4">
                {words.map((word, i) => (
                    <div 
                        key={i} 
                        onMouseEnter={() => setHoverIndex(i)}
                        onMouseLeave={() => setHoverIndex(null)}
                        className={`
                            px-3 py-2 rounded-lg border font-mono text-sm cursor-default transition-all duration-300
                            ${hoverIndex === i ? 'bg-indigo-500 border-indigo-400 text-white scale-110' : 'bg-slate-900 border-slate-700 text-slate-400'}
                        `}
                    >
                        {word}
                    </div>
                ))}
            </div>
            
            <div className="flex flex-col items-center gap-2">
                <div className="text-[10px] text-slate-500 font-mono uppercase tracking-[0.2em]">Context Weighting</div>
                <div className="flex gap-4 items-end h-24">
                    {currentWeights.map((w, i) => (
                        <div key={i} className="flex flex-col items-center gap-1">
                            <div 
                                className="w-10 bg-indigo-500/40 border border-indigo-500/60 rounded-t transition-all duration-500 shadow-[0_0_15px_rgba(99,102,241,0.2)]" 
                                style={{ height: `${w * 100}%` }}
                            ></div>
                            <span className="text-[8px] text-slate-600 font-mono">{(w * 100).toFixed(0)}%</span>
                        </div>
                    ))}
                </div>
            </div>
            
            <p className="text-[9px] text-slate-600 font-mono text-center max-w-xs uppercase tracking-widest leading-relaxed">
                {hoverIndex !== null 
                    ? `Showing how the model focuses on other words when processing "${words[hoverIndex]}"`
                    : "Hover over a word to see its attention map"}
            </p>
        </div>
    );
};

const EmbeddingsViz = () => (
    <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" hide domain={[0, 8]} />
                <YAxis type="number" dataKey="y" hide domain={[0, 8]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: '8px' }} />
                
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 2, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" />
                <ReferenceLine segment={[{x: 6, y: 2}, {x: 6, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" />
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 6, y: 2}]} stroke="#475569" strokeDasharray="2 2" />

                <Scatter data={embeddingData} shape="circle">
                    <LabelList dataKey="label" position="top" fill="#cbd5e1" fontSize={11} offset={10} fontWeight="bold" />
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
    </div>
);

const BackpropViz = () => (
    <div className="flex flex-col gap-4 p-5 bg-slate-900/50 rounded-2xl border border-slate-800/50">
        <div className="flex justify-between items-center text-[8px] font-black uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-3">
             <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div> Forward</span>
             <span className="flex items-center gap-1">Backward <div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div></span>
        </div>
        <div className="flex justify-between items-center gap-2 py-2">
            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-indigo-500 flex items-center justify-center bg-indigo-900/20 text-indigo-300 font-bold shadow-lg shadow-indigo-900/20">x</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">In</span>
            </div>
            
            <div className="flex-1 h-[2px] bg-slate-800 relative group overflow-hidden">
                <div className="absolute inset-0 bg-indigo-500/20 animate-pulse"></div>
            </div>

            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-slate-600 flex items-center justify-center bg-slate-800 text-slate-300 font-bold">h</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">Hidden</span>
            </div>

             <div className="flex-1 h-[2px] bg-slate-800 relative group overflow-hidden">
                <div className="absolute inset-0 bg-rose-500/20 animate-pulse"></div>
            </div>

            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full border-2 border-emerald-500 flex items-center justify-center bg-emerald-900/20 text-emerald-300 font-bold">y&#770;</div>
                <span className="text-[8px] mt-2 text-slate-600 font-black uppercase">Pred</span>
            </div>
        </div>
        <div className="bg-slate-950 p-4 rounded-xl text-[10px] font-mono text-slate-500 border border-slate-800 leading-relaxed italic">
            <span className="text-rose-400 font-bold not-italic">Backprop Rule:</span> <br/>
            Chain gradients backwards to update weights using calculus.
        </div>
    </div>
);

const BPTTViz = () => {
    const [phase, setPhase] = useState<'forward' | 'backward'>('forward');
    const [step, setStep] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setStep(prev => {
                if (phase === 'forward') {
                    if (prev >= 3) { 
                        setPhase('backward');
                        return 2;
                    }
                    return prev + 1;
                } else {
                    if (prev <= 0) {
                        setPhase('forward');
                        return 0;
                    }
                    return prev - 1;
                }
            });
        }, 1000);
        return () => clearInterval(interval);
    }, [phase]);

    return (
        <div className="w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-6 relative overflow-hidden">
            <div className="absolute top-3 right-4 text-[9px] font-mono font-black uppercase tracking-widest bg-slate-900 px-2 py-1 rounded border border-slate-800">
                Mode: <span className={phase === 'forward' ? 'text-indigo-400' : 'text-rose-400'}>{phase === 'forward' ? 'Inference' : 'Learning (BPTT)'}</span>
            </div>
            
            <div className="flex justify-around items-center mt-8">
                {[0, 1, 2].map((t) => {
                    const isActive = phase === 'forward' ? step >= t : step <= t;
                    const isGradient = phase === 'backward' && step <= t;

                    return (
                        <div key={t} className="flex flex-col items-center gap-2 relative group">
                             <div className="absolute -top-7 text-[8px] text-slate-600 font-mono font-bold tracking-widest">T={t}</div>
                             <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-all duration-500 ${isGradient ? 'border-rose-500 bg-rose-900/30 text-rose-300 scale-110' : isActive ? 'border-emerald-500 bg-emerald-900/30 text-emerald-300' : 'border-slate-800 bg-slate-900 text-slate-700'}`}>
                                Y
                             </div>
                             <div className={`w-[2px] h-6 transition-colors duration-500 ${isGradient ? 'bg-rose-500' : isActive ? 'bg-slate-700' : 'bg-slate-900'}`}></div>
                             <div className={`w-10 h-10 rounded-xl border-2 flex items-center justify-center text-xs font-bold transition-all duration-500 z-10 ${isGradient ? 'border-rose-500 bg-rose-900/30 text-rose-300 scale-110 shadow-[0_0_20px_rgba(244,63,94,0.3)]' : isActive ? 'border-indigo-500 bg-indigo-900/20 text-indigo-300' : 'border-slate-800 bg-slate-900 text-slate-700'}`}>
                                H
                             </div>
                             <div className={`w-[2px] h-6 transition-colors duration-500 ${isActive ? 'bg-slate-700' : 'bg-slate-900'}`}></div>
                             <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-all duration-500 ${isActive ? 'border-slate-500 bg-slate-800 text-white' : 'border-slate-900 bg-slate-950 text-slate-800'}`}>
                                X
                             </div>
                             {t < 2 && (
                                 <div className="absolute top-[3.5rem] left-9 w-12 h-8 flex items-center justify-center">
                                     <div className={`absolute w-full h-[2px] transition-colors duration-500 ${phase === 'forward' && step > t ? 'bg-indigo-500' : 'bg-slate-900'}`}></div>
                                     <div className={`absolute w-full h-[2px] transition-all duration-500 ${phase === 'backward' && step <= t ? 'bg-rose-500 translate-y-1 opacity-100' : 'opacity-0'}`}></div>
                                 </div>
                             )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export const DeepLearningView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header className="mb-12 border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Deep Learning</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
          Harnessing the power of multi-layered artificial neural networks. Deep learning identifies complex hierarchies of features within massive datasets to solve tasks once thought impossible for machines.
        </p>
      </header>

      <AlgorithmCard
        id="mlp"
        title="Multilayer Perceptrons"
        complexity="Intermediate"
        theory={`The fundamental neural architecture. It stacks fully-connected layers, applying non-linear activation functions (like ReLU) at each stage. Through the backpropagation algorithm, it learns to map raw inputs into sophisticated high-dimensional feature representations.

### Forward Pass
[Input x] -> [Weights W1] -> [ReLU] -> [Hidden h] -> [Weights W2] -> [Softmax] -> [Output y]`}
        math={<span>a<sup>[l]</sup> = &sigma;(W<sup>[l]</sup> a<sup>[l-1]</sup> + b<sup>[l]</sup>)</span>}
        mathLabel="Forward Pass"
        code={`from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])`}
        pros={['Universal function approximator', 'Scales with data', 'Foundational for all deep models']}
        cons={['Requires vast amounts of labeled data', 'Hyperparameter tuning is difficult', 'Difficult to explain decision logic']}
      >
        <NeuralNetworkViz />
        <div className="mt-12">
            <h4 className="text-[10px] font-black text-rose-400 uppercase tracking-[0.3em] mb-6">Learning Process: Backpropagation</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                <p className="text-slate-500 text-sm leading-relaxed font-light">
                    Networks learn by minimizing error. <strong className="text-slate-300">Backpropagation</strong> utilizes the Chain Rule to distribute error gradients from the output layer back through every weight in the network, informing precisely how to nudge each parameter to improve future predictions.
                </p>
                <BackpropViz />
            </div>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="cnn"
        title="Convolutional Networks"
        complexity="Intermediate"
        theory={`Specialized for grid-structured data like images. CNNs use convolutional kernels to extract spatial features while preserving local relationships. They are highly efficient due to parameter sharing and spatial pooling mechanisms.

### CNN Pipeline
[Image] -> [Conv] -> [ReLU] -> [Pool] -> [Conv] -> [ReLU] -> [Pool] -> [FC] -> [Class]`}
        math={<span>(I * K)(i, j) = &Sigma; &Sigma; I(m, n) K(i-m, j-n)</span>}
        mathLabel="Convolution Operation"
        code={`model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])`}
        pros={['Translation invariance', 'Automatic hierarchical feature extraction', 'Highly efficient for visual data']}
        cons={['High compute requirement', 'Loss of fine-grained spatial information through pooling', 'Adversarial vulnerability']}
      >
        <ConvolutionViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="rnn"
        title="Recurrent Networks (LSTM)"
        complexity="Advanced"
        theory={`Designed for sequential data processing. RNNs maintain an internal hidden state that acts as memory across time-steps. LSTMs (Long Short-Term Memory) refine this with specialized 'gates' that control information flow, solving long-range dependency issues.

### Unrolled RNN
h(t-1) --> [Cell A] --> h(t) --> [Cell A] --> h(t+1)
              ^                    ^
              |                    |
            x(t)                 x(t+1)`}
        math={<span>h<sub>t</sub> = &sigma;(W<sub>hh</sub> h<sub>t-1</sub> + W<sub>xh</sub> x<sub>t</sub>)</span>}
        mathLabel="Hidden Update"
        code={`model = models.Sequential([
    layers.LSTM(128, input_shape=(None, 10))
])`}
        pros={['Handles variable-length inputs', 'Captures temporal dependencies', 'Standard for audio/time-series']}
        cons={['Slow to train (non-parallelizable)', 'Vanishing gradient issues in vanilla versions', 'Memory limitations for very long sequences']}
      >
         <BPTTViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="embeddings"
        title="Embeddings"
        complexity="Intermediate"
        theory={`Learned dense representations of discrete items. Unlike one-hot encoding, embeddings place similar items closer together in a continuous vector space, allowing models to learn semantic relationships mathematically.

### Word2Vec Concept
"King" - "Man" + "Woman" ≈ "Queen"`}
        math={<span>L = E[ log P(w<sub>context</sub> | w<sub>target</sub>) ]</span>}
        mathLabel="Word2Vec Objective"
        code={`from tensorflow.keras.layers import Embedding
layer = Embedding(input_dim=10000, output_dim=300)`}
        pros={['Dramatic dimensionality reduction', 'Captures semantic meaning', 'Transferable via pre-trained models']}
        cons={['Learns biases from training corpora', 'Static embeddings ignore context (e.g., "bank")']}
      >
        <EmbeddingsViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="transformers"
        title="Transformers"
        complexity="Advanced"
        theory={`The modern standard for large-scale modeling. It utilizes a Self-Attention mechanism to draw global dependencies regardless of distance in the sequence. This enables massive parallelization and state-of-the-art performance in LLMs.

### Transformer Block
[Input] -> [Attention] -> [Add & Norm] -> [Feed Forward] -> [Add & Norm] -> [Output]`}
        math={<span>Attn(Q, K, V) = softmax(<sup>QK<sup>T</sup></sup>&frasl;<sub>&radic;d<sub>k</sub></sub>)V</span>}
        mathLabel="Scaled Dot-Product Attention"
        code={`# Attention mechanism logic
def attention(q, k, v):
    scores = matmul(q, k.T) / sqrt(dk)
    return matmul(softmax(scores), v)`}
        pros={['Massive parallel training capability', 'State-of-the-art across nearly all NLP benchmarks', 'Excellent long-range memory']}
        cons={['Quadratic compute cost with sequence length', 'Extreme data appetite', 'Complex implementation']}
      >
        <AttentionViz />
      </AlgorithmCard>
    </div>
  );
};
