import React from 'react';
import { NeuralNetworkViz } from '../components/NeuralNetworkViz';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ScatterChart, Scatter, ReferenceLine, LabelList } from 'recharts';

// Data for RNN
const timeSeriesData = Array.from({ length: 20 }, (_, i) => ({
    time: i,
    actual: Math.sin(i * 0.5),
    predicted: Math.sin(i * 0.5 - 0.5) // lagged prediction
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
        {/* Input Grid */}
        <div className="grid grid-cols-4 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 16}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${[5,6,9,10].includes(i) ? 'bg-indigo-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
        <div className="text-slate-500 font-mono text-xl">×</div>
        {/* Kernel */}
        <div className="grid grid-cols-2 gap-1 p-1 bg-indigo-900 border border-indigo-500">
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">0</div>
            <div className="w-4 h-4 md:w-6 md:h-6 bg-white/20 text-[8px] flex items-center justify-center">1</div>
        </div>
        <div className="text-slate-500 font-mono text-xl">=</div>
        {/* Output Grid */}
        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-800 border border-slate-700">
             {Array.from({length: 9}).map((_,i) => (
                 <div key={i} className={`w-4 h-4 md:w-6 md:h-6 ${i === 4 ? 'bg-emerald-500' : 'bg-slate-700'}`}></div>
             ))}
        </div>
    </div>
);

const AttentionViz = () => (
    <div className="flex flex-col items-center py-4">
        <div className="flex gap-2 mb-2">
            {['The', 'cat', 'sat', 'on'].map((word, i) => (
                <div key={i} className="text-xs text-slate-400 w-8 text-center">{word}</div>
            ))}
        </div>
        <div className="grid grid-cols-4 gap-1">
            {/* Heatmap Grid */}
            {[
                1.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0,
                0.0, 0.2, 0.7, 0.1,
                0.0, 0.0, 0.1, 0.9
            ].map((val, i) => (
                <div key={i} className="w-8 h-8 bg-indigo-500 border border-slate-900" style={{ opacity: val }}></div>
            ))}
        </div>
        <div className="flex gap-2 mt-2">
             <div className="text-xs text-slate-500">Self-Attention Matrix</div>
        </div>
    </div>
);

const EmbeddingsViz = () => (
    <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" hide domain={[0, 8]} />
                <YAxis type="number" dataKey="y" hide domain={[0, 8]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                
                {/* Vectors */}
                {/* Man -> Woman */}
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 2, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" markerEnd="url(#arrow)" />
                {/* King -> Queen */}
                <ReferenceLine segment={[{x: 6, y: 2}, {x: 6, y: 6}]} stroke="#94a3b8" strokeDasharray="3 3" markerEnd="url(#arrow)" />
                {/* Man -> King */}
                <ReferenceLine segment={[{x: 2, y: 2}, {x: 6, y: 2}]} stroke="#475569" strokeDasharray="2 2" />

                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
                    </marker>
                </defs>

                <Scatter data={embeddingData} shape="circle">
                    <LabelList dataKey="label" position="top" fill="#cbd5e1" fontSize={12} offset={10} />
                    {embeddingData.map((entry, index) => (
                         <ReferenceLine key={index} /> // Dummy to fix type issues if needed, mostly handled by Scatter
                    ))}
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-center text-slate-500 mt-2">
            Vector Arithmetic: <em>King - Man + Woman &approx; Queen</em>. Semantic relationships are preserved as geometric distances.
        </p>
    </div>
);

export const DeepLearningView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Deep Learning</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          A subset of machine learning based on artificial neural networks with representation learning. It allows computational models composed of multiple processing layers to learn representations of data with multiple levels of abstraction.
        </p>
      </header>

      <AlgorithmCard
        id="mlp"
        title="Multilayer Perceptrons (MLP)"
        theory="The classical neural network consisting of an input layer, one or more hidden layers, and an output layer. Nodes are fully connected. It learns non-linear function approximations via backpropagation."
        math={<span>a<sup>[l]</sup> = &sigma;(W<sup>[l]</sup> a<sup>[l-1]</sup> + b<sup>[l]</sup>)</span>}
        mathLabel="Forward Propagation"
        code={`from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])`}
        pros={['Universal function approximator', 'Handles non-linear data', 'Flexible architecture']}
        cons={['Black box nature', 'Requires large data', 'Prone to overfitting']}
        hyperparameters={[
          {
            name: 'hidden_layer_sizes',
            description: 'The ith element represents the number of neurons in the ith hidden layer.',
            default: '(100,)',
            range: 'Tuple of integers'
          },
          {
            name: 'activation',
            description: 'Activation function for the hidden layer.',
            default: 'relu',
            range: 'identity, logistic, tanh, relu'
          },
          {
            name: 'solver',
            description: 'The solver for weight optimization.',
            default: 'adam',
            range: 'lbfgs, sgd, adam'
          }
        ]}
      >
        <NeuralNetworkViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="cnn"
        title="Convolutional Neural Networks (CNN)"
        theory="Specialized for processing grid-like data (e.g., images). Uses Convolutional layers to apply filters (kernels) that automatically learn spatial hierarchies of features, followed by Pooling layers to downsample."
        math={<span>(I * K)(i, j) = &Sigma;<sub>m</sub> &Sigma;<sub>n</sub> I(m, n) K(i-m, j-n)</span>}
        mathLabel="Convolution Operation"
        code={`model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])`}
        pros={['Parameter sharing (efficient)', 'Translation invariance', 'State-of-the-art for vision']}
        cons={['Requires fixed input size', 'Computationally heavy training', 'Loss of spatial resolution (pooling)']}
        hyperparameters={[
          {
            name: 'filters',
            description: 'Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).',
            default: '32',
            range: 'Integer'
          },
          {
            name: 'kernel_size',
            description: 'An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.',
            default: '(3, 3)',
            range: 'Integer or Tuple'
          },
          {
            name: 'strides',
            description: 'An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.',
            default: '(1, 1)',
            range: 'Integer or Tuple'
          }
        ]}
      >
        <ConvolutionViz />
        <p className="text-xs text-center text-slate-500 mt-2">Filter sliding over input feature map to create output map.</p>
      </AlgorithmCard>

      <AlgorithmCard
        id="rnn"
        title="Recurrent Neural Networks (RNN/LSTM)"
        theory="Designed for sequential data. They have 'memory' that captures information about what has been calculated so far. LSTMs (Long Short-Term Memory) solve the vanishing gradient problem of standard RNNs."
        math={<span>h<sub>t</sub> = &sigma;(W<sub>hh</sub> h<sub>t-1</sub> + W<sub>xh</sub> x<sub>t</sub>)</span>}
        mathLabel="Hidden State Update"
        code={`model = models.Sequential([
    layers.LSTM(128, input_shape=(None, 10)),
    layers.Dense(1)
])`}
        pros={['Handles variable length sequences', 'Captures temporal dependencies', 'Good for NLP/Time-series']}
        cons={['Slow to train (sequential)', 'Vanishing gradient (Vanilla RNN)', 'Short-term memory limitations']}
        hyperparameters={[
          {
            name: 'units',
            description: 'Positive integer, dimensionality of the output space (hidden state size).',
            default: '50',
            range: 'Integer'
          },
          {
            name: 'return_sequences',
            description: 'Whether to return the last output in the output sequence, or the full sequence.',
            default: 'False',
            range: 'True / False'
          },
          {
            name: 'dropout',
            description: 'Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.',
            default: '0.0',
            range: '[0, 1)'
          }
        ]}
      >
         <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
               <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" hide />
                  <YAxis hide />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                  <Line type="monotone" dataKey="actual" stroke="#818cf8" strokeWidth={2} dot={false} name="Actual" />
                  <Line type="monotone" dataKey="predicted" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Predicted" />
               </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-center text-slate-500 mt-2">Sequence Prediction: Learning time-series patterns.</p>
         </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="embeddings"
        title="Embeddings"
        theory="Embeddings are dense vector representations of discrete variables (like words or users). Unlike one-hot encoding, embeddings capture semantic relationships—similar items are closer in the vector space."
        math={<span>J(&theta;) = <sup>1</sup>&frasl;<sub>T</sub> &Sigma;<sub>t=1</sub><sup>T</sup> &Sigma;<sub>-c &le; j &le; c, j &ne; 0</sub> log p(w<sub>t+j</sub> | w<sub>t</sub>)</span>}
        mathLabel="Skip-Gram Objective (Word2Vec)"
        code={`from tensorflow.keras.layers import Embedding

# Input: Integers (indices), Output: Vectors
# Vocab size: 10,000, Vector dim: 300
embedding_layer = Embedding(input_dim=10000, output_dim=300)

# Lookup vector for word index 5
vector = embedding_layer(5)`}
        pros={['Captures semantic meaning', 'Reduces dimensionality compared to one-hot', 'Transferable (pre-trained embeddings)']}
        cons={['Requires large datasets to learn good representations', 'Static (traditional embeddings ignore context)', 'Bias amplification']}
        hyperparameters={[
          {
            name: 'output_dim',
            description: 'Dimension of the dense embedding. Higher dimension captures more nuances but requires more data.',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'input_dim',
            description: 'Size of the vocabulary, i.e., maximum integer index + 1.',
            default: 'None',
            range: 'Integer'
          }
        ]}
      >
        <EmbeddingsViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="transformers"
        title="Transformers"
        theory="The modern architecture for NLP. Abandoning recurrence, it relies entirely on an attention mechanism (Self-Attention) to draw global dependencies between input and output. It enables massive parallelization."
        math={<span>Attention(Q, K, V) = softmax(<sup>QK<sup>T</sup></sup>&frasl;<sub>&radic;d<sub>k</sub></sub>)V</span>}
        mathLabel="Scaled Dot-Product Attention"
        code={`# Pseudocode for a Transformer Block
def transformer_block(x):
    attn = MultiHeadAttention(x, x, x)
    x = LayerNorm(x + attn)
    ffn = FeedForward(x)
    return LayerNorm(x + ffn)`}
        pros={['Parallel training', 'Captures long-range dependencies', 'Foundation of LLMs (GPT/BERT)']}
        cons={['Quadratic complexity with sequence length', 'Data hungry', 'High compute requirements']}
        hyperparameters={[
          {
            name: 'num_heads',
            description: 'Number of attention heads. Allows the model to jointly attend to information from different representation subspaces.',
            default: '8',
            range: 'Integer'
          },
          {
            name: 'd_model',
            description: 'The dimension of the embedding vector. Typically 512, 768, etc.',
            default: '512',
            range: 'Integer'
          },
          {
            name: 'num_layers',
            description: 'Number of encoder/decoder layers (blocks) in the stack.',
            default: '6',
            range: 'Integer'
          }
        ]}
      >
        <AttentionViz />
      </AlgorithmCard>
    </div>
  );
};