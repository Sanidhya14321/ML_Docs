
import React from 'react';
import { DocMetadata } from '../../types';
import { MathBlock } from '../../components/MathBlock';
import { LatexRenderer } from '../../components/LatexRenderer';
import { CodeBlock } from '../../components/CodeBlock';
import { AlgorithmCard } from '../../components/AlgorithmCard';
import { Callout } from '../../components/Callout';

export const metadata: DocMetadata = {
  title: "The Attention Mechanism",
  description: "Breaking down the core mathematical operation that powered the Transformer revolution and modern LLMs.",
  date: "2023-10-27",
  difficulty: "Advanced",
  tags: ["NLP", "Transformers", "Deep Learning", "Math"],
  readTimeMinutes: 15
};

export const Content: React.FC = () => {
  return (
    <div className="space-y-8">
      <section>
        <p className="text-lg text-slate-300 leading-relaxed">
          Before Transformers, sequence modeling relied heavily on Recurrent Neural Networks (RNNs). 
          However, RNNs suffered from <span className="text-indigo-400 font-bold">sequential processing limitations</span> and 
          forgot long-range dependencies. The Attention Mechanism solved this by allowing the model to "look at" the entire sequence at once.
        </p>

        <Callout type="tip" title="Why 'Attention'?">
           Imagine reading a sentence. When you see the word "bank", you need context to know if it means a "river bank" or "financial bank". Attention mechanisms mathematically calculate which other words in the sentence (context) are most relevant to understanding the current word.
        </Callout>
      </section>

      <section>
        <h2 id="query-key-value" className="text-2xl font-bold text-white mb-4">Query, Key, and Value</h2>
        <p className="text-slate-400 mb-6">
          The fundamental intuition is a retrieval system. For every word (Query), we check its compatibility with every other word (Key) to decide how much info to pull (Value).
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {['Query (Q)', 'Key (K)', 'Value (V)'].map((term, i) => (
            <div key={i} className="bg-slate-900 p-6 rounded-xl border border-slate-800 text-center hover:border-indigo-500/50 transition-colors cursor-default group">
              <div className="text-xl font-mono font-bold text-indigo-400 mb-2 group-hover:scale-110 transition-transform">{term}</div>
              <div className="text-xs text-slate-500 uppercase tracking-widest">Vector Representation</div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 id="math-definition" className="text-2xl font-bold text-white mb-4">Scaled Dot-Product Attention</h2>
        <p className="text-slate-400 mb-4">
          The core equation computes a weighted sum of values, where weights are determined by the dot product of Query and Key vectors.
        </p>

        <MathBlock label="Attention Formula">
          <div className="flex items-center justify-center gap-2">
            <span>Attention(Q, K, V) = softmax</span>
            <div className="flex flex-col items-center justify-center mx-2">
              <div className="border-b border-slate-500 pb-1 mb-1">
                 <LatexRenderer formula="QK^T" />
              </div>
              <LatexRenderer formula="\sqrt{d_k}" />
            </div>
            <span>V</span>
          </div>
        </MathBlock>

        <Callout type="warning" title="Gradient Stability">
          We divide by <LatexRenderer formula="\sqrt{d_k}" /> to prevent the dot products from growing too large in magnitude. Large values push the Softmax function into regions with extremely small gradients (vanishing gradients), which kills training.
        </Callout>
      </section>

      <section>
        <h2 id="implementation" className="text-2xl font-bold text-white mb-4">PyTorch Implementation</h2>
        <CodeBlock 
          filename="self_attention.py"
          language="python"
          code={`import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights`} 
        />
      </section>
    </div>
  );
};
