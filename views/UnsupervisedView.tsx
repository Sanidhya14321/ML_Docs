
import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceDot, ReferenceLine, ComposedChart, Line } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

// Generate clustered data
const generateCluster = (cx: number, cy: number, count: number, spread: number, clusterName: string) => {
  return Array.from({ length: count }, (_, i) => ({
    x: cx + (Math.random() - 0.5) * spread,
    y: cy + (Math.random() - 0.5) * spread,
    cluster: clusterName
  }));
};

const cluster1 = generateCluster(20, 20, 15, 12, 'Cluster A');
const cluster2 = generateCluster(70, 60, 15, 15, 'Cluster B');
const cluster3 = generateCluster(30, 80, 15, 12, 'Cluster C');

const centroidsData = [
  { x: 20, y: 20, label: 'Centroid A', fill: '#818cf8' },
  { x: 70, y: 60, label: 'Centroid B', fill: '#34d399' },
  { x: 30, y: 80, label: 'Centroid C', fill: '#f472b6' }
];

const KMeansViz = () => {
    return (
        <div className="space-y-4">
            <div className="h-72 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                        <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                        
                        {/* Assignment Lines */}
                        {[...cluster1, ...cluster2, ...cluster3].map((p, i) => {
                            const centroid = p.cluster === 'Cluster A' ? centroidsData[0] : p.cluster === 'Cluster B' ? centroidsData[1] : centroidsData[2];
                            return (
                                <Line 
                                    key={i} 
                                    data={[{x: p.x, y: p.y}, {x: centroid.x, y: centroid.y}]} 
                                    dataKey="y" 
                                    stroke={centroid.fill} 
                                    strokeOpacity={0.15} 
                                    dot={false} 
                                    activeDot={false} 
                                    animationDuration={0}
                                />
                            );
                        })}

                        <Scatter name="Points" data={[...cluster1, ...cluster2, ...cluster3]} fill="#475569" shape="circle" fillOpacity={0.6} />
                        <Scatter name="Centroids" data={centroidsData} fill="#ffffff" shape="cross" strokeWidth={2} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-center text-slate-500 uppercase tracking-widest font-mono">Each point is assigned to its nearest white cross (Centroid)</p>
        </div>
    );
};

const DendrogramViz = () => (
  <div className="h-64 w-full flex flex-col items-center justify-center bg-slate-950 overflow-hidden relative rounded-2xl border border-slate-800/50 p-6">
     <div className="absolute top-3 right-4 text-[8px] text-slate-600 font-mono tracking-widest uppercase">Agglomerative Linkage</div>
     <svg width="100%" height="100%" viewBox="0 0 400 240" className="stroke-indigo-500 stroke-2 w-full max-w-lg">
        <text x="60" y="230" fill="#475569" textAnchor="middle" fontSize="10" stroke="none">P1</text>
        <text x="100" y="230" fill="#475569" textAnchor="middle" fontSize="10" stroke="none">P2</text>
        <text x="140" y="230" fill="#475569" textAnchor="middle" fontSize="10" stroke="none">P3</text>
        <text x="220" y="230" fill="#475569" textAnchor="middle" fontSize="10" stroke="none">P4</text>
        <text x="260" y="230" fill="#475569" textAnchor="middle" fontSize="10" stroke="none">P5</text>
        <path d="M60,210 V180 H100 V210" fill="none" />
        <path d="M220,210 V160 H260 V210" fill="none" />
        <path d="M80,180 V120 H140 V210" fill="none" />
        <path d="M110,120 V60 H240 V160" fill="none" />
        <line x1="20" y1="90" x2="380" y2="90" stroke="#f43f5e" strokeDasharray="6 4" strokeWidth="1" opacity="0.6" />
        <text x="30" y="85" fill="#f43f5e" fontSize="10" stroke="none" fontWeight="bold" className="uppercase tracking-tighter">Threshold (k=2)</text>
     </svg>
  </div>
);

const TSNEVisualizer: React.FC = () => {
  const [perplexity, setPerplexity] = useState(30);
  const data = useMemo(() => {
    const p = (perplexity - 5) / 45; 
    const spread = 3 + (p * 17);
    const lerp = (start: number, end: number, t: number) => start * (1 - t) + end * t;
    const c1 = { x: lerp(10, 30, p), y: lerp(50, 50, p) };
    const c2 = { x: lerp(50, 50, p), y: lerp(10, 30, p) };
    const c3 = { x: lerp(90, 70, p), y: lerp(90, 70, p) };
    return {
        b1: generateCluster(c1.x, c1.y, 25, spread, 'G1'),
        b2: generateCluster(c2.x, c2.y, 25, spread, 'G2'),
        b3: generateCluster(c3.x, c3.y, 25, spread, 'G3')
    };
  }, [perplexity]);

  return (
    <div className="space-y-4">
       <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                 <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                 <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                 <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569' }} />
                 <Scatter name="Group 1" data={data.b1} fill="#818cf8" />
                 <Scatter name="Group 2" data={data.b2} fill="#f472b6" />
                 <Scatter name="Group 3" data={data.b3} fill="#fbbf24" />
              </ScatterChart>
            </ResponsiveContainer>
       </div>
       <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
          <div className="flex justify-between items-center mb-2">
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Perplexity: <span className="text-indigo-400 text-sm ml-2">{perplexity}</span></label>
            <span className="text-[8px] font-mono px-2 py-1 rounded bg-slate-950 text-slate-500 border border-slate-800 uppercase">
                {perplexity < 15 ? "Local Focus" : perplexity > 40 ? "Global Focus" : "Balanced"}
            </span>
          </div>
          <input type="range" min="5" max="50" step="1" value={perplexity} onChange={(e) => setPerplexity(Number(e.target.value))} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
       </div>
    </div>
  );
};

export const UnsupervisedView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header className="mb-12 border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Unsupervised Learning</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
          Extracting structure from noise. Unsupervised algorithms organize data without the guidance of explicit labels, identifying clusters, dimensions, and latent distributions.
        </p>
      </header>

      <AlgorithmCard
        id="k-means"
        title="K-Means Clustering"
        complexity="Fundamental"
        theory="Partitions data into 'k' distinct clusters. It iteratively assigns each point to its closest centroid and then recalculates centroids based on assigned points until convergence."
        math={<span>J = &Sigma; &Sigma; || x - &mu;<sub>i</sub> ||<sup>2</sup></span>}
        mathLabel="Inertia"
        code={`from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3).fit(X)`}
        pros={['Extremely fast and simple', 'Scales to massive datasets', 'Consistent performance']}
        cons={['Requires manual k selection', 'Sensitive to initial seeds', 'Assumes spherical cluster shapes']}
      >
        <KMeansViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="hierarchical"
        title="Hierarchical Clustering"
        complexity="Intermediate"
        theory="Creates a nested hierarchy of clusters. Agglomerative (bottom-up) approach starts with individual points and merges them into larger clusters based on proximity."
        math={<span>d(u, v) = min(dist(u, v))</span>}
        mathLabel="Linkage Logic"
        code={`from sklearn.cluster import AgglomerativeClustering\nhc = AgglomerativeClustering(n_clusters=3).fit(X)`}
        pros={['No need to specify k up-front', 'Produces interpretable dendrograms', 'Captures nested structures']}
        cons={['High compute cost O(n³)', 'Sensitive to noise and outliers']}
      >
        <DendrogramViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="tsne"
        title="t-SNE"
        complexity="Advanced"
        theory="A non-linear dimensionality reduction technique. It maps high-dimensional data into 2D or 3D, preserving local neighbor structures for exploratory visualization."
        math={<span>C = &Sigma; p<sub>ij</sub> log(p<sub>ij</sub>/q<sub>ij</sub>)</span>}
        mathLabel="KL Divergence Loss"
        code={`from sklearn.manifold import TSNE\nvis = TSNE(n_components=2).fit_transform(X)`}
        pros={['Unmatched visualization of clusters', 'Captures non-linear relationships']}
        cons={['Non-deterministic output', 'Distances can be misleading', 'Slow on large data']}
      >
        <TSNEVisualizer />
      </AlgorithmCard>
    </div>
  );
};
