import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceDot, ReferenceLine } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

// Generate clustered data (Keep existing)
const generateCluster = (cx: number, cy: number, count: number, spread: number, clusterName: string) => {
  return Array.from({ length: count }, (_, i) => ({
    x: cx + (Math.random() - 0.5) * spread,
    y: cy + (Math.random() - 0.5) * spread,
    cluster: clusterName
  }));
};

const cluster1 = generateCluster(20, 20, 20, 15, 'Cluster A');
const cluster2 = generateCluster(70, 60, 20, 20, 'Cluster B');
const cluster3 = generateCluster(30, 80, 20, 15, 'Cluster C');

const centroids = [
  { x: 20, y: 20, label: 'Centroid A' },
  { x: 70, y: 60, label: 'Centroid B' },
  { x: 30, y: 80, label: 'Centroid C' }
];

// Data for DBSCAN
const dbscanCore = generateCluster(50, 50, 30, 20, 'Core');
const dbscanNoise = [
    { x: 10, y: 90 }, { x: 90, y: 10 }, { x: 5, y: 5 }, { x: 95, y: 95 }
];

// Data for PCA
const pcaData = Array.from({ length: 30 }, (_, i) => ({
    x: i * 2 + Math.random() * 10,
    y: i * 1.5 + Math.random() * 10
}));

const DendrogramViz = () => (
  <div className="h-64 w-full flex flex-col items-center justify-center bg-slate-900 overflow-hidden relative rounded-lg border border-slate-800 p-4">
     <div className="absolute top-3 right-4 text-xs text-slate-500 font-mono tracking-wider">AGGLOMERATIVE LINKAGE</div>
     <svg width="100%" height="100%" viewBox="0 0 400 240" className="stroke-indigo-400 stroke-2 w-full max-w-lg">
        {/* Labels */}
        <text x="60" y="230" fill="#94a3b8" textAnchor="middle" fontSize="12" stroke="none">P1</text>
        <text x="100" y="230" fill="#94a3b8" textAnchor="middle" fontSize="12" stroke="none">P2</text>
        <text x="140" y="230" fill="#94a3b8" textAnchor="middle" fontSize="12" stroke="none">P3</text>
        <text x="220" y="230" fill="#94a3b8" textAnchor="middle" fontSize="12" stroke="none">P4</text>
        <text x="260" y="230" fill="#94a3b8" textAnchor="middle" fontSize="12" stroke="none">P5</text>

        {/* Cluster 1 (P1, P2) -> Mid 80, Y 180 */}
        <path d="M60,210 V180 H100 V210" fill="none" />
        
        {/* Cluster 2 (P4, P5) -> Mid 240, Y 160 */}
        <path d="M220,210 V160 H260 V210" fill="none" />

        {/* Cluster 3 (P3 + (P1P2)) -> Mid (80+140)/2 = 110, Y 120 */}
        {/* P1P2 stem at 80,180. P3 stem at 140,210 */}
        <path d="M80,180 V120 H140 V210" fill="none" />

        {/* Cluster 4 ( (P3P1P2) + (P4P5) ) -> Mid (110 + 240)/2 = 175, Y 60 */}
        {/* Left stem at 110,120. Right stem at 240,160 */}
        <path d="M110,120 V60 H240 V160" fill="none" />

        {/* Dashed Threshold Line */}
        <line x1="20" y1="90" x2="380" y2="90" stroke="#ef4444" strokeDasharray="6 4" strokeWidth="1" opacity="0.8" />
        <text x="30" y="85" fill="#ef4444" fontSize="12" stroke="none" fontWeight="bold">Cut Threshold (k=2)</text>
     </svg>
     <div className="text-xs text-slate-500 mt-2">Dendrogram showing hierarchical merging order</div>
  </div>
);

const TSNEVisualizer: React.FC = () => {
  const [perplexity, setPerplexity] = useState(30);

  const data = useMemo(() => {
    // Simulation parameters based on perplexity (5 to 50)
    // Low perp: Tight clusters, spread out centroids (simulating local focus)
    // High perp: Loose clusters, centroids pull in (simulating global averaging)
    
    // Normalize p from 0 (perp 5) to 1 (perp 50)
    const p = (perplexity - 5) / 45; 
    
    // Spread increases with perplexity (simulating "loose" relations)
    // Range 3 to 20
    const spread = 3 + (p * 17);
    
    // Lerp function for centroid movement
    const lerp = (start: number, end: number, t: number) => start * (1 - t) + end * t;
    
    // Centroids move slightly to simulate the structural change
    const c1 = { x: lerp(10, 30, p), y: lerp(50, 50, p) };
    const c2 = { x: lerp(50, 50, p), y: lerp(10, 30, p) };
    const c3 = { x: lerp(90, 70, p), y: lerp(90, 70, p) };
    
    const blob1 = generateCluster(c1.x, c1.y, 25, spread, 'Group 1');
    const blob2 = generateCluster(c2.x, c2.y, 25, spread, 'Group 2');
    const blob3 = generateCluster(c3.x, c3.y, 25, spread, 'Group 3');
    
    return { blob1, blob2, blob3 };
  }, [perplexity]);

  return (
    <div className="flex flex-col gap-4">
       <div className="h-64 w-full bg-slate-900 rounded-lg border border-slate-800 p-2">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                 <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                 <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                 <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                 <Scatter name="Group 1" data={data.blob1} fill="#818cf8" />
                 <Scatter name="Group 2" data={data.blob2} fill="#f472b6" />
                 <Scatter name="Group 3" data={data.blob3} fill="#fbbf24" />
              </ScatterChart>
            </ResponsiveContainer>
       </div>
       
       <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
          <div className="flex justify-between items-center mb-2">
            <label htmlFor="perp-slider" className="text-sm font-bold text-slate-300">Perplexity: <span className="text-indigo-400">{perplexity}</span></label>
            <span className="text-xs font-mono px-2 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">
                {perplexity < 15 ? "Local (Tight)" : perplexity > 40 ? "Global (Loose)" : "Balanced"}
            </span>
          </div>
          <input 
            id="perp-slider"
            type="range" 
            min="5" 
            max="50" 
            step="1"
            value={perplexity} 
            onChange={(e) => setPerplexity(Number(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
          <p className="text-xs text-slate-500 mt-3 leading-relaxed">
            <strong className="text-slate-400">Interactive:</strong> Adjusting perplexity changes the balance between local and global aspects of the data. 
            Low values cause the algorithm to focus on the closest neighbors (tight clumps), while high values consider a broader context (smoother, more spread out).
          </p>
       </div>
    </div>
  );
};

export const UnsupervisedView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Unsupervised Learning</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Unsupervised learning algorithms find hidden patterns or intrinsic structures in data without labeled responses. Common tasks include clustering and dimensionality reduction.
        </p>
      </header>

      <AlgorithmCard
        id="k-means"
        title="K-Means Clustering"
        theory="Partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean (centroid). It iteratively minimizes the within-cluster sum of squares."
        math={<span>J = &Sigma;<sub>i=1</sub><sup>k</sup> &Sigma;<sub>x &isin; S<sub>i</sub></sub> || x - &mu;<sub>i</sub> ||<sup>2</sup></span>}
        mathLabel="Inertia (Within-Cluster Sum of Squares)"
        code={`from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_`}
        pros={['Scales to large datasets', 'Simple to implement', 'Guaranteed convergence']}
        cons={['Must specify k manually', 'Sensitive to initialization', 'Assumes spherical clusters']}
        hyperparameters={[
          {
            name: 'n_clusters',
            description: 'The number of clusters to form as well as the number of centroids to generate.',
            default: '8',
            range: 'Integer'
          },
          {
            name: 'init',
            description: 'Method for initialization. k-means++ selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.',
            default: 'k-means++',
            range: 'k-means++, random'
          },
          {
            name: 'n_init',
            description: 'Number of time the k-means algorithm will be run with different centroid seeds.',
            default: '10',
            range: 'Integer'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <Legend />
              
              {/* Cluster Boundaries (Visual Hints) */}
              <ReferenceDot x={20} y={20} r={50} fill="#818cf8" fillOpacity={0.15} stroke="none" />
              <ReferenceDot x={70} y={60} r={55} fill="#34d399" fillOpacity={0.15} stroke="none" />
              <ReferenceDot x={30} y={80} r={50} fill="#f472b6" fillOpacity={0.15} stroke="none" />

              <Scatter name="Cluster A" data={cluster1} fill="#818cf8" shape="circle" />
              <Scatter name="Cluster B" data={cluster2} fill="#34d399" shape="circle" />
              <Scatter name="Cluster C" data={cluster3} fill="#f472b6" shape="circle" />
              
              {/* Centroids */}
              <Scatter name="Centroids" data={centroids} fill="#ffffff" shape="cross" legendType="cross" />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">
            Points clustered around white Cross centroids. Transparent zones indicate boundaries.
          </p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="hierarchical"
        title="Hierarchical Clustering"
        theory="Builds a hierarchy of clusters. Agglomerative clustering starts with each point as a cluster and merges pairs as it moves up the hierarchy. The result is often visualized as a Dendrogram."
        math={<span>d(u, v) = min(dist(u[i], v[j]))</span>}
        mathLabel="Linkage Criteria (Single Linkage Example)"
        code={`from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3)
labels = hc.fit_predict(X)`}
        pros={['No need to specify k initially', 'Produces a dendrogram', 'Captures hierarchical structure']}
        cons={['High time complexity O(n³)', 'Sensitive to noise', 'Hard to handle large datasets']}
        hyperparameters={[
          {
            name: 'n_clusters',
            description: 'The number of clusters to find. It must be None if distance_threshold is not None.',
            default: '2',
            range: 'Integer or None'
          },
          {
            name: 'linkage',
            description: 'Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.',
            default: 'ward',
            range: 'ward, complete, average, single'
          },
          {
            name: 'metric',
            description: 'Metric used to compute the linkage.',
            default: 'euclidean',
            range: 'euclidean, l1, l2, manhattan, cosine'
          }
        ]}
      >
        <DendrogramViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="dbscan"
        title="DBSCAN"
        theory="Density-Based Spatial Clustering of Applications with Noise. It groups together points that are closely packed together (points with many neighbors), marking as outliers points that lie alone in low-density regions."
        math={<span>N<sub>&epsilon;</sub>(p) = {`{q | dist(p,q) ≤ &epsilon;}`}</span>}
        mathLabel="Epsilon Neighborhood"
        code={`from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
labels = db.fit_predict(X)`}
        pros={['Finds arbitrarily shaped clusters', 'Robust to outliers', 'No need to specify k']}
        cons={['Struggles with varying densities', 'Sensitive to parameters eps and min_samples']}
        hyperparameters={[
          {
            name: 'eps',
            description: 'The maximum distance between two samples for one to be considered as in the neighborhood of the other.',
            default: '0.5',
            range: 'Float'
          },
          {
            name: 'min_samples',
            description: 'The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.',
            default: '5',
            range: 'Integer'
          }
        ]}
      >
        <div className="h-64 w-full">
           <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <Scatter name="Core Points" data={dbscanCore} fill="#34d399" />
              <Scatter name="Noise/Outliers" data={dbscanNoise} fill="#ef4444" shape="cross" />
              {/* Radius hints */}
              <ReferenceDot x={50} y={50} r={20} fill="#34d399" fillOpacity={0.1} stroke="none" />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">
            Dense green cluster (Core) vs Red crosses (Noise/Outliers) in sparse regions.
          </p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="pca"
        title="Principal Component Analysis (PCA)"
        theory="A dimensionality reduction technique that computes the principal components (eigenvectors) of the data and projects the data onto a lower-dimensional space while maximizing variance."
        math={<span>X = T P<sup>T</sup></span>}
        mathLabel="Decomposition (Scores * Loadings)"
        code={`from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_original)`}
        pros={['Removes correlated features', 'Improves algorithm speed', 'Visualizes high-dim data']}
        cons={['Loss of interpretability', 'Linear transformation only', 'Sensitive to scaling']}
        hyperparameters={[
          {
            name: 'n_components',
            description: 'Number of components to keep. if n_components is not set all components are kept.',
            default: 'None',
            range: 'Integer or Float (0 < x < 1)'
          },
          {
            name: 'svd_solver',
            description: 'The solver to use for the singular value decomposition.',
            default: 'auto',
            range: 'auto, full, arpack, randomized'
          }
        ]}
      >
         <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="x" stroke="#94a3b8" />
                <YAxis type="number" dataKey="y" stroke="#94a3b8" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                <Scatter name="Data" data={pcaData} fill="#94a3b8" opacity={0.5} />
                {/* Vectors */}
                <ReferenceLine segment={[{x: 0, y: 0}, {x: 60, y: 45}]} stroke="#fbbf24" strokeWidth={3} label={{value: "PC1", fill: "#fbbf24", position: "insideBottomRight"}}/>
                <ReferenceLine segment={[{x: 30, y: 22.5}, {x: 20, y: 40}]} stroke="#818cf8" strokeWidth={2} label={{value: "PC2", fill: "#818cf8", position: "top"}} />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-xs text-center text-slate-500 mt-2">
              Yellow line (PC1) captures maximum variance. Blue line (PC2) is orthogonal.
            </p>
         </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="tsne"
        title="t-SNE"
        theory="t-Distributed Stochastic Neighbor Embedding. A non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets."
        math={<span>C = KL(P||Q) = &Sigma; p<sub>ij</sub> log(p<sub>ij</sub>/q<sub>ij</sub>)</span>}
        mathLabel="Kullback-Leibler Divergence"
        code={`from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)`}
        pros={['Preserves local structure', 'Excellent for visualization', 'Non-linear']}
        cons={['Computationally expensive', 'Non-deterministic', 'Cluster sizes/distances can be misleading']}
        hyperparameters={[
          {
            name: 'perplexity',
            description: 'The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.',
            default: '30.0',
            range: '5 - 50'
          },
          {
            name: 'n_iter',
            description: 'Maximum number of iterations for the optimization.',
            default: '1000',
            range: 'Integer > 250'
          },
          {
            name: 'learning_rate',
            description: 'The learning rate for t-SNE is usually in the range [10.0, 1000.0].',
            default: 'auto',
            range: 'Float'
          }
        ]}
      >
        <TSNEVisualizer />
      </AlgorithmCard>
    </div>
  );
};