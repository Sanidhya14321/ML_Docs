
import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ComposedChart, Line } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { Network, ArrowRight } from 'lucide-react';

// --- HELPERS ---

const generateBlobs = (count: number, centers: {x: number, y: number}[], spread: number) => {
    const points = [];
    for (let i = 0; i < count; i++) {
        const center = centers[Math.floor(Math.random() * centers.length)];
        points.push({
            id: i,
            x: center.x + (Math.random() - 0.5) * spread,
            y: center.y + (Math.random() - 0.5) * spread
        });
    }
    return points;
};

const generateMoons = (count: number, noise: number) => {
    const points = [];
    for (let i = 0; i < count / 2; i++) {
        // Top moon
        const r = 30 + Math.random() * 5;
        const theta = Math.PI * (i / (count/2));
        points.push({
            x: 50 + r * Math.cos(theta) + (Math.random() - 0.5) * noise,
            y: 35 + r * Math.sin(theta) + (Math.random() - 0.5) * noise
        });
        // Bottom moon
        const r2 = 30 + Math.random() * 5;
        const theta2 = Math.PI + Math.PI * (i / (count/2));
        points.push({
            x: 65 + r2 * Math.cos(theta2) + (Math.random() - 0.5) * noise,
            y: 65 + r2 * Math.sin(theta2) + (Math.random() - 0.5) * noise
        });
    }
    return points;
};

// --- VISUALIZATIONS ---

const KMeansViz = () => {
    const [k, setK] = useState(3);
    const [points] = useState(() => generateBlobs(150, [{x: 20, y: 20}, {x: 80, y: 80}, {x: 20, y: 80}, {x: 80, y: 20}, {x: 50, y: 50}], 25));
    
    // Naive K-Means implementation for viz
    const { clusters, centroids } = useMemo(() => {
        // 1. Initialize random centroids
        let currentCentroids = points.slice(0, k).map(p => ({ x: p.x, y: p.y }));
        let assignments = new Array(points.length).fill(0);
        
        // Run a few iterations to stabilize
        for (let iter = 0; iter < 5; iter++) {
            // Assign
            assignments = points.map(p => {
                let minDist = Infinity;
                let clusterIdx = 0;
                currentCentroids.forEach((c, idx) => {
                    const dist = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIdx = idx;
                    }
                });
                return clusterIdx;
            });

            // Update
            const newCentroids = Array.from({ length: k }, () => ({ x: 0, y: 0, count: 0 }));
            points.forEach((p, idx) => {
                const clusterId = assignments[idx];
                newCentroids[clusterId].x += p.x;
                newCentroids[clusterId].y += p.y;
                newCentroids[clusterId].count++;
            });
            currentCentroids = newCentroids.map(c => c.count === 0 ? {x: 50, y: 50} : { x: c.x / c.count, y: c.y / c.count });
        }

        const colors = ['#14b8a6', '#6366f1', '#f43f5e', '#f59e0b', '#8b5cf6', '#ec4899'];
        
        return {
            clusters: points.map((p, i) => ({ ...p, cluster: assignments[i], fill: colors[assignments[i] % colors.length] })),
            centroids: currentCentroids.map((c, i) => ({ ...c, fill: '#ffffff', stroke: colors[i % colors.length] }))
        };
    }, [k, points]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Clusters (k): <span className="text-indigo-400 ml-2">{k}</span></label>
                    <input 
                        type="range" min="1" max="6" step="1" 
                        value={k} onChange={(e) => setK(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
                    />
                </div>
                <div className="text-[10px] font-mono px-3 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400">
                    Minimizing Variance
                </div>
            </div>

            <div className="h-72 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                        <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Scatter name="Points" data={clusters} shape="circle">
                            {clusters.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Scatter>
                        <Scatter name="Centroids" data={centroids} shape="cross" strokeWidth={3} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const DBSCANViz = () => {
    const [epsilon, setEpsilon] = useState(12);
    const [points] = useState(() => generateMoons(200, 5));

    // Simplified DBSCAN for visualization
    const clusteredData = useMemo(() => {
        const labels = new Array(points.length).fill(-1); // -1: Noise, 0+: Cluster
        let clusterId = 0;
        const visited = new Set();

        const getNeighbors = (idx: number) => {
            const neighbors = [];
            for(let i=0; i<points.length; i++) {
                if (i === idx) continue;
                const dist = Math.sqrt(Math.pow(points[i].x - points[idx].x, 2) + Math.pow(points[i].y - points[idx].y, 2));
                if (dist <= epsilon) neighbors.push(i);
            }
            return neighbors;
        };

        for(let i=0; i<points.length; i++) {
            if (visited.has(i)) continue;
            visited.add(i);
            
            const neighbors = getNeighbors(i);
            if (neighbors.length < 3) {
                labels[i] = -1; // Noise
            } else {
                clusterId++;
                labels[i] = clusterId;
                let seeds = [...neighbors];
                let seedIdx = 0;
                while(seedIdx < seeds.length) {
                    const currentIdx = seeds[seedIdx];
                    seedIdx++;
                    if (!visited.has(currentIdx)) {
                        visited.add(currentIdx);
                        const currentNeighbors = getNeighbors(currentIdx);
                        if (currentNeighbors.length >= 3) {
                            seeds = [...seeds, ...currentNeighbors];
                        }
                    }
                    if (labels[currentIdx] === -1) labels[currentIdx] = clusterId;
                }
            }
        }

        const colors = ['#94a3b8', '#14b8a6', '#f59e0b', '#6366f1', '#f43f5e']; // 0 is Noise (Slate)
        return points.map((p, i) => {
            const lbl = labels[i] === -1 ? 0 : (labels[i] % (colors.length - 1)) + 1;
            return { ...p, fill: colors[lbl], opacity: lbl === 0 ? 0.3 : 1 };
        });

    }, [epsilon, points]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Epsilon (Radius): <span className="text-emerald-400 ml-2">{epsilon}</span></label>
                    <input 
                        type="range" min="5" max="25" step="1" 
                        value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500 mt-2"
                    />
                </div>
                <div className="text-[10px] font-mono px-3 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400">
                    Density Based
                </div>
            </div>

            <div className="h-72 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" domain={[0, 120]} hide />
                        <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        
                        <Scatter name="Data" data={clusteredData} shape="circle">
                            {clusteredData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} fillOpacity={entry.opacity} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const PCAViz = () => {
    const [spread, setSpread] = useState(0.8); // Correlation factor
    const [showProjections, setShowProjections] = useState(true);

    const data = useMemo(() => {
        const points = [];
        const n = 50;
        const meanX = 50;
        const meanY = 50;
        // Generate correlated data y = x + noise
        for (let i = 0; i < n; i++) {
            const xRaw = (Math.random() - 0.5) * 80;
            const x = meanX + xRaw;
            // The more spread (closer to 1), the more y depends on x. 
            // We want spread=1 to be a line, spread=0 to be a cloud.
            const noise = (Math.random() - 0.5) * 60 * (1 - spread);
            const y = meanY + (xRaw * spread) + noise; 
            points.push({ x, y });
        }
        return points;
    }, [spread]);

    // Calculate PCA manually for 2D
    const pca = useMemo(() => {
        const n = data.length;
        const meanX = data.reduce((acc, p) => acc + p.x, 0) / n;
        const meanY = data.reduce((acc, p) => acc + p.y, 0) / n;

        let varX = 0, varY = 0, covXY = 0;
        data.forEach(p => {
            varX += (p.x - meanX) ** 2;
            varY += (p.y - meanY) ** 2;
            covXY += (p.x - meanX) * (p.y - meanY);
        });
        varX /= (n - 1);
        varY /= (n - 1);
        covXY /= (n - 1);

        // Eigenvalues
        const trace = varX + varY;
        const det = varX * varY - covXY * covXY;
        const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
        const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;

        // Eigenvector for lambda1 (PC1)
        let theta = 0;
        if (covXY !== 0) {
            theta = Math.atan2(lambda1 - varX, covXY);
        } else {
            theta = varX > varY ? 0 : Math.PI / 2;
        }

        return {
            mean: { x: meanX, y: meanY },
            lambda1,
            lambda2,
            vector1: { x: Math.cos(theta), y: Math.sin(theta) },
            vector2: { x: Math.cos(theta + Math.PI/2), y: Math.sin(theta + Math.PI/2) }
        };
    }, [data]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Correlation: <span className="text-indigo-400 ml-2">{spread.toFixed(2)}</span></label>
                    <input 
                        type="range" min="0.1" max="1" step="0.05" 
                        value={spread} onChange={(e) => setSpread(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
                    />
                </div>
                <button 
                    onClick={() => setShowProjections(!showProjections)}
                    className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase transition-all ${showProjections ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' : 'bg-slate-900 text-slate-500 border border-slate-700 hover:text-white'}`}
                >
                    {showProjections ? 'Hide Projections' : 'Show Projections'}
                </button>
            </div>

            <div className="h-72 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-4 relative overflow-hidden flex items-center justify-center select-none">
                <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                    <defs>
                        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#1e293b" strokeWidth="0.5"/>
                        </pattern>
                        <marker id="arrowPC1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                            <path d="M0,0 L0,6 L6,3 z" fill="#f43f5e" />
                        </marker>
                        <marker id="arrowPC2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                            <path d="M0,0 L0,6 L6,3 z" fill="#3b82f6" />
                        </marker>
                    </defs>
                    <rect width="100" height="100" fill="url(#grid)" />

                    {/* Infinite-looking PC1 Line */}
                    <line 
                        x1={pca.mean.x - pca.vector1.x * 100} y1={pca.mean.y - pca.vector1.y * 100} 
                        x2={pca.mean.x + pca.vector1.x * 100} y2={pca.mean.y + pca.vector1.y * 100} 
                        stroke="#f43f5e" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.5"
                    />
                    
                    {/* Infinite-looking PC2 Line */}
                    <line 
                        x1={pca.mean.x - pca.vector2.x * 100} y1={pca.mean.y - pca.vector2.y * 100} 
                        x2={pca.mean.x + pca.vector2.x * 100} y2={pca.mean.y + pca.vector2.y * 100} 
                        stroke="#3b82f6" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.3"
                    />

                    {/* Projections onto PC1 */}
                    {showProjections && data.map((p, i) => {
                        const vx = p.x - pca.mean.x;
                        const vy = p.y - pca.mean.y;
                        const projLen = vx * pca.vector1.x + vy * pca.vector1.y;
                        const projX = pca.mean.x + projLen * pca.vector1.x;
                        const projY = pca.mean.y + projLen * pca.vector1.y;

                        return (
                            <g key={`proj-${i}`}>
                                <line x1={p.x} y1={p.y} x2={projX} y2={projY} stroke="#f43f5e" strokeWidth="0.3" opacity="0.3" />
                                <circle cx={projX} cy={projY} r="1" fill="#f43f5e" opacity="0.6" />
                            </g>
                        )
                    })}

                    {/* PC1 Vector */}
                    <line 
                        x1={pca.mean.x} y1={pca.mean.y} 
                        x2={pca.mean.x + pca.vector1.x * 30} y2={pca.mean.y + pca.vector1.y * 30} 
                        stroke="#f43f5e" strokeWidth="2" markerEnd="url(#arrowPC1)"
                    />
                    {/* PC2 Vector */}
                    <line 
                        x1={pca.mean.x} y1={pca.mean.y} 
                        x2={pca.mean.x + pca.vector2.x * 15} y2={pca.mean.y + pca.vector2.y * 15} 
                        stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowPC2)"
                    />

                    {/* Data Points */}
                    {data.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r="1.8" fill="#94a3b8" opacity="0.9" />
                    ))}

                    {/* Centroid */}
                    <circle cx={pca.mean.x} cy={pca.mean.y} r="3" fill="#fff" stroke="#0f172a" strokeWidth="1" />
                </svg>

                {/* Info Overlay */}
                <div className="absolute top-4 right-4 flex flex-col gap-2 bg-slate-900/90 p-3 rounded-xl border border-slate-800 backdrop-blur-md shadow-xl">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-1 bg-rose-500 rounded"></div>
                        <span className="text-[10px] text-slate-300 font-bold uppercase">PC1 (Var: {(pca.lambda1).toFixed(0)})</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-1 bg-blue-500 rounded"></div>
                        <span className="text-[10px] text-slate-300 font-bold uppercase">PC2 (Var: {(pca.lambda2).toFixed(0)})</span>
                    </div>
                </div>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 text-xs text-slate-400 leading-relaxed">
                The red vector <strong>(PC1)</strong> aligns with the direction of maximum variance. The blue vector <strong>(PC2)</strong> is orthogonal to it. 
                Dimensionality reduction involves projecting points onto PC1 (red dots) and discarding the distance to the line (information loss).
            </div>
        </div>
    );
};

const IsolationForestViz = () => {
    const [contamination, setContamination] = useState(0.1);
    
    const data = useMemo(() => {
        const points = generateBlobs(100, [{x: 50, y: 50}], 30);
        // Add outliers
        points.push({ id: 101, x: 10, y: 10, outlier: true });
        points.push({ id: 102, x: 90, y: 90, outlier: true });
        points.push({ id: 103, x: 15, y: 85, outlier: true });
        points.push({ id: 104, x: 85, y: 15, outlier: true });
        
        // Calculate naive outlier score (distance from center)
        const scored = points.map(p => {
            const dist = Math.sqrt(Math.pow(p.x - 50, 2) + Math.pow(p.y - 50, 2));
            return { ...p, score: dist };
        });
        
        // Determine threshold based on contamination %
        const sortedScores = [...scored].sort((a, b) => b.score - a.score);
        const cutoffIndex = Math.floor(scored.length * contamination);
        const threshold = sortedScores[cutoffIndex].score;

        return scored.map(p => ({
            ...p,
            isAnomaly: p.score >= threshold,
            fill: p.score >= threshold ? '#f43f5e' : '#10b981'
        }));
    }, [contamination]);

    return (
        <div className="space-y-6">
             <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <div className="w-1/2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Contamination: <span className="text-rose-400 ml-2">{(contamination * 100).toFixed(0)}%</span></label>
                    <input 
                        type="range" min="0.01" max="0.25" step="0.01" 
                        value={contamination} onChange={(e) => setContamination(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500 mt-2"
                    />
                </div>
                <div className="flex gap-4 text-[10px] font-bold uppercase tracking-widest">
                    <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> Normal</div>
                    <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-rose-500"></div> Anomaly</div>
                </div>
            </div>

            <div className="h-72 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                        <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Scatter name="Data" data={data} shape="circle">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

// --- MAIN VIEW ---

export const UnsupervisedView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header className="mb-12 border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Unsupervised Learning</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
          Extracting structure from noise. Unsupervised algorithms organize data without the guidance of explicit labels, identifying clusters, densities, dimensions, and anomalies.
        </p>
      </header>

      {/* 1. K-MEANS */}
      <AlgorithmCard
        id="k-means"
        title="K-Means Clustering"
        complexity="Fundamental"
        theory="Partitions data into 'k' distinct clusters. It iteratively assigns each point to its closest centroid and then recalculates centroids based on assigned points. It minimizes the within-cluster variance."
        math={<span>J = &Sigma; &Sigma; || x - &mu;<sub>i</sub> ||<sup>2</sup></span>}
        mathLabel="Inertia (Sum of Squared Errors)"
        code={`from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_`}
        pros={['Extremely fast and simple O(n)', 'Scales to massive datasets', 'General purpose clustering']}
        cons={['Requires manual k selection', 'Sensitive to initial seeds', 'Assumes spherical cluster shapes']}
      >
        <KMeansViz />
      </AlgorithmCard>

      {/* 2. DBSCAN */}
      <AlgorithmCard
        id="dbscan"
        title="DBSCAN"
        complexity="Intermediate"
        theory="Density-Based Spatial Clustering of Applications with Noise. It groups together points that are closely packed (points with many nearby neighbors), marking points as outliers if they lie alone in low-density regions."
        math={<span>N<sub>&epsilon;</sub>(p) = {'{'}q &isin; D | dist(p,q) &le; &epsilon;{'}'}</span>}
        mathLabel="Epsilon Neighborhood"
        code={`from sklearn.cluster import DBSCAN
# eps: max distance, min_samples: min points for core
db = DBSCAN(eps=0.3, min_samples=5).fit(X)`}
        pros={['No need to specify k', 'Can find arbitrarily shaped clusters (non-linear)', 'Robust to outliers (noise)']}
        cons={['Struggles with varying densities', 'Sensitive to epsilon parameter', 'O(n²) worst case without indexing']}
      >
        <DBSCANViz />
      </AlgorithmCard>

      {/* 3. PCA */}
      <AlgorithmCard
        id="pca"
        title="Principal Component Analysis"
        complexity="Fundamental"
        theory="Linear dimensionality reduction. PCA projects data onto orthogonal vectors (Principal Components) that maximize variance. It is used to compress data while preserving the most important information."
        math={<span>\Sigma v = \lambda v</span>}
        mathLabel="Eigendecomposition of Covariance"
        code={`from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)`}
        pros={['Removes correlated features', 'Reduces overfitting', 'Speeds up training significantly']}
        cons={['Linear transformation only', 'Principal components are hard to interpret', 'Sensitive to feature scaling']}
      >
        <PCAViz />
      </AlgorithmCard>

      {/* 4. ISOLATION FOREST */}
      <AlgorithmCard
        id="isolation-forest"
        title="Isolation Forest"
        complexity="Intermediate"
        theory="An anomaly detection algorithm using tree ensembles. It works on the principle that anomalies are few and different. Random partitioning produces shorter paths for anomalies, making them easy to isolate."
        math={<span>s(x, n) = 2<sup>-E(h(x))/c(n)</sup></span>}
        mathLabel="Anomaly Score"
        code={`from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.1)
outliers = clf.fit_predict(X)`}
        pros={['Highly efficient for high-dim data', 'No distance computation required', 'Handles masking effects']}
        cons={['Heuristic based', 'Requires contamination estimate', 'Randomness can affect scores']}
      >
        <IsolationForestViz />
      </AlgorithmCard>

      {/* 5. T-SNE */}
      <AlgorithmCard
        id="tsne"
        title="t-SNE"
        complexity="Advanced"
        theory="A non-linear dimensionality reduction technique for visualization. It converts similarities between data points to joint probabilities and minimizes the Kullback-Leibler divergence between the low-dimensional embedding and the high-dimensional data."
        math={<span>C = &Sigma; p<sub>ij</sub> log(p<sub>ij</sub>/q<sub>ij</sub>)</span>}
        mathLabel="KL Divergence"
        code={`from sklearn.manifold import TSNE
# Perplexity controls local vs global focus
tsne = TSNE(n_components=2, perplexity=30)`}
        pros={['Unmatched visualization of clusters', 'Captures non-linear manifold structures']}
        cons={['Probabilistic (non-deterministic)', 'Distances are not preserved globally', 'Computationally expensive']}
      >
        <div className="bg-slate-950 p-8 rounded-3xl border border-slate-800 text-center">
            <Network className="w-16 h-16 text-indigo-500 mx-auto mb-4" />
            <p className="text-slate-400 text-sm">
                t-SNE is best used for <strong className="text-white">Exploratory Data Analysis (EDA)</strong> to visualize high-dimensional datasets in 2D or 3D.
                Unlike PCA, it preserves local neighborhoods.
            </p>
        </div>
      </AlgorithmCard>

    </div>
  );
};
