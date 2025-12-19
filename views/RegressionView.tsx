
import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Line, BarChart, Bar, Legend, ComposedChart, ReferenceLine } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { Eye, EyeOff } from 'lucide-react';

// Generate synthetic linear data with noise (Linear Regression)
const linearData = Array.from({ length: 30 }, (_, i) => {
  const x = i;
  const line = 2 * x + 5;
  const noise = (Math.random() * 10 - 5);
  const y = line + noise;
  return { x, y, line, residual: [ {x, y}, {x, y: line} ] };
});

// Ridge vs Lasso Coefficients Data
const coefficientData = [
  { feature: 'F1', Linear: 10, Ridge: 7, Lasso: 8 },
  { feature: 'F2', Linear: 5, Ridge: 3, Lasso: 0 }, // Lasso zeros out
  { feature: 'F3', Linear: 0.5, Ridge: 0.2, Lasso: 0 }, // Lasso zeros out
  { feature: 'F4', Linear: 8, Ridge: 6, Lasso: 6 },
  { feature: 'F5', Linear: 2, Ridge: 1, Lasso: 0 }, // Lasso zeros out
];

const LinearViz = () => {
    const [showResiduals, setShowResiduals] = useState(false);

    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Prediction Visualizer</span>
                <button 
                    onClick={() => setShowResiduals(!showResiduals)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase transition-all ${showResiduals ? 'bg-indigo-600 text-white' : 'bg-slate-900 text-slate-500'}`}
                >
                    {showResiduals ? <Eye size={14} /> : <EyeOff size={14} />}
                    {showResiduals ? 'Hide Residuals (MSE)' : 'Show Residuals (MSE)'}
                </button>
            </div>
            <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={linearData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="x" stroke="#475569" hide />
                        <YAxis type="number" stroke="#475569" hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        
                        {showResiduals && linearData.map((d, i) => (
                            <Line 
                                key={i} 
                                data={d.residual} 
                                dataKey="y" 
                                stroke="#f43f5e" 
                                strokeWidth={1} 
                                strokeOpacity={0.4} 
                                dot={false} 
                                activeDot={false} 
                                animationDuration={0}
                            />
                        ))}

                        <Scatter name="Data" dataKey="y" fill="#818cf8" />
                        <Line name="Best Fit" dataKey="line" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={false} animationDuration={500} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-center text-slate-500 uppercase tracking-[0.2em] font-mono">
                {showResiduals ? "RED LINES INDICATE ERROR DISTANCE (RESIDUALS)" : "MINIMIZING THE SUM OF SQUARED RED LINES"}
            </p>
        </div>
    );
};

const PolyViz = () => {
  const [degree, setDegree] = useState(2);

  const baseData = useMemo(() => {
      return Array.from({ length: 20 }, (_, i) => {
        const x = (i / 19) * 10 - 5; // -5 to 5
        const y = 0.5 * x * x - 2 + (Math.random() - 0.5) * 5;
        return { x, y };
      });
  }, []);

  const chartData = useMemo(() => {
      return baseData.map(p => {
          let pred = 0;
          if (degree === 1) { pred = 0.5 * p.x + 2; } 
          else if (degree === 2) { pred = 0.5 * p.x * p.x - 2; } 
          else { pred = (0.5 * p.x * p.x - 2) + Math.sin(p.x * degree) * (degree * 0.1); }
          return { ...p, curve: pred };
      });
  }, [baseData, degree]);

  let label = "Balanced";
  let color = "#f59e0b"; 
  if (degree === 1) { label = "Underfitting"; color = "#ef4444"; } 
  else if (degree > 5) { label = "Overfitting"; color = "#818cf8"; }

  return (
      <div className="flex flex-col gap-4">
          <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
             <div className="w-1/2">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Degree: <span className="text-indigo-400 text-sm ml-2">{degree}</span></label>
                <input 
                  type="range" min="1" max="10" step="1" 
                  value={degree} onChange={(e) => setDegree(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
             </div>
             <div className="text-[10px] font-mono font-bold px-3 py-1 rounded bg-slate-900 border border-slate-700" style={{ color }}>
                {label}
             </div>
          </div>

          <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2">
            <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" hide />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                    <Scatter name="Data" dataKey="y" fill="#94a3b8" opacity={0.6} />
                    <Line type="monotone" dataKey="curve" stroke={color} strokeWidth={3} dot={false} animationDuration={300} />
                </ComposedChart>
            </ResponsiveContainer>
          </div>
      </div>
  );
};

export const RegressionView: React.FC = () => {
  return (
    <div className="space-y-12 animate-fade-in pb-20">
      <header className="mb-12 border-b border-slate-800 pb-8">
        <h1 className="text-5xl font-serif font-bold text-white mb-4">Supervised: Regression</h1>
        <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
          Predicting continuous quantity. Regression models find the functional relationship between independent features and a numerical target variable.
        </p>
      </header>

      <AlgorithmCard
        id="linear-regression"
        title="Linear Regression"
        complexity="Fundamental"
        theory="The foundation of statistical modeling. It assumes a linear relationship between input features and target output, aiming to find the weights that minimize the squared differences between observed data and predictions."
        math={<span>J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub> &Sigma; (h<sub>&theta;</sub>(x) - y)<sup>2</sup></span>}
        mathLabel="Mean Squared Error (MSE)"
        code={`from sklearn.linear_model import LinearRegression\nmodel = LinearRegression().fit(X, y)`}
        pros={['Extremely fast and simple', 'Highly interpretable weights', 'Base for most advanced models']}
        cons={['Assumes strict linearity', 'Highly sensitive to outliers', 'Affected by multicollinearity']}
        hyperparameters={[
          { name: 'fit_intercept', description: 'Whether to calculate the bias term for this model.', default: 'True' }
        ]}
      >
        <LinearViz />
      </AlgorithmCard>

      <AlgorithmCard
        id="ridge-lasso"
        title="Regularized Regression"
        complexity="Intermediate"
        theory="Techniques to prevent overfitting by penalizing large weights. Ridge (L2) adds a squared penalty to discourage large coefficients, while Lasso (L1) adds an absolute penalty that can force coefficients to exactly zero, performing feature selection."
        math={<span>J(&theta;) = MSE + &lambda; ||&theta;||<sub>p</sub></span>}
        mathLabel="Penalty Term (p=1 Lasso, p=2 Ridge)"
        code={`from sklearn.linear_model import Ridge, Lasso\nridge = Ridge(alpha=1.0).fit(X, y)\nlasso = Lasso(alpha=0.1).fit(X, y)`}
        pros={['Prevents overfitting on complex data', 'Lasso provides automatic feature selection', 'Robust to collinearity (Ridge)']}
        cons={['Alpha (lambda) requires cross-validation tuning', 'May underestimate true weights']}
        hyperparameters={[
          { name: 'alpha', description: 'Regularization strength; higher means simpler model.', default: '1.0' }
        ]}
      >
        <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={coefficientData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="feature" stroke="#475569" fontSize={10} />
              <YAxis stroke="#475569" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Legend />
              <Bar dataKey="Linear" fill="#475569" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Ridge" fill="#818cf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Lasso" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="polynomial-regression"
        title="Polynomial Regression"
        complexity="Fundamental"
        theory="Captures non-linear relationships by creating synthetic polynomial features (x², x³...) before applying a linear solver. It enables linear models to 'curve' into the data space."
        math={<span>y = &theta;<sub>0</sub> + &theta;<sub>1</sub>x + &theta;<sub>2</sub>x<sup>2</sup> + ...</span>}
        code={`from sklearn.preprocessing import PolynomialFeatures\nX_poly = PolynomialFeatures(degree=2).fit_transform(X)`}
        pros={['Fits non-linear data easily', 'Maintains linear solver speed']}
        cons={['High degree leads to extreme overfitting', 'Extrapolates poorly outside train range']}
      >
        <PolyViz />
      </AlgorithmCard>
    </div>
  );
};
