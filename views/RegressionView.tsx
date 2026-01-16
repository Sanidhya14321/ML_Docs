
import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Line, BarChart, Bar, Legend, ComposedChart } from 'recharts';
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

  return (
    <div className="space-y-4">
        <div className="flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                Polynomial Degree: <span className="text-indigo-400 ml-2">{degree}</span>
            </label>
            <input 
                type="range" min="1" max="15" step="1" 
                value={degree} onChange={(e) => setDegree(Number(e.target.value))}
                className="w-32 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
        </div>
        <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-2 relative">
             <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" stroke="#475569" hide />
                    <YAxis type="number" stroke="#475569" hide />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                    <Scatter name="Data" dataKey="y" fill="#818cf8" />
                    <Line type="monotone" dataKey="curve" stroke="#f43f5e" strokeWidth={3} dot={false} animationDuration={300} />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
         <p className="text-[10px] text-center text-slate-500 uppercase tracking-widest font-mono">
            {degree === 1 ? "Underfitting (High Bias)" : degree > 8 ? "Overfitting (High Variance)" : "Balanced Fit"}
        </p>
    </div>
  );
};

const RegularizationViz = () => {
    return (
      <div className="h-64 w-full bg-slate-950 rounded-2xl border border-slate-800/50 p-4">
          <ResponsiveContainer width="100%" height="100%">
              <BarChart data={coefficientData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="feature" stroke="#475569" />
                  <YAxis stroke="#475569" />
                  <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                  <Legend />
                  <Bar dataKey="Linear" fill="#94a3b8" />
                  <Bar dataKey="Ridge" fill="#6366f1" />
                  <Bar dataKey="Lasso" fill="#f43f5e" />
              </BarChart>
          </ResponsiveContainer>
          <p className="text-[10px] text-center text-slate-500 mt-4 uppercase tracking-widest font-mono">
              Lasso (Red) drives coefficients to zero (Feature Selection). Ridge (Purple) shrinks them.
          </p>
      </div>
    );
};

export const RegressionView: React.FC = () => {
    return (
      <div className="space-y-12 animate-fade-in pb-20">
        <header className="mb-12 border-b border-slate-800 pb-8">
          <h1 className="text-5xl font-serif font-bold text-white mb-4">Supervised: Regression</h1>
          <p className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light">
            Predicting continuous values. From simple trend lines to complex polynomial curves, regression is the workhorse of forecasting and quantification.
          </p>
        </header>
  
        <AlgorithmCard
          id="linear-regression"
          title="Linear Regression"
          complexity="Fundamental"
          theory="The process of finding the optimal straight line that minimizes the distance (residual) between the predicted points and actual data points. It assumes a linear relationship between input variables and the single output variable."
          math={<span>y = \beta_0 + \beta_1 x + \epsilon</span>}
          mathLabel="Linear Equation"
          code={`from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)`}
          pros={['Simple and interpretable', 'Fast to train', 'Basis for many other methods']}
          cons={['Assumes linearity', 'Sensitive to outliers']}
          steps={[
            "Start a new notebook in Google Colab.",
            "Import `pandas` for dataframes and `sklearn.linear_model.LinearRegression`.",
            "Load your dataset (e.g., `pd.read_csv('housing.csv')`).",
            "Separate features (X) and target variable (y).",
            "Instantiate the model: `reg = LinearRegression()`.",
            "Fit the model: `reg.fit(X_train, y_train)`.",
            "Predict outcomes: `y_pred = reg.predict(X_test)`.",
            "Evaluate using `mean_squared_error` from `sklearn.metrics`."
          ]}
        >
          <LinearViz />
        </AlgorithmCard>
  
        <AlgorithmCard
          id="polynomial-regression"
          title="Polynomial Regression"
          complexity="Intermediate"
          theory="Extends linear models to model non-linear relationships by transforming the original features into polynomial features (e.g., x², x³). It fits a curve rather than a straight line."
          math={<span>y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n</span>}
          mathLabel="Polynomial Equation"
          code={`from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model.fit(X_poly, y)`}
          pros={['Models non-linear data', 'Still uses linear optimization under hood']}
          cons={['Prone to overfitting with high degrees', 'Feature explosion']}
          hyperparameters={[
              { name: 'degree', description: 'The degree of the polynomial features.', default: '2' }
          ]}
          steps={[
            "Open Google Colab and import `PolynomialFeatures` from `sklearn.preprocessing`.",
            "Define your input features X.",
            "Create a polynomial transformer: `poly = PolynomialFeatures(degree=2)`.",
            "Transform X: `X_poly = poly.fit_transform(X)`.",
            "Treat `X_poly` as your new features and fit a standard `LinearRegression` model.",
            "Visualize the fitted curve against the original data points."
          ]}
        >
          <PolyViz />
        </AlgorithmCard>

        <AlgorithmCard
            id="regularization"
            title="Regularization (Ridge & Lasso)"
            complexity="Intermediate"
            theory="Regularization adds a penalty term to the loss function to prevent overfitting. Ridge (L2) shrinks coefficients evenly, while Lasso (L1) can shrink coefficients to zero, effectively performing feature selection."
            math={<span>J(\theta) = MSE + \lambda \Sigma |\beta_i|</span>}
            mathLabel="L1 (Lasso) Cost Function"
            code={`from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)`}
            pros={['Prevents overfitting', 'Lasso handles feature selection', 'Ridge handles multicollinearity']}
            cons={['Requires hyperparameter tuning (alpha)', 'Introduces bias to reduce variance']}
            hyperparameters={[
                { name: 'alpha', description: 'Regularization strength; must be a positive float.', default: '1.0' }
            ]}
            steps={[
                "Load a dataset with many features in Colab.",
                "Import `Ridge` and `Lasso` from `sklearn.linear_model`.",
                "Instantiate `Ridge(alpha=1.0)` for L2 regularization.",
                "Instantiate `Lasso(alpha=0.1)` for L1 regularization.",
                "Fit both models to the training data.",
                "Inspect coefficients (`model.coef_`). Notice how Lasso drives some to exactly zero.",
                "Tune `alpha` to balance bias and variance."
            ]}
        >
            <RegularizationViz />
        </AlgorithmCard>
      </div>
    );
};
