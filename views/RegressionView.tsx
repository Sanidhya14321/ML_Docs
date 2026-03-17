
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
            <div className="flex justify-between items-center bg-app border border-border-strong p-4">
                <span className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.2em]">PREDICTION_VISUALIZER</span>
                <button 
                    onClick={() => setShowResiduals(!showResiduals)}
                    className={`flex items-center gap-3 px-4 py-2 border transition-all text-[9px] font-mono font-black uppercase tracking-widest ${showResiduals ? 'bg-brand border-brand text-app' : 'bg-surface border-border-strong text-text-muted hover:text-text-primary'}`}
                >
                    {showResiduals ? <Eye size={14} /> : <EyeOff size={14} />}
                    {showResiduals ? 'HIDE_RESIDUALS' : 'SHOW_RESIDUALS'}
                </button>
            </div>
            <div className="h-64 w-full bg-app border border-border-strong p-2 relative overflow-hidden">
                <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle, #1e293b 1px, transparent 1px)', backgroundSize: '20px 20px' }} />
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={linearData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis type="number" dataKey="x" hide />
                        <YAxis type="number" hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', borderRadius: '0px', fontSize: '9px', fontFamily: 'monospace' }} />
                        
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

                        <Scatter name="Data" dataKey="y" fill="var(--brand)" />
                        <Line name="Best Fit" dataKey="line" stroke="#f43f5e" strokeWidth={2} dot={false} activeDot={false} animationDuration={500} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <p className="text-[8px] text-center text-text-muted uppercase tracking-[0.4em] font-mono font-black">
                {showResiduals ? "ERROR_DISTANCE_METRIC: RESIDUALS_ACTIVE" : "OPTIMIZATION_GOAL: MINIMIZE_SUM_OF_SQUARES"}
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
        <div className="flex justify-between items-center bg-app border border-border-strong p-4">
            <label className="text-[9px] font-mono font-black text-text-muted uppercase tracking-widest">
                POLYNOMIAL_DEGREE: <span className="text-brand ml-2">{degree}</span>
            </label>
            <input 
                type="range" min="1" max="15" step="1" 
                value={degree} onChange={(e) => setDegree(Number(e.target.value))}
                className="w-32 h-1 bg-border-strong rounded-none appearance-none cursor-pointer accent-brand"
            />
        </div>
        <div className="h-64 w-full bg-app border border-border-strong p-2 relative overflow-hidden">
             <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'linear-gradient(#1e293b 1px, transparent 1px), linear-gradient(90deg, #1e293b 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
             <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" hide />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', borderRadius: '0px', fontSize: '9px', fontFamily: 'monospace' }} />
                    <Scatter name="Data" dataKey="y" fill="var(--brand)" />
                    <Line type="monotone" dataKey="curve" stroke="#f43f5e" strokeWidth={2} dot={false} animationDuration={300} />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
         <p className="text-[8px] text-center text-text-muted uppercase tracking-[0.4em] font-mono font-black">
            {degree === 1 ? "STATUS: UNDERFITTING_HIGH_BIAS" : degree > 8 ? "STATUS: OVERFITTING_HIGH_VARIANCE" : "STATUS: BALANCED_OPTIMAL_FIT"}
        </p>
    </div>
  );
};

const RegularizationViz = () => {
    return (
      <div className="h-64 w-full bg-app border border-border-strong p-6 relative overflow-hidden">
          <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'linear-gradient(#1e293b 1px, transparent 1px), linear-gradient(90deg, #1e293b 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
          <ResponsiveContainer width="100%" height="100%">
              <BarChart data={coefficientData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="feature" stroke="var(--text-muted)" fontSize={10} fontFamily="monospace" />
                  <YAxis stroke="var(--text-muted)" fontSize={10} fontFamily="monospace" />
                  <Tooltip cursor={{fill: 'rgba(16, 185, 129, 0.05)'}} contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border-strong)', borderRadius: '0px', fontSize: '9px', fontFamily: 'monospace' }} />
                  <Legend wrapperStyle={{ fontSize: '9px', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.1em' }} />
                  <Bar dataKey="Linear" fill="var(--text-muted)" />
                  <Bar dataKey="Ridge" fill="var(--brand)" />
                  <Bar dataKey="Lasso" fill="#f43f5e" />
              </BarChart>
          </ResponsiveContainer>
          <p className="text-[8px] text-center text-text-muted mt-6 uppercase tracking-[0.4em] font-mono font-black">
              LASSO_L1: FEATURE_SELECTION_ACTIVE // RIDGE_L2: WEIGHT_DECAY_ACTIVE
          </p>
      </div>
    );
};

import { motion } from 'framer-motion';

export const RegressionView: React.FC = () => {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="space-y-12 pb-20"
      >
      <header className="mb-12 border-b border-border-strong pb-10 relative overflow-hidden">
        <div className="absolute top-0 right-0 font-mono text-[8px] text-brand/20 tracking-[0.5em] pointer-events-none">
          REGRESSION_ENGINE_v4.0
        </div>
        <motion.div
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="flex items-center gap-3 mb-6"
        >
          <div className="w-12 h-[1px] bg-brand" />
          <span className="text-[10px] font-mono font-black text-brand uppercase tracking-[0.3em]">SUPERVISED_LEARNING</span>
        </motion.div>
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-6xl font-heading font-black text-text-primary mb-6 uppercase tracking-tight"
        >
          Regression
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-text-secondary text-lg max-w-3xl leading-relaxed font-sans uppercase tracking-tight"
        >
          Predict continuous numerical values by modeling relationships between dependent and independent variables.
        </motion.p>
      </header>
  
        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: { opacity: 0 },
            show: {
              opacity: 1,
              transition: {
                staggerChildren: 0.1
              }
            }
          }}
        >
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
        </motion.div>
      </motion.div>
    );
};
