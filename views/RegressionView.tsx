import React from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Line, BarChart, Bar, Legend, ComposedChart } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

// Generate synthetic linear data with noise (Linear Regression)
const data = Array.from({ length: 30 }, (_, i) => {
  const x = i;
  const y = 2 * x + 5 + (Math.random() * 10 - 5);
  return { x, y, line: 2 * x + 5 };
});

// Ridge vs Lasso Coefficients Data
const coefficientData = [
  { feature: 'F1', Linear: 10, Ridge: 7, Lasso: 8 },
  { feature: 'F2', Linear: 5, Ridge: 3, Lasso: 0 }, // Lasso zeros out
  { feature: 'F3', Linear: 0.5, Ridge: 0.2, Lasso: 0 }, // Lasso zeros out
  { feature: 'F4', Linear: 8, Ridge: 6, Lasso: 6 },
  { feature: 'F5', Linear: 2, Ridge: 1, Lasso: 0 }, // Lasso zeros out
];

// Polynomial Data
const polyData = Array.from({ length: 20 }, (_, i) => {
  const x = i - 10;
  // y = x^2 + noise
  const y = x * x + (Math.random() * 15 - 7.5);
  const curve = x * x;
  return { x, y, curve };
});

export const RegressionView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Supervised Learning: Regression</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Regression algorithms predict continuous values by modeling the relationship between independent variables (features) and a dependent variable (target).
        </p>
      </header>

      <AlgorithmCard
        id="linear-regression"
        title="Linear Regression"
        theory="Linear regression models the relationship between a scalar response and one or more explanatory variables using a linear equation. It assumes a linear relationship and minimizes the sum of squared errors between observed and predicted values."
        math={<span>J(&theta;) = <sup>1</sup>&frasl;<sub>2m</sub> &Sigma;<sub>i=1</sub><sup>m</sup> (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup></span>}
        mathLabel="Mean Squared Error (MSE)"
        code={`from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)`}
        pros={['Simple to implement and interpret', 'Computationally efficient', 'Basis for many other methods']}
        cons={['Assumes linearity between variables', 'Sensitive to outliers', 'Prone to multicollinearity']}
        hyperparameters={[
          {
            name: 'fit_intercept',
            description: 'Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).',
            default: 'True',
            range: 'True / False'
          },
          {
            name: 'n_jobs',
            description: 'The number of jobs to use for the computation. This will only provide a speedup for n_targets > 1 and sufficient large problems.',
            default: 'None',
            range: 'Integer'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" name="Input" />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" name="Target" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <Scatter name="Data" data={data} fill="#818cf8" />
              <Line type="monotone" dataKey="line" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={false} legendType="none" />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Data points (blue) and Best Fit Line (red)</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="ridge-lasso"
        title="Ridge & Lasso Regression"
        theory="Regularized regression techniques that add a penalty term to the loss function to prevent overfitting. Ridge (L2) shrinks coefficients towards zero, while Lasso (L1) can shrink them exactly to zero, effectively performing feature selection."
        math={<span>J(&theta;) = MSE + &lambda; &Sigma;<sub>j=1</sub><sup>n</sup> |&theta;<sub>j</sub>|<sup>p</sup></span>}
        mathLabel="Regularized Cost Function (p=1 for Lasso, p=2 for Ridge)"
        code={`from sklearn.linear_model import Ridge, Lasso

# Ridge (L2 Regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)`}
        pros={['Reduces overfitting', 'Handles multicollinearity (Ridge)', 'Feature selection (Lasso)']}
        cons={['Requires tuning of hyperparameter lambda', 'Lasso may struggle with correlated features']}
        hyperparameters={[
          {
            name: 'alpha',
            description: 'Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.',
            default: '1.0',
            range: '[0, infinity)'
          },
          {
            name: 'solver',
            description: 'Solver to use in the computational routines.',
            default: 'auto',
            range: 'auto, svd, cholesky, lsqr, sparse_cg, sag, saga'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={coefficientData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="feature" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" label={{ value: 'Coefficient Value', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <Legend />
              <Bar dataKey="Linear" fill="#94a3b8" />
              <Bar dataKey="Ridge" fill="#818cf8" />
              <Bar dataKey="Lasso" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Comparison of Coefficients: Lasso (Red) zeroes out irrelevant features (F2, F3, F5).</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="polynomial-regression"
        title="Polynomial Regression"
        theory="An extension of linear regression that models the relationship between variables as an nth degree polynomial. It transforms the original features into polynomial features to capture non-linear patterns."
        math={<span>y = &theta;<sub>0</sub> + &theta;<sub>1</sub>x + &theta;<sub>2</sub>x<sup>2</sup> + ... + &theta;<sub>n</sub>x<sup>n</sup></span>}
        code={`from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)`}
        pros={['Captures non-linear relationships', 'Uses linear regression solver']}
        cons={['Prone to overfitting with high degrees', 'Computationally expensive for many features']}
        hyperparameters={[
          {
            name: 'degree',
            description: 'The degree of the polynomial features. Higher degrees can capture more complex relationships but risk overfitting.',
            default: '2',
            range: 'Integer >= 1'
          },
          {
            name: 'interaction_only',
            description: 'If true, only interaction features are produced: features that are products of at most degree distinct input features.',
            default: 'False',
            range: 'True / False'
          },
          {
            name: 'include_bias',
            description: 'If True (default), then include a bias column, the feature in which all polynomial powers are zero.',
            default: 'True',
            range: 'True / False'
          }
        ]}
      >
        <div className="h-64 w-full">
           <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={polyData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
              <Scatter name="Data" dataKey="y" fill="#94a3b8" opacity={0.5} />
              <Line type="monotone" dataKey="curve" stroke="#f59e0b" strokeWidth={3} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Quadratic Curve Fitting (Degree 2) to non-linear data.</p>
        </div>
      </AlgorithmCard>
    </div>
  );
};