import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line } from 'recharts';
import { AlgorithmCard } from '../components/AlgorithmCard';

const featureImportanceData = [
  { feature: 'Age', importance: 0.15 },
  { feature: 'BMI', importance: 0.12 },
  { feature: 'Glucose', importance: 0.35 },
  { feature: 'BP', importance: 0.08 },
  { feature: 'Insulin', importance: 0.25 },
];

const boostingData = Array.from({ length: 20 }, (_, i) => ({
  iter: i + 1,
  trainLoss: 10 * Math.exp(-0.2 * i) + 1,
  valLoss: 10 * Math.exp(-0.15 * i) + 2 + (i > 10 ? (i - 10) * 0.1 : 0) // Overfitting simulation
}));

const adaWeightsData = [
    { estimator: 'Tree 1', weight: 0.3 },
    { estimator: 'Tree 2', weight: 0.5 },
    { estimator: 'Tree 3', weight: 0.8 },
    { estimator: 'Tree 4', weight: 1.1 },
    { estimator: 'Tree 5', weight: 1.5 },
];

export const EnsembleView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <header className="mb-12">
        <h1 className="text-4xl font-serif font-bold text-white mb-4">Ensemble Methods</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Ensemble methods combine multiple machine learning models to create a more powerful and robust model. They typically reduce variance (bagging), bias (boosting), or improve predictions (stacking).
        </p>
      </header>

      <AlgorithmCard
        id="random-forest"
        title="Random Forest"
        theory="An ensemble learning method that constructs a multitude of decision trees at training time. It uses 'Bagging' (Bootstrap Aggregating) and feature randomness to create diverse trees and outputs the mode of the classes (classification) or mean prediction (regression)."
        math={<span>y&#770; = <sup>1</sup>&frasl;<sub>B</sub> &Sigma;<sub>b=1</sub><sup>B</sup> f<sub>b</sub>(x)</span>}
        mathLabel="Averaging Predictions"
        code={`from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)`}
        pros={['Reduces overfitting compared to decision trees', 'Handles missing values', 'Provides feature importance']}
        cons={['Slow to train and predict', 'Complex model (black box)', 'Large memory footprint']}
        hyperparameters={[
          {
            name: 'n_estimators',
            description: 'The number of trees in the forest.',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'max_depth',
            description: 'The maximum depth of the tree.',
            default: 'None',
            range: 'Integer'
          },
          {
            name: 'max_features',
            description: 'The number of features to consider when looking for the best split.',
            default: 'sqrt',
            range: 'sqrt, log2, None'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={featureImportanceData} margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={80} />
              <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
              <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} name="Importance" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Feature Importance Extraction</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="adaboost"
        title="AdaBoost (Adaptive Boosting)"
        theory="A boosting technique that trains predictors sequentially. Each subsequent model attempts to correct the errors of its predecessor by increasing the weights of misclassified instances."
        math={<span>H(x) = sign(&Sigma;<sub>t=1</sub><sup>T</sup> &alpha;<sub>t</sub>h<sub>t</sub>(x))</span>}
        mathLabel="Weighted Sum of Weak Learners"
        code={`from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
pred = ada.predict(X_test)`}
        pros={['Less prone to overfitting', 'Easy to implement', 'Can use various base classifiers']}
        cons={['Sensitive to noisy data and outliers', 'Slower training than bagging']}
        hyperparameters={[
          {
            name: 'n_estimators',
            description: 'The maximum number of estimators at which boosting is terminated.',
            default: '50',
            range: 'Integer'
          },
          {
            name: 'learning_rate',
            description: 'Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.',
            default: '1.0',
            range: 'Float'
          }
        ]}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
             <BarChart data={adaWeightsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="estimator" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                <Bar dataKey="weight" fill="#f472b6" radius={[4, 4, 0, 0]} name="Estimator Weight (Alpha)" />
             </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-center text-slate-500 mt-2">Subsequent estimators get higher weight (alpha) as they correct previous errors.</p>
        </div>
      </AlgorithmCard>

      <AlgorithmCard
        id="gradient-boosting"
        title="Gradient Boosting (XGBoost/LightGBM)"
        theory="Builds an additive model in a forward stage-wise fashion. It generalizes other boosting methods by allowing optimization of an arbitrary differentiable loss function. XGBoost is an optimized distributed gradient boosting library."
        math={<span>Obj = &Sigma; L(y<sub>i</sub>, y&#770;<sub>i</sub>) + &Sigma; &Omega;(f<sub>k</sub>)</span>}
        mathLabel="Objective: Loss + Regularization"
        code={`import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)`}
        pros={['State-of-the-art performance on tabular data', 'Handles missing data', 'Built-in regularization']}
        cons={['Many hyperparameters to tune', 'Computationally expensive', 'Can overfit if not tuned']}
        hyperparameters={[
          {
            name: 'learning_rate',
            description: 'Step size shrinkage used in update to prevent overfitting.',
            default: '0.1',
            range: '[0, 1]'
          },
          {
            name: 'n_estimators',
            description: 'Number of boosting rounds.',
            default: '100',
            range: 'Integer'
          },
          {
            name: 'subsample',
            description: 'Subsample ratio of the training instances.',
            default: '1.0',
            range: '(0, 1]'
          }
        ]}
      >
         <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
               <LineChart data={boostingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="iter" stroke="#94a3b8" label={{ value: 'Iterations', position: 'insideBottom', offset: -5, fill: '#64748b' }} />
                  <YAxis stroke="#94a3b8" label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }} />
                  <Legend verticalAlign="top" height={36}/>
                  <Line type="monotone" dataKey="trainLoss" stroke="#34d399" strokeWidth={2} dot={false} name="Training Loss" />
                  <Line type="monotone" dataKey="valLoss" stroke="#f472b6" strokeWidth={2} dot={false} name="Validation Loss" />
               </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-center text-slate-500 mt-2">Loss reduction over iterations. Divergence indicates overfitting.</p>
         </div>
      </AlgorithmCard>
    </div>
  );
};