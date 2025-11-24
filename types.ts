export enum ViewSection {
  FOUNDATIONS = 'Foundations',
  REGRESSION = 'Regression',
  CLASSIFICATION = 'Classification',
  ENSEMBLE = 'Ensemble Methods',
  UNSUPERVISED = 'Unsupervised',
  DEEP_LEARNING = 'Deep Learning',
  REINFORCEMENT = 'Reinforcement Learning',
  MODEL_COMPARISON = 'Model Battleground',
  PROJECT_LAB = 'Project Lab'
}

export enum MLModelType {
  LOGISTIC_REGRESSION = 'Logistic Regression',
  RANDOM_FOREST = 'Random Forest',
  SVM = 'Support Vector Machine',
  KNN = 'K-Nearest Neighbors',
  XGBOOST = 'XGBoost'
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  confusionMatrix: { name: string; value: number }[];
  rocCurve: { fpr: number; tpr: number }[];
}

export interface NavigationItem {
  id: ViewSection;
  label: string;
  icon: React.ReactNode;
  subItems?: { id: string; label: string }[];
}