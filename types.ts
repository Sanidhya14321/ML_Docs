
import React from 'react';

export enum ViewSection {
  FOUNDATIONS = 'foundations',
  OPTIMIZATION = 'optimization',
  REGRESSION = 'regression',
  CLASSIFICATION = 'classification',
  ENSEMBLE = 'ensemble',
  UNSUPERVISED = 'unsupervised',
  DEEP_LEARNING = 'deep-learning',
  REINFORCEMENT = 'reinforcement',
  MODEL_COMPARISON = 'battleground',
  PROJECT_LAB = 'lab'
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
  category: 'Core' | 'Supervised' | 'Advanced' | 'Lab';
  subItems?: { id: string; label: string }[];
}
