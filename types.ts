
import React from 'react';

export enum ViewSection {
  DASHBOARD = 'dashboard',
  FOUNDATIONS = 'foundations',
  OPTIMIZATION = 'optimization',
  REGRESSION = 'regression',
  CLASSIFICATION = 'classification',
  ENSEMBLE = 'ensemble',
  UNSUPERVISED = 'unsupervised',
  DEEP_LEARNING = 'deep-learning',
  REINFORCEMENT = 'reinforcement',
  MODEL_COMPARISON = 'battleground',
  PROJECT_LAB = 'lab',
  SITEMAP = 'sitemap'
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

// --- NEW DATA BACKBONE TYPES ---

export type ContentType = 'doc' | 'lab' | 'quiz';

export interface LabConfig {
  initialCode: string;
  solution: string;
  hints: string[];
}

export interface Question {
  id: string;
  text: string;
  options: string[];
  correctIndex: number;
  explanation?: string;
}

export interface QuizConfig {
  questions: Question[];
  passingScore?: number; // percentage, default 70
}

export interface Topic {
  id: string;
  title: string;
  type: ContentType;
  description?: string;
  labConfig?: LabConfig;
  quizConfig?: QuizConfig;
}

export interface Chapter {
  id: string;
  title: string;
  topics: Topic[];
}

export interface Module {
  id: string;
  title: string;
  icon?: React.ReactNode;
  chapters: Chapter[];
}

// Legacy support (can be deprecated later)
export interface NavigationItem {
  id: string; 
  label: string;
  icon?: React.ReactNode;
  category?: string; 
  items?: NavigationItem[];
}

export interface DocMetadata {
  title: string;
  description: string;
  date: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  tags: string[];
  readTimeMinutes: number;
}

export interface ContentModule {
  metadata: DocMetadata;
  Content: React.FC;
}
