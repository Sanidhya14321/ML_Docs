import { MLModelType, ModelMetrics } from './types';

export const MEDICAL_MODEL_DATA: Record<MLModelType, ModelMetrics> = {
  [MLModelType.LOGISTIC_REGRESSION]: {
    accuracy: 85.4, precision: 82.1, recall: 78.5,
    confusionMatrix: [{ name: 'TP', value: 120 }, { name: 'FP', value: 30 }, { name: 'TN', value: 145 }, { name: 'FN', value: 25 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.1, tpr: 0.6 }, { fpr: 0.3, tpr: 0.8 }, { fpr: 0.5, tpr: 0.88 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.RANDOM_FOREST]: {
    accuracy: 94.2, precision: 93.5, recall: 91.0,
    confusionMatrix: [{ name: 'TP', value: 140 }, { name: 'FP', value: 10 }, { name: 'TN', value: 160 }, { name: 'FN', value: 10 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.05, tpr: 0.8 }, { fpr: 0.1, tpr: 0.92 }, { fpr: 0.2, tpr: 0.96 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.SVM]: {
    accuracy: 88.9, precision: 86.4, recall: 84.2,
    confusionMatrix: [{ name: 'TP', value: 130 }, { name: 'FP', value: 20 }, { name: 'TN', value: 150 }, { name: 'FN', value: 20 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.15, tpr: 0.7 }, { fpr: 0.25, tpr: 0.85 }, { fpr: 0.4, tpr: 0.9 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.KNN]: {
    accuracy: 86.1, precision: 83.0, recall: 81.5,
    confusionMatrix: [{ name: 'TP', value: 125 }, { name: 'FP', value: 28 }, { name: 'TN', value: 147 }, { name: 'FN', value: 22 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.12, tpr: 0.65 }, { fpr: 0.28, tpr: 0.82 }, { fpr: 0.45, tpr: 0.89 }, { fpr: 1, tpr: 1 }]
  },
  [MLModelType.XGBOOST]: {
    accuracy: 96.5, precision: 95.8, recall: 94.2,
    confusionMatrix: [{ name: 'TP', value: 145 }, { name: 'FP', value: 5 }, { name: 'TN', value: 165 }, { name: 'FN', value: 8 }],
    rocCurve: [{ fpr: 0, tpr: 0 }, { fpr: 0.02, tpr: 0.85 }, { fpr: 0.08, tpr: 0.95 }, { fpr: 0.15, tpr: 0.98 }, { fpr: 1, tpr: 1 }]
  }
};

export const MOTION_VARIANTS = {
  container: {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  },
  item: {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  }
};