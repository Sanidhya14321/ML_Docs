
import React from 'react';
import { 
  BookOpen, Database, Cpu, BrainCircuit, Gamepad2, Server, FlaskConical, 
  Code, Layers, Network, GitBranch, Terminal 
} from 'lucide-react';
import { ViewSection, NavigationItem } from '../types';

export const NAV_ITEMS: NavigationItem[] = [
  // CATEGORY 1: MATHEMATICAL FOUNDATIONS
  {
    id: 'cat-math', 
    label: '01. Mathematical Foundations', 
    icon: <BookOpen size={16} />, 
    category: 'Category 1: Math',
    items: [
      { id: 'math/linear-algebra', label: 'Linear Algebra', items: [
          { id: ViewSection.FOUNDATIONS, label: 'Vectors & Spaces' }, // Interactive
          { id: 'math/matrix-ops', label: 'Matrix Operations' },
          { id: 'math/eigen', label: 'Eigenvalues & Eigenvectors' },
          { id: 'math/svd', label: 'Singular Value Decomposition (SVD)' },
          { id: 'math/pca', label: 'Principal Component Analysis (PCA)' },
          { id: 'math/factorization', label: 'Matrix Factorization' }
      ]},
      { id: 'math/calculus', label: 'Calculus', items: [
          { id: 'math/derivatives', label: 'Derivatives & Gradients' },
          { id: 'math/chain-rule', label: 'The Chain Rule' },
          { id: 'math/jacobians', label: 'Jacobians & Hessians' },
          { id: 'math/taylor', label: 'Taylor Series' }
      ]},
      { id: 'math/probability', label: 'Probability & Stats', items: [
          { id: 'math/random-vars', label: 'Random Variables' },
          { id: 'math/distributions', label: 'Probability Distributions' },
          { id: 'math/bayes', label: 'Bayes\' Theorem' },
          { id: 'math/mle', label: 'Maximum Likelihood (MLE)' },
          { id: 'math/hypothesis', label: 'Hypothesis Testing' },
          { id: 'math/monte-carlo', label: 'Monte Carlo Methods' }
      ]},
      { id: 'math/optimization', label: 'Optimization', items: [
          { id: 'math/convexity', label: 'Convex vs Non-Convex' },
          { id: ViewSection.OPTIMIZATION, label: 'Gradient Descent (SGD, Adam)' }, // Interactive
          { id: 'math/lagrange', label: 'Lagrange Multipliers' },
          { id: 'math/newton', label: 'Newton\'s Method' }
      ]}
    ]
  },

  // CATEGORY 2: DATA ENGINEERING & INFRASTRUCTURE
  {
    id: 'cat-de', 
    label: '02. Data Engineering', 
    icon: <Database size={16} />, 
    category: 'Category 2: Data Eng',
    items: [
      { id: 'de/databases', label: 'Databases', items: [
          { id: 'de/sql-advanced', label: 'SQL Advanced (Window Fn)' },
          { id: 'de/nosql', label: 'NoSQL (Mongo, Cassandra)' },
          { id: 'de/cap', label: 'CAP Theorem & ACID' },
          { id: 'de/indexing', label: 'Indexing Strategies' }
      ]},
      { id: 'de/big-data', label: 'Big Data Processing', items: [
          { id: 'de/spark', label: 'Apache Spark (RDDs)' },
          { id: 'de/hadoop', label: 'Hadoop MapReduce' },
          { id: 'de/data-lakes', label: 'Data Lakes vs Warehouses' }
      ]},
      { id: 'de/orchestration', label: 'Workflow Orchestration', items: [
          { id: 'de/airflow', label: 'Apache Airflow (DAGs)' },
          { id: 'de/prefect', label: 'Prefect & Cron' }
      ]},
      { id: 'de/streaming', label: 'Streaming', items: [
          { id: 'de/kafka', label: 'Apache Kafka' },
          { id: 'de/flink', label: 'Flink & Real-time' }
      ]}
    ]
  },

  // CATEGORY 3: CLASSICAL MACHINE LEARNING
  {
    id: 'cat-ml', 
    label: '03. Classical ML', 
    icon: <Cpu size={16} />, 
    category: 'Category 3: ML',
    items: [
      { id: 'ml/supervised', label: 'Supervised Learning', items: [
          { id: ViewSection.REGRESSION, label: 'Linear & Logistic Reg' }, // Interactive
          { id: ViewSection.CLASSIFICATION, label: 'Support Vector Machines' }, // Interactive
          { id: 'ml/trees', label: 'Decision Trees' },
          { id: ViewSection.ENSEMBLE, label: 'Random Forests & Boosting' }, // Interactive
          { id: 'ml/knn', label: 'k-Nearest Neighbors' }
      ]},
      { id: 'ml/unsupervised', label: 'Unsupervised Learning', items: [
          { id: ViewSection.UNSUPERVISED, label: 'k-Means Clustering' }, // Interactive
          { id: 'ml/dbscan', label: 'DBSCAN' },
          { id: 'ml/hierarchical', label: 'Hierarchical Clustering' },
          { id: 'ml/tsne', label: 't-SNE & UMAP' }
      ]},
      { id: 'ml/evaluation', label: 'Model Evaluation', items: [
          { id: 'ml/bias-variance', label: 'Bias-Variance Tradeoff' },
          { id: 'ml/cross-val', label: 'Cross-Validation' },
          { id: 'ml/metrics', label: 'ROC-AUC & Precision-Recall' },
          { id: 'ml/regularization', label: 'Regularization (L1/L2)' }
      ]}
    ]
  },

  // CATEGORY 4: DEEP LEARNING
  {
    id: 'cat-dl', 
    label: '04. Deep Learning', 
    icon: <BrainCircuit size={16} />, 
    category: 'Category 4: DL',
    items: [
      { id: 'dl/neural-networks', label: 'Neural Network Basics', items: [
          { id: ViewSection.DEEP_LEARNING, label: 'Perceptrons & MLP' }, // Interactive
          { id: 'dl/activations', label: 'Activation Functions' },
          { id: 'dl/backprop', label: 'Backpropagation Algorithm' },
          { id: 'dl/init', label: 'Weight Initialization' }
      ]},
      { id: 'dl/computer-vision', label: 'Computer Vision (CNNs)', items: [
          { id: 'dl/cnn-layers', label: 'Conv Layers & Pooling' },
          { id: 'dl/architectures', label: 'Architectures (ResNet, VGG)' },
          { id: 'dl/detection', label: 'Object Detection (YOLO)' }
      ]},
      { id: 'dl/sequence', label: 'Sequence Models (RNNs)', items: [
          { id: 'dl/rnn-vanilla', label: 'Vanilla RNNs' },
          { id: 'dl/lstm-gru', label: 'LSTMs & GRUs' },
          { id: 'dl/seq2seq', label: 'Seq2Seq Models' }
      ]},
      { id: 'dl/transformers', label: 'Transformers & NLP', items: [
          { id: 'deep-learning/attention-mechanism', label: 'The Attention Mechanism' }, // Interactive
          { id: 'dl/bert-gpt', label: 'BERT & GPT' },
          { id: 'dl/tokenization', label: 'Tokenization & Embeddings' }
      ]},
      { id: 'dl/generative', label: 'Generative AI', items: [
          { id: 'dl/vae', label: 'Autoencoders (VAE)' },
          { id: 'dl/gans', label: 'GANs' },
          { id: 'dl/diffusion', label: 'Diffusion Models' },
          { id: 'dl/lora', label: 'LLM Fine-tuning (LoRA)' }
      ]}
    ]
  },

  // CATEGORY 5: REINFORCEMENT LEARNING
  {
    id: 'cat-rl', 
    label: '05. Reinforcement Learning', 
    icon: <Gamepad2 size={16} />, 
    category: 'Category 5: RL',
    items: [
      { id: 'rl/foundations', label: 'Foundations', items: [
          { id: ViewSection.REINFORCEMENT, label: 'MDPs & Bellman Eq' }, // Interactive
          { id: 'rl/exploration', label: 'Exploration vs Exploitation' }
      ]},
      { id: 'rl/model-free', label: 'Model-Free Methods', items: [
          { id: 'rl/monte-carlo', label: 'Monte Carlo & TD' },
          { id: 'rl/q-learning', label: 'Q-Learning & DQN' }
      ]},
      { id: 'rl/policy', label: 'Policy Gradients', items: [
          { id: 'rl/reinforce', label: 'REINFORCE Algorithm' },
          { id: 'rl/actor-critic', label: 'Actor-Critic (A2C/A3C)' },
          { id: 'rl/ppo', label: 'PPO & TRPO' }
      ]}
    ]
  },

  // CATEGORY 6: MLOPS & PRODUCTION
  {
    id: 'cat-mlops', 
    label: '06. MLOps & Production', 
    icon: <Server size={16} />, 
    category: 'Category 6: MLOps',
    items: [
      { id: 'mlops/deployment', label: 'Deployment', items: [
          { id: 'mlops/docker', label: 'Docker & Containers' },
          { id: 'mlops/k8s', label: 'Kubernetes for ML' },
          { id: 'mlops/serving', label: 'Model Serving' }
      ]},
      { id: 'mlops/lifecycle', label: 'Lifecycle Management', items: [
          { id: 'mlops/cicd', label: 'CI/CD for ML' },
          { id: 'mlops/tracking', label: 'Experiment Tracking' },
          { id: 'mlops/registry', label: 'Model Registry' }
      ]},
      { id: 'mlops/monitoring', label: 'Monitoring', items: [
          { id: 'mlops/drift', label: 'Drift Detection' },
          { id: 'mlops/performance', label: 'Performance Monitoring' }
      ]}
    ]
  },

  // LABS
  {
    id: 'labs', 
    label: '07. Project Labs', 
    icon: <FlaskConical size={16} />, 
    category: 'Labs',
    items: [
      { id: ViewSection.MODEL_COMPARISON, label: 'Algorithm Battleground' }, // Interactive
      { id: ViewSection.PROJECT_LAB, label: 'Clinical Case Study' } // Interactive
    ]
  }
];
