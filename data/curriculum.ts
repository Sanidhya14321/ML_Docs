
import { 
  BookOpen, 
  Code, 
  Brain, 
  Database, 
  Calculator, 
  Layers, 
  Gamepad2, 
  CheckCircle,
  TrendingUp,
  Cpu,
  Network,
  Sigma,
  GitBranch,
  Search,
  Zap,
  Box,
  Server,
  FlaskConical,
  Activity,
  Swords,
  PieChart
} from "lucide-react";
import { Course } from "../types";

export const CURRICULUM: Course = {
  id: "ai-engineering-mastery",
  title: "The Complete AI Engineering Pipeline",
  description: "From Linear Algebra to Large Language Models. A definitive curriculum for modern AI.",
  modules: [
    // ========================================================================
    // MODULE 1: MATHEMATICAL FOUNDATIONS
    // ========================================================================
    {
      id: "mod-math",
      title: "Module 1: Mathematical Foundations",
      description: "Linear Algebra, Calculus, Statistics, and Probability theory required for ML.",
      icon: Calculator,
      chapters: [
        {
          id: "chap-linalg",
          title: "Linear Algebra",
          topics: [
            {
              id: "foundations", // Mapped to FoundationsView
              title: "Vectors & Matrices",
              type: "doc",
              icon: Sigma,
              description: "The geometric interpretation of data: Vectors, Matrices, and Dot Products.",
              content: `
# The Language of Data

To speak to a GPU, you must speak in Tensors.

### 1. Hierarchy of Tensors
* **Scalar (Rank 0):** A single number (e.g., $x = 5$).
* **Vector (Rank 1):** A 1D array (e.g., a row of data).
* **Matrix (Rank 2):** A 2D grid (e.g., a grayscale image).
* **Tensor (Rank 3+):** N-dimensional arrays (e.g., RGB video data).

### 2. Matrix Multiplication
The engine of Deep Learning. If $A$ is $(m \\times n)$ and $B$ is $(n \\times p)$, then $C = A \\cdot B$ is $(m \\times p)$.

$$
C_{ij} = \\sum_{k=1}^n A_{ik} B_{kj}
$$
              `
            },
            {
              id: "topic-eigen",
              title: "Eigenvalues & Eigenvectors",
              type: "doc",
              icon: Sigma,
              description: "Understanding principal components and dimensionality.",
              content: `
# Eigen decomposition

An **Eigenvector** is a vector that, when a linear transformation is applied to it, does not change direction. It only scales by a factor called the **Eigenvalue** ($\\lambda$).

$$
Av = \\lambda v
$$

This concept is crucial for:
1.  **PCA (Principal Component Analysis):** Reducing dimensions by finding the axes of maximum variance.
2.  **SVD (Singular Value Decomposition):** Used in recommendation systems.
              `
            },
            {
              id: "topic-matrix-decomp",
              title: "Matrix Decompositions",
              type: "doc",
              icon: Layers,
              description: "SVD, LU, and QR: Breaking matrices into fundamental building blocks.",
              content: `
# Singular Value Decomposition (SVD)

Every matrix $A$ (even non-square ones) can be decomposed into three specific matrices. This is the "Data Compression" of Linear Algebra.

$$ A = U \\Sigma V^T $$

*   **U (Left Singular Vectors):** Orthogonal matrix representing the "input" basis.
*   **$\\Sigma$ (Singular Values):** Diagonal matrix scaling the axes (strength of features).
*   **$V^T$ (Right Singular Vectors):** Orthogonal matrix representing the "output" basis.

Applications include **Image Compression**, **Denoising**, and **Dimensionality Reduction**.
              `
            }
          ]
        },
        {
          id: "chap-calculus",
          title: "Calculus & Optimization",
          topics: [
            {
              id: "topic-multivar-calc",
              title: "Multivariable Calculus",
              type: "doc",
              icon: TrendingUp,
              description: "Gradients, Jacobians, and Hessians: Navigating high-dimensional spaces.",
              content: `
# The Gradient Vector

In ML, we deal with functions of millions of variables (weights). The gradient $\\nabla f$ is a vector of partial derivatives pointing in the direction of steepest ascent.

$$
\\nabla f(x) = \\left[ \\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\dots, \\frac{\\partial f}{\\partial x_n} \\right]^T
$$

### The Jacobian & Hessian
*   **Jacobian Matrix:** The matrix of all first-order partial derivatives of a vector-valued function. Used in Backpropagation.
*   **Hessian Matrix:** The matrix of second-order derivatives. Tells us about the curvature of the loss landscape (convexity).
              `
            },
            {
              id: "optimization", // Mapped to OptimizationView
              title: "Optimization Engines",
              type: "doc",
              icon: Zap,
              description: "Gradient Descent and Backtracking: How machines learn by minimizing error.",
            },
            {
              id: "lab-autograd-scratch",
              title: "Lab: Build Autograd",
              type: "lab",
              icon: Code,
              description: "Implement simple dual-number automatic differentiation.",
              labConfig: {
                initialCode: `class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort logic would go here
        self._backward()

# TASK: Create variables a=2, b=-3. Calculate c = a*b + a. 
# Manually verify gradients.`,
                solution: "Implements basic DAG gradient flow",
                hints: ["Chain rule adds gradients", "Multiplication distributes gradients based on value"]
              }
            }
          ]
        },
        {
          id: "chap-prob-stats",
          title: "Probability & Statistics",
          topics: [
            {
              id: "topic-probability",
              title: "Probability Theory",
              type: "doc",
              icon: PieChart, // Placeholder icon, assumes PieChart is generic enough or add import
              description: "Quantifying uncertainty: Random Variables, Bayes' Theorem, and Distributions.",
              content: `
# Bayes' Theorem

The framework for updating beliefs based on new evidence.

$$ P(A|B) = \\frac{P(B|A)P(A)}{P(B)} $$

*   **P(A):** Prior (Initial belief).
*   **P(B|A):** Likelihood (How probable is the evidence given the hypothesis?).
*   **P(A|B):** Posterior (Updated belief).

In ML, we often want to find the parameters $\\theta$ that maximize the probability of the data $D$: $P(\\theta | D)$.
              `
            },
            {
              id: "topic-stats-inference",
              title: "Statistical Inference",
              type: "doc",
              icon: Activity,
              description: "Learning from data: Maximum Likelihood Estimation (MLE) and MAP.",
              content: `
# Maximum Likelihood Estimation (MLE)

The standard method for training neural networks. We search for parameters $\\theta$ that make the observed data most probable.

$$ \\hat{\\theta}_{MLE} = \\underset{\\theta}{\\text{argmax}} \\ \\mathcal{L}(\\theta; x) $$

Often, we minimize the **Negative Log-Likelihood (NLL)** because logarithms turn products into sums, which are numerically stable and easier to differentiate. This directly leads to Cross-Entropy Loss in classification.
              `
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 2: DATA ENGINEERING
    // ========================================================================
    {
      id: "mod-data",
      title: "Module 2: Data Engineering",
      description: "ETL, Feature Engineering, and Handling messy real-world data.",
      icon: Database,
      chapters: [
        {
          id: "chap-preprocessing",
          title: "Data Cleaning & Prep",
          topics: [
            {
              id: "topic-scaling",
              title: "Normalization & Standardization",
              type: "doc",
              icon: Database,
              description: "Feature Scaling techniques for distance-based algorithms.",
              content: `
# Feature Scaling

Models that rely on distance (KNN, SVM, K-Means) or Gradients fit poorly if features have vastly different scales (e.g., Age: 0-100 vs Income: 0-100,000).

### Min-Max Normalization
Scales data to [0, 1].
$$ x' = \\frac{x - \\min(x)}{\\max(x) - \\min(x)} $$

### Z-Score Standardization
Scales data to mean 0 and variance 1.
$$ z = \\frac{x - \\mu}{\\sigma} $$
              `
            },
            {
              id: "lab-pandas-pipeline",
              title: "Lab: Pandas Pipeline",
              type: "lab",
              icon: Code,
              description: "Build a robust pipeline handling missing data and categorical encoding.",
              labConfig: {
                initialCode: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# MOCK DATA
data = pd.DataFrame({
    'age': [25, np.nan, 30, 22],
    'salary': [50000, 60000, 55000, np.nan],
    'city': ['NY', 'SF', 'NY', 'Chicago']
})

def preprocess(df):
    # 1. Fill Missing Values (Imputation)
    df['age'] = df['age'].fillna(df['age'].mean())
    
    # 2. Normalize 'salary'
    # TODO: Implement Standard Scaling for Salary
    
    # 3. One-Hot Encode 'city'
    # TODO: Use pd.get_dummies
    
    return df

print(preprocess(data))`,
                solution: "Use fillna and standard scaler",
                hints: ["One-Hot Encoding expands columns", "Always handle NaNs before scaling"]
              }
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 3: CLASSICAL MACHINE LEARNING
    // ========================================================================
    {
      id: "mod-ml-classical",
      title: "Module 3: Classical Machine Learning",
      description: "From Regressions to advanced Boosting algorithms.",
      icon: GitBranch,
      chapters: [
        {
          id: "chap-supervised-basics",
          title: "Supervised Learning",
          topics: [
            {
              id: "regression", // Mapped to RegressionView
              title: "Regression Analysis",
              type: "doc",
              icon: Activity,
              description: "Predicting continuous variables using Linear, Ridge, Lasso, and Polynomial models.",
            },
            {
              id: "classification", // Mapped to ClassificationView
              title: "Classification Logic",
              type: "doc",
              icon: BookOpen,
              description: "Decision boundaries: Logistic Regression, KNN, SVM, and Naive Bayes.",
            }
          ]
        },
        {
          id: "chap-ensemble",
          title: "Ensemble Learning",
          topics: [
            {
              id: "ensemble", // Mapped to EnsembleView
              title: "Random Forests & Boosting",
              type: "doc",
              icon: Layers,
              description: "Random Forests and Gradient Boosting: Combining weak learners for robustness.",
            },
            {
              id: "battleground", // Mapped to ModelComparisonView
              title: "Model Battleground",
              type: "doc",
              icon: Swords,
              description: "Direct comparison of algorithm performance, training time, and interpretability.",
            },
            {
              id: "lab-xgboost-opt",
              title: "Lab: Training XGBoost",
              type: "lab",
              icon: Code,
              description: "Train a gradient boosting model and tune hyperparameters.",
              labConfig: {
                initialCode: `import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

# Mock Data Generation
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

def train_xgb():
    # 1. Create DMatrix (XGBoost specific structure)
    dtrain = xgb.DMatrix(X, label=y)
    
    # 2. Set Parameters
    params = {
        'max_depth': 3,  # Tree depth
        'eta': 0.1,      # Learning rate
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    # 3. Train
    bst = xgb.train(params, dtrain, num_boost_round=10)
    
    # 4. Predict
    preds = bst.predict(dtrain)
    predictions = [1 if p > 0.5 else 0 for p in preds]
    
    acc = accuracy_score(y, predictions)
    print(f"Model Accuracy: {acc}")
    return "Training Complete"

train_xgb()`,
                solution: "Basic XGBoost workflow",
                hints: ["DMatrix is faster than numpy arrays", "eta is the learning rate"]
              }
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 4: DEEP LEARNING
    // ========================================================================
    {
      id: "mod-dl",
      title: "Module 4: Deep Learning & Vision",
      description: "Neural Networks, CNNs, ResNets, and Vision Transformers.",
      icon: Brain,
      chapters: [
        {
          id: "chap-nn-foundations",
          title: "Neural Networks",
          topics: [
            {
              id: "deep-learning", // Mapped to DeepLearningView
              title: "Deep Neural Networks",
              type: "doc",
              icon: Brain,
              description: "MLPs, CNNs, RNNs, and the core Backpropagation algorithm.",
            },
            {
              id: "unsupervised", // Mapped to UnsupervisedView
              title: "Unsupervised Learning",
              type: "doc",
              icon: Search,
              description: "Clustering and Dimensionality Reduction: K-Means, Hierarchical, and t-SNE.",
            }
          ]
        },
        {
          id: "chap-cnn-arch",
          title: "Advanced CNNs",
          topics: [
            {
              id: "topic-resnet",
              title: "ResNet & Skip Connections",
              type: "doc",
              icon: Layers,
              description: "Solving vanishing gradients with residual blocks.",
              content: `
# The Vanishing Gradient Problem

As networks got deeper, they became harder to train because gradients would shrink to zero during backpropagation.

### Residual Networks (ResNet)
ResNet introduced "Skip Connections" (or shortcuts). Instead of learning $H(x)$, layers learn the residual $F(x) = H(x) - x$.
The output becomes $y = F(x) + x$. This allows gradients to flow directly through the network, enabling training of 100+ layer models.
              `
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 5: NLP & TRANSFORMERS
    // ========================================================================
    {
      id: "mod-nlp",
      title: "Module 5: NLP & LLMs",
      description: "From RNNs to GPT-4. Mastering Sequence Modeling.",
      icon: BookOpen,
      chapters: [
        {
          id: "chap-transformers",
          title: "The Transformer Revolution",
          topics: [
             {
              id: "deep-learning/attention-mechanism", // Mapped to custom content module
              title: "The Attention Mechanism",
              type: "doc",
              icon: Zap,
              description: "The mathematical heart of Transformers and LLMs.",
            },
            {
              id: "topic-bert-gpt",
              title: "BERT vs GPT",
              type: "doc",
              icon: Brain,
              description: "Encoder vs Decoder architectures.",
              content: `
# Encoder vs Decoder

### BERT (Bidirectional Encoder Representations from Transformers)
* **Architecture:** Encoder-only.
* **Task:** Masked Language Modeling (Fill in the blank).
* **Use Case:** Understanding, Classification, QA.

### GPT (Generative Pre-trained Transformer)
* **Architecture:** Decoder-only.
* **Task:** Causal Language Modeling (Predict next token).
* **Use Case:** Text Generation, Reasoning.
              `
            },
            {
              id: "lab-attention-mech",
              title: "Lab: Code Self-Attention",
              type: "lab",
              icon: Code,
              description: "Implement scaled dot-product attention from scratch.",
              labConfig: {
                initialCode: `import numpy as np

def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    
    # 1. Dot Product of Q and K Transpose
    scores = np.dot(q, k.T)
    
    # 2. Scale by square root of d_k
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Apply Softmax
    weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=-1, keepdims=True)
    
    # 4. Multiply by Values
    output = np.dot(weights, v)
    
    return output, weights

# Mock Vectors (Batch size 1, Seq len 3, Dim 4)
Q = np.random.rand(3, 4)
K = np.random.rand(3, 4)
V = np.random.rand(3, 4)

out, w = scaled_dot_product_attention(Q, K, V)
print("Attention Weights:\\n", w)`,
                solution: "Implements the math of the Attention paper",
                hints: ["Softmax turns scores into probabilities", "Scaling prevents exploding gradients"]
              }
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 6: RL & GENERATIVE AI
    // ========================================================================
    {
      id: "mod-advanced",
      title: "Module 6: RL & Generative AI",
      description: "Deep Reinforcement Learning, GANs, and Diffusion Models.",
      icon: Gamepad2,
      chapters: [
        {
          id: "chap-drl",
          title: "Deep Reinforcement Learning",
          topics: [
            {
              id: "reinforcement", // Mapped to ReinforcementView
              title: "RL Foundations",
              type: "doc",
              icon: Cpu,
              description: "Agents, Environments, Q-Learning, and Actor-Critic methods.",
            },
            {
              id: "topic-dqn",
              title: "Deep Q-Networks (DQN)",
              type: "doc",
              icon: Cpu,
              description: "Using Neural Networks to approximate the Q-function.",
              content: `
# Playing Atari with Deep Learning

Q-Learning uses a table. DQN replaces the table with a Neural Network to approximate the Q-function.

### Innovations:
1.  **Experience Replay:** Storing past transitions in a buffer and sampling randomly to break correlation.
2.  **Target Network:** Using a frozen copy of the network to stabilize training.
              `
            }
          ]
        },
        {
          id: "chap-genai",
          title: "Generative Deep Learning",
          topics: [
            {
              id: "topic-gans",
              title: "GANs",
              type: "doc",
              icon: Layers,
              description: "Generative Adversarial Networks: The game of two networks.",
              content: `
# The Adversarial Game

Two networks competing against each other:
1.  **Generator:** Tries to create fake data that looks real.
2.  **Discriminator:** Tries to distinguish real data from fake.

$$ \\min_G \\max_D V(D, G) $$
              `
            },
            {
              id: "topic-diffusion",
              title: "Diffusion Models",
              type: "doc",
              icon: Box,
              description: "Generating data by reversing noise.",
              content: `
# Denoising Diffusion

GANs are notoriously hard to train (mode collapse). Diffusion models work differently:
1.  **Forward Process:** Slowly add noise to an image until it is pure static.
2.  **Reverse Process:** Train a neural network to predict the noise added at each step and subtract it.

This allows generating high-quality images from random noise.
              `
            },
            {
              id: "quiz-final-boss",
              title: "Final Exam: AI Engineering",
              type: "quiz",
              icon: CheckCircle,
              description: "Test your knowledge across the entire curriculum.",
              quizConfig: {
                questions: [
                  {
                    id: "q1",
                    text: "Which Transformer component allows it to process words in parallel?",
                    options: ["Recurrent Connections", "Self-Attention", "Convolutional Filters", "Forget Gates"],
                    correctIndex: 1,
                    explanation: "Self-Attention computes relationships between all words simultaneously, unlike sequential RNNs."
                  },
                  {
                    id: "q2",
                    text: "In XGBoost, what is the role of the 'Learning Rate' (eta)?",
                    options: ["It determines the depth of trees", "It scales the contribution of each new tree", "It removes missing values", "It changes the loss function"],
                    correctIndex: 1,
                    explanation: "Learning rate shrinks the weight of each new tree to prevent overfitting."
                  },
                  {
                    id: "q3",
                    text: "What is the primary advantage of CatBoost over XGBoost?",
                    options: ["It runs on CPUs only", "It handles categorical features natively without One-Hot Encoding", "It uses Neural Networks", "It is older"],
                    correctIndex: 1,
                    explanation: "CatBoost uses 'ordered target statistics' to handle categories without preprocessing."
                  },
                  {
                    id: "q4",
                    text: "Which algorithm is commonly used for RLHF (Reinforcement Learning from Human Feedback) in LLMs?",
                    options: ["DQN", "K-Means", "PPO", "Linear Regression"],
                    correctIndex: 2,
                    explanation: "Proximal Policy Optimization (PPO) is the standard for fine-tuning LLMs with human feedback."
                  }
                ]
              }
            }
          ]
        },
        {
            id: "chap-capstone",
            title: "Capstone Project",
            topics: [
                {
                    id: "lab", // Mapped to ProjectLabView
                    title: "Medical Case Study",
                    type: "lab",
                    icon: FlaskConical,
                    description: "End-to-end project: EDA, Model Selection, and Performance Analysis on clinical heart disease data.",
                }
            ]
        }
      ]
    }
  ]
};
