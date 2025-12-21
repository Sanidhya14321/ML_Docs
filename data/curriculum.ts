
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
  PieChart,
  Eye,
  Container,
  Workflow,
  Sparkles,
  Settings
} from "lucide-react";
import { Course } from "../types";

export const CURRICULUM: Course = {
  id: "ai-engineering-mastery",
  title: "The Complete AI Engineering Pipeline",
  description: "From Mathematical First Principles to MLOps. A definitive curriculum for modern AI.",
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
              title: "Vectors, Matrices & Tensors",
              type: "doc",
              icon: Sigma,
              description: "The geometric interpretation of data: Basis, Norms, and Operations.",
              content: `
# The Language of Data

To speak to a GPU, you must speak in Tensors.

### 1. Hierarchy of Tensors
* **Scalar (Rank 0):** A single number (e.g., $x = 5$).
* **Vector (Rank 1):** A 1D array (e.g., a row of data).
* **Matrix (Rank 2):** A 2D grid (e.g., a grayscale image).
* **Tensor (Rank 3+):** N-dimensional arrays (e.g., RGB video data).

### 2. Norms
Measuring the "size" of vectors is crucial for regularization (L1/L2).
* **L1 Norm (Manhattan):** $||x||_1 = \\sum |x_i|$
* **L2 Norm (Euclidean):** $||x||_2 = \\sqrt{\\sum x_i^2}$
              `
            },
            {
              id: "topic-matrix-decomp",
              title: "Matrix Decompositions",
              type: "doc",
              icon: Layers,
              description: "SVD, LU, and Cholesky: Breaking matrices into fundamental building blocks.",
              content: `
# Singular Value Decomposition (SVD)

Every matrix $A$ can be decomposed into three specific matrices. This is the "Data Compression" of Linear Algebra.

$$ A = U \\Sigma V^T $$

*   **U (Left Singular Vectors):** Orthogonal matrix representing the "input" basis.
*   **$\\Sigma$ (Singular Values):** Diagonal matrix scaling the axes (strength of features).
*   **$V^T$ (Right Singular Vectors):** Orthogonal matrix representing the "output" basis.
              `
            },
            {
              id: "topic-eigen",
              title: "Eigenvalues & PCA Math",
              type: "doc",
              icon: Sigma,
              description: "Understanding principal components and dimensionality.",
              content: `
# Eigen decomposition

An **Eigenvector** is a vector that, when a linear transformation is applied to it, does not change direction. It only scales by a factor called the **Eigenvalue** ($\\lambda$).

$$
Av = \\lambda v
$$

This is the mathematical backbone of **Principal Component Analysis (PCA)**, which rotates data to align with axes of maximum variance.
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
              description: "Gradients, Jacobians, and Hessians.",
              content: `
# The Gradient Vector

In ML, we deal with functions of millions of variables (weights). The gradient $\\nabla f$ is a vector of partial derivatives pointing in the direction of steepest ascent.

$$
\\nabla f(x) = \\left[ \\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_n} \\right]^T
$$

### The Jacobian & Hessian
*   **Jacobian ($J$):** Matrix of first-order partial derivatives for vector-valued functions. Crucial for Backpropagation.
*   **Hessian ($H$):** Matrix of second-order derivatives. Determines the curvature of the loss landscape.
              `
            },
            {
              id: "optimization", // Mapped to OptimizationView
              title: "Optimization Algorithms",
              type: "doc",
              icon: Zap,
              description: "Gradient Descent, Newton's Method, and Convexity.",
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
    
    # ... Implement __mul__ and backward() ...
`,
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
              icon: PieChart, 
              description: "Bayes' Theorem, Distributions (Gaussian, Poisson), and MLE.",
              content: `
# Bayes' Theorem

$$ P(A|B) = \\frac{P(B|A)P(A)}{P(B)} $$

# Maximum Likelihood Estimation (MLE)
We search for parameters $\\theta$ that make the observed data most probable.
$$ \\hat{\\theta}_{MLE} = \\underset{\\theta}{\\text{argmax}} \\ \\mathcal{L}(\\theta; x) $$
              `
            },
            {
              id: "topic-hypothesis",
              title: "Hypothesis Testing",
              type: "doc",
              icon: Activity,
              description: "p-values, t-tests, and A/B testing fundamentals.",
              content: `
# Statistical Significance

How do we know if a model improvement is real or just noise?

*   **Null Hypothesis ($H_0$):** No effect exists.
*   **p-value:** The probability of observing results at least as extreme as the data, assuming $H_0$ is true.

If $p < \\alpha$ (usually 0.05), we reject the null hypothesis.
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
      description: "EDA, Cleaning, Feature Engineering, and Dimensionality Reduction.",
      icon: Database,
      chapters: [
        {
          id: "chap-eda",
          title: "Exploratory Analysis",
          topics: [
            {
              id: "topic-eda",
              title: "EDA & Visualization",
              type: "doc",
              icon: Search,
              description: "Univariate/Bivariate analysis, Correlation Matrices, and Outlier Detection.",
              content: `
# Know Your Data

Before training, you must understand the distribution.

1.  **Univariate:** Histograms, Box Plots (IQR for outliers).
2.  **Bivariate:** Scatter plots, Correlation Heatmaps (Pearson/Spearman).
3.  **Missingness:** Heatmap of null values.
              `
            }
          ]
        },
        {
          id: "chap-feature-eng",
          title: "Feature Engineering",
          topics: [
            {
              id: "topic-scaling",
              title: "Scaling & Encoding",
              type: "doc",
              icon: Database,
              description: "Normalization, Standardization, and Categorical Encodings.",
              content: `
# Transformations

*   **Min-Max:** Scales to [0,1]. Sensitive to outliers.
*   **Standardization (Z-Score):** Centers around 0 with std dev 1.
*   **Log Transform:** Handles skewed distributions (e.g., income).

### Categorical
*   **One-Hot:** Good for low cardinality.
*   **Target Encoding:** Good for high cardinality, risk of leakage.
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

# MOCK DATA
data = pd.DataFrame({
    'age': [25, np.nan, 30, 22],
    'salary': [50000, 60000, 55000, np.nan],
    'city': ['NY', 'SF', 'NY', 'Chicago']
})

# TODO: Fill NA, Normalize Salary, Encode City`,
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
      description: "Supervised, Unsupervised, and Ensemble methods.",
      icon: GitBranch,
      chapters: [
        {
          id: "chap-supervised",
          title: "Supervised Learning",
          topics: [
            {
              id: "regression", // Mapped to RegressionView
              title: "Regression Analysis",
              type: "doc",
              icon: Activity,
              description: "Linear, Polynomial, Lasso (L1), and Ridge (L2).",
            },
            {
              id: "classification", // Mapped to ClassificationView
              title: "Classification Algorithms",
              type: "doc",
              icon: BookOpen,
              description: "Logistic Regression, KNN, SVM, Naive Bayes, and Decision Trees.",
            }
          ]
        },
        {
          id: "chap-ensemble",
          title: "Ensemble Learning",
          topics: [
            {
              id: "ensemble", // Mapped to EnsembleView
              title: "Bagging & Boosting",
              type: "doc",
              icon: Layers,
              description: "Random Forests, XGBoost, LightGBM, and CatBoost.",
            },
            {
              id: "battleground", // Mapped to ModelComparisonView
              title: "Model Battleground",
              type: "doc",
              icon: Swords,
              description: "Direct comparison of algorithm performance and trade-offs.",
            }
          ]
        },
        {
          id: "chap-unsupervised",
          title: "Unsupervised Learning",
          topics: [
            {
              id: "unsupervised", // Mapped to UnsupervisedView
              title: "Clustering & Dim Reduction",
              type: "doc",
              icon: Network,
              description: "K-Means, Hierarchical, PCA, t-SNE, and Association Rules.",
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 4: DEEP LEARNING FUNDAMENTALS
    // ========================================================================
    {
      id: "mod-dl",
      title: "Module 4: Deep Learning Fundamentals",
      description: "ANNs, Backpropagation, and Optimization strategies.",
      icon: Brain,
      chapters: [
        {
          id: "chap-ann",
          title: "Neural Networks",
          topics: [
            {
              id: "deep-learning", // Mapped to DeepLearningView
              title: "Deep Neural Networks",
              type: "doc",
              icon: Brain,
              description: "Perceptrons, MLPs, Activation Functions, and Backpropagation.",
            }
          ]
        },
        {
          id: "chap-optimization",
          title: "Training & Optimization",
          topics: [
            {
              id: "topic-optimizers",
              title: "Optimizers & Loss",
              type: "doc",
              icon: TrendingUp,
              description: "Adam, RMSProp, SGD, and Cross-Entropy.",
              content: `
# Beyond Gradient Descent

Standard SGD gets stuck in local minima.

*   **Momentum:** Accumulates velocity to power through flat regions.
*   **RMSProp:** Adapts learning rates per parameter.
*   **Adam (Adaptive Moment Estimation):** Combines Momentum and RMSProp. The default for most tasks.
              `
            },
            {
              id: "topic-regularization-dl",
              title: "Regularization in DL",
              type: "doc",
              icon: CheckCircle,
              description: "Dropout, Batch Normalization, and Early Stopping.",
              content: `
# Fighting Overfitting

*   **Dropout:** Randomly ignoring neurons during training to prevent co-adaptation.
*   **Batch Normalization:** Normalizing layer inputs to stabilize learning.
*   **Early Stopping:** Monitoring validation loss and stopping when it degrades.
              `
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 5: COMPUTER VISION
    // ========================================================================
    {
      id: "mod-cv",
      title: "Module 5: Computer Vision",
      description: "CNNs, Object Detection, and Segmentation.",
      icon: Eye,
      chapters: [
        {
          id: "chap-cnn",
          title: "CNN Architectures",
          topics: [
            {
              id: "topic-cnn-layers",
              title: "Convolutions & Pooling",
              type: "doc",
              icon: Layers,
              description: "Filters, Stride, Padding, and Max Pooling.",
              content: "Visualizing kernels and feature map extraction."
            },
            {
              id: "topic-resnet",
              title: "ResNet & Modern Arcs",
              type: "doc",
              icon: Network,
              description: "Solving vanishing gradients with Residual Connections.",
              content: `
# ResNet

$y = F(x) + x$

By learning the residual $F(x)$, gradients can flow directly through the identity connection $x$, allowing for extremely deep networks (100+ layers).
              `
            }
          ]
        },
        {
          id: "chap-vision-tasks",
          title: "Advanced Vision",
          topics: [
            {
              id: "topic-yolo",
              title: "Object Detection (YOLO)",
              type: "doc",
              icon: Box,
              description: "Single-shot detection vs R-CNN approaches.",
              content: `
# You Only Look Once (YOLO)

Treats object detection as a single regression problem.
Divides image into a grid; each cell predicts bounding boxes and probabilities. Much faster than region-proposal methods (R-CNN).
              `
            },
            {
              id: "topic-segmentation",
              title: "Segmentation (U-Net)",
              type: "doc",
              icon: Layers,
              description: "Semantic vs Instance Segmentation.",
              content: "Pixel-level classification using Encoder-Decoder architectures with skip connections."
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 6: NLP
    // ========================================================================
    {
      id: "mod-nlp",
      title: "Module 6: NLP & Transformers",
      description: "Text processing, Embeddings, RNNs, and Transformers.",
      icon: BookOpen,
      chapters: [
        {
          id: "chap-embeddings",
          title: "Word Embeddings",
          topics: [
            {
              id: "topic-embeddings",
              title: "Word2Vec & GloVe",
              type: "doc",
              icon: Search,
              description: "Dense vector representations of text.",
              content: `
# Semantic Space

Mapping words to vectors such that:
$Vec(\\text{King}) - Vec(\\text{Man}) + Vec(\\text{Woman}) \\approx Vec(\\text{Queen})$
              `
            }
          ]
        },
        {
          id: "chap-transformers",
          title: "The Transformer Era",
          topics: [
             {
              id: "deep-learning/attention-mechanism", // Mapped to custom content module
              title: "The Attention Mechanism",
              type: "doc",
              icon: Zap,
              description: "Self-Attention, Multi-Head Attention, and Positional Encoding.",
            },
            {
              id: "topic-bert-gpt",
              title: "BERT vs GPT",
              type: "doc",
              icon: Brain,
              description: "Encoder (Bi-directional) vs Decoder (Auto-regressive) models.",
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
    # TODO: Implement Attention Formula
    pass`,
                solution: "Implements the math of the Attention paper",
                hints: ["Softmax turns scores into probabilities", "Divide by sqrt(d_k)"]
              }
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 7: LLMs & GenAI
    // ========================================================================
    {
      id: "mod-genai",
      title: "Module 7: LLMs & GenAI",
      description: "Fine-tuning, RAG, GANs, and Diffusion.",
      icon: Sparkles, 
      chapters: [
        {
          id: "chap-llm-ops",
          title: "LLM Engineering",
          topics: [
            {
              id: "topic-peft",
              title: "Fine-Tuning (PEFT/LoRA)",
              type: "doc",
              icon: Settings,
              description: "Low-Rank Adaptation and RLHF.",
              content: `
# LoRA (Low-Rank Adaptation)

Instead of updating all weights $W$, we freeze $W$ and train low-rank matrices $A$ and $B$:
$$ W' = W + \\Delta W = W + BA $$

This reduces trainable parameters by 10,000x.
              `
            },
            {
              id: "topic-rag",
              title: "RAG Systems",
              type: "doc",
              icon: Database,
              description: "Retrieval Augmented Generation with Vector DBs.",
              content: "Combining Parametric memory (LLM) with Non-Parametric memory (Vector DB) to reduce hallucinations."
            }
          ]
        },
        {
          id: "chap-image-gen",
          title: "Generative Vision",
          topics: [
            {
              id: "topic-gans",
              title: "GANs",
              type: "doc",
              icon: Layers,
              description: "Generator vs Discriminator.",
            },
            {
              id: "topic-diffusion",
              title: "Diffusion Models",
              type: "doc",
              icon: Box,
              description: "Stable Diffusion and Latent Space denoising.",
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 8: REINFORCEMENT LEARNING
    // ========================================================================
    {
      id: "mod-rl",
      title: "Module 8: Reinforcement Learning",
      description: "MDPs, Q-Learning, and Policy Gradients.",
      icon: Gamepad2,
      chapters: [
        {
          id: "chap-rl-basics",
          title: "RL Fundamentals",
          topics: [
            {
              id: "reinforcement", // Mapped to ReinforcementView
              title: "MDPs & Bellman Equations",
              type: "doc",
              icon: Cpu,
              description: "States, Actions, Rewards, and Q-Learning.",
            }
          ]
        },
        {
          id: "chap-rl-modern",
          title: "Modern RL",
          topics: [
            {
              id: "topic-dqn",
              title: "Deep Q-Networks (DQN)",
              type: "doc",
              icon: Network,
              description: "Experience Replay and Target Networks.",
            },
            {
              id: "topic-ppo",
              title: "Policy Gradients (PPO)",
              type: "doc",
              icon: TrendingUp,
              description: "Proximal Policy Optimization and Actor-Critic methods.",
              content: `
# Policy Gradients

Instead of learning value values $Q(s,a)$, we directly optimize the policy $\\pi(a|s)$.
**PPO** constrains the update step to prevent the policy from changing too drastically, ensuring stability.
              `
            }
          ]
        }
      ]
    },

    // ========================================================================
    // MODULE 9: MLOps
    // ========================================================================
    {
      id: "mod-mlops",
      title: "Module 9: MLOps & Deployment",
      description: "Serving, CI/CD, and Infrastructure.",
      icon: Container,
      chapters: [
        {
          id: "chap-serving",
          title: "Model Serving",
          topics: [
            {
              id: "topic-api",
              title: "REST APIs & ONNX",
              type: "doc",
              icon: Server,
              description: "FastAPI, TorchServe, and ONNX Runtime.",
              content: "Packaging models into microservices for production inference."
            }
          ]
        },
        {
          id: "chap-infra",
          title: "Infrastructure",
          topics: [
            {
              id: "topic-docker",
              title: "Docker & Kubernetes",
              type: "doc",
              icon: Box,
              description: "Containerization and orchestration for ML workloads.",
            },
            {
              id: "topic-cicd",
              title: "CI/CD for ML",
              type: "doc",
              icon: Workflow,
              description: "Automated training pipelines and Experiment Tracking (MLflow).",
            },
            {
              id: "quiz-final-boss",
              title: "Final Certification",
              type: "quiz",
              icon: CheckCircle,
              description: "Comprehensive exam covering all 9 modules.",
              quizConfig: {
                questions: [
                  {
                    id: "q1",
                    text: "What prevents the vanishing gradient problem in ResNets?",
                    options: ["Skip Connections", "Max Pooling", "Dropout", "L2 Regularization"],
                    correctIndex: 0
                  },
                  {
                    id: "q2",
                    text: "Which attention component computes the relevance score?",
                    options: ["Value * Key", "Query * Key", "Query * Value", "Softmax"],
                    correctIndex: 1
                  }
                ]
              }
            }
          ]
        }
      ]
    }
  ]
};
