
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
              details: {
                theory: `**Tensors** are the fundamental data structure of Deep Learning. They generalize vectors and matrices to N-dimensions.
                
*   **Scalar (Rank 0):** Single magnitude ($x=5$).
*   **Vector (Rank 1):** 1D array, representing a point in space.
*   **Matrix (Rank 2):** 2D grid, representing a linear transformation.
*   **Tensor (Rank 3+):** N-dimensional array (e.g., RGB Image: Height x Width x Channels).

Understanding **Norms** is crucial for regularization. The L2 norm (Euclidean distance) penalizes large outliers heavily, while L1 (Manhattan) encourages sparsity.`,
                math: "||x||_2 = \\sqrt{\\sum_{i=1}^{n} x_i^2}",
                mathLabel: "L2 Euclidean Norm",
                code: `import torch
# Creating a Rank-3 Tensor (e.g., 2x2 Image with 3 Channels)
tensor = torch.randn(3, 2, 2)
print(f"Shape: {tensor.shape}, Rank: {tensor.ndim}")`
              }
            },
            {
              id: "topic-matrix-decomp",
              title: "Matrix Decompositions",
              type: "doc",
              icon: Layers,
              description: "SVD, LU, and Cholesky: Breaking matrices into fundamental building blocks.",
              details: {
                theory: `**Singular Value Decomposition (SVD)** is the "Data Compression" of Linear Algebra. It states that *any* matrix $A$ can be factorized into three distinct operations: a rotation ($V^T$), a scaling ($\Sigma$), and another rotation ($U$).
                
This is the mathematical engine behind **Dimensionality Reduction**, **Denoising**, and **Recommendation Systems**.`,
                math: "A = U \\Sigma V^T",
                mathLabel: "SVD Factorization",
                code: `import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)
# U: Left Singular Vectors
# S: Singular Values (Diagonal)
# Vt: Right Singular Vectors`
              }
            },
            {
              id: "topic-eigen",
              title: "Eigenvalues & PCA Math",
              type: "doc",
              icon: Sigma,
              description: "Understanding principal components and dimensionality.",
              details: {
                theory: `An **Eigenvector** is a vector that does not change its direction under a linear transformation; it only stretches or shrinks. The factor by which it stretches is the **Eigenvalue** ($\lambda$).
                
In **Principal Component Analysis (PCA)**, the eigenvectors of the data's Covariance Matrix represent the axes of maximum variance (information), and eigenvalues represent the magnitude of that variance.`,
                math: "A v = \\lambda v",
                mathLabel: "Eigenvalue Equation",
                code: `import numpy as np
from sklearn.decomposition import PCA

# Find principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)
print(f"Explained Variance: {pca.explained_variance_ratio_}")`
              }
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
              details: {
                theory: `Deep Learning is essentially the optimization of a complex function via **Calculus**.
                
*   **The Gradient ($\nabla f$):** A vector pointing in the direction of the steepest ascent. We move opposite to it to minimize error.
*   **The Jacobian ($J$):** A matrix of all first-order partial derivatives of a vector-valued function. Crucial for understanding how changes in network weights affect outputs.
*   **The Hessian ($H$):** A matrix of second-order derivatives describing the curvature of the loss landscape.`,
                math: "\\nabla f(x) = \\left[ \\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_n} \\right]^T",
                mathLabel: "Gradient Vector",
                code: `import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward() # Compute Gradient
print(f"dy/dx at x=2: {x.grad.item()}") # 2x + 3 = 7`
              }
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
    """
    A scalar value with autograd capability.
    Wraps a float and tracks its history.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    # TODO: Implement __mul__ and backward() logic
    def __mul__(self, other):
        pass

    def backward(self):
        # Topological sort required here
        pass
`,
                solution: "Implements basic DAG gradient flow",
                hints: ["The derivative of (a*b) with respect to 'a' is 'b'.", "You need to sort the graph topologically before calling _backward()."]
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
              details: {
                theory: `**Bayes' Theorem** is the framework for updating beliefs based on new evidence. It relates the conditional probability $P(A|B)$ to $P(B|A)$.
                
**Maximum Likelihood Estimation (MLE)** is a method to estimate the parameters of a statistical model (like a Neural Network) by maximizing the probability of observing the given data under those parameters.`,
                math: "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}",
                mathLabel: "Bayes' Theorem",
                code: `from scipy.stats import norm
# Calculate Probability Density Function (PDF)
prob = norm.pdf(0, loc=0, scale=1) 
print(f"Probability of x=0 in standard normal: {prob}")`
              }
            },
            {
              id: "topic-hypothesis",
              title: "Hypothesis Testing",
              type: "doc",
              icon: Activity,
              description: "p-values, t-tests, and A/B testing fundamentals.",
              details: {
                theory: `Hypothesis testing provides a rigorous way to ask: "Is this result real, or just noise?"
                
*   **Null Hypothesis ($H_0$):** The default position (e.g., "The new model is no better than the old one").
*   **p-value:** The probability of observing results at least as extreme as the data, assuming $H_0$ is true. If $p < 0.05$, we reject $H_0$.`,
                math: "t = \\frac{\\bar{x} - \\mu_0}{s / \\sqrt{n}}",
                mathLabel: "T-Statistic Formula",
                code: `from scipy import stats
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
# Perform T-Test
print(stats.ttest_ind(rvs1, rvs2))`
              }
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
              details: {
                theory: `**EDA** is the detective work of Data Science. Before any modeling, you must understand the distribution, frequency, and relationship of your data points.
                
Key Techniques:
*   **Univariate:** Histograms (distribution), Box Plots (outliers/IQR).
*   **Bivariate:** Scatter plots, Correlation Heatmaps.
*   **Z-Score:** Identifying how many standard deviations a data point is from the mean to detect anomalies.`,
                math: "z = \\frac{x - \\mu}{\\sigma}",
                mathLabel: "Z-Score Standardization",
                code: `import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()`
              }
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
              details: {
                theory: `Models that rely on distance (KNN, SVM, K-Means) or gradients (Neural Networks) require features to be on a similar scale.
                
*   **Standardization (Z-Score):** Centers data around 0 with variance 1. Best for Gaussian data.
*   **Normalization (Min-Max):** Scales data to [0, 1]. Best for image pixel intensities or bounded data.
*   **One-Hot Encoding:** Converts categorical variables into binary vectors (orthogonal).`,
                math: "x' = \\frac{x - \\min(x)}{\\max(x) - \\min(x)}",
                mathLabel: "Min-Max Normalization",
                code: `from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y_categories)`
              }
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
              details: {
                theory: `Training a network means navigating a high-dimensional loss landscape to find a global minimum.
                
*   **SGD (Stochastic Gradient Descent):** Updates weights using mini-batches. Can be noisy.
*   **Momentum:** Accumulates velocity to power through flat regions and local minima.
*   **RMSProp:** Adapts learning rates for each parameter, dividing by the moving average of squared gradients.
*   **Adam (Adaptive Moment Estimation):** The gold standard. Combines Momentum and RMSProp for fast convergence.`,
                math: "\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{v_t + \\epsilon}} m_t",
                mathLabel: "Adam Update Rule (Simplified)",
                code: `import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)`
              }
            },
            {
              id: "topic-regularization-dl",
              title: "Regularization in DL",
              type: "doc",
              icon: CheckCircle,
              description: "Dropout, Batch Normalization, and Early Stopping.",
              details: {
                theory: `Deep networks are prone to overfitting due to their massive capacity.
                
*   **Dropout:** Randomly zeroes out neurons during training (p=0.5). Forces the network to learn robust, redundant representations.
*   **Batch Normalization:** Normalizes layer inputs to mean 0 and variance 1. Stabilizes training and allows higher learning rates.
*   **Early Stopping:** Monitoring validation loss and halting training when it stops improving to prevent overfitting.`,
                math: "\\hat{x}^{(k)} = \\frac{x^{(k)} - \\mu_B}{\\sqrt{\\sigma^2_B + \\epsilon}}",
                mathLabel: "Batch Normalization",
                code: `model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5)
)`
              }
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
              details: {
                theory: `Convolutional Neural Networks (CNNs) exploit the spatial structure of images.
                
*   **Convolution:** A learnable filter slides over the image (dot product) to detect features like edges or textures.
*   **Pooling:** Downsamples the feature map (e.g., Max Pooling takes the largest value in a 2x2 window), reducing computation and providing translation invariance.`,
                math: "(I * K)(i, j) = \\sum_m \\sum_n I(m, n) K(i-m, j-n)",
                mathLabel: "2D Convolution Operation",
                code: `layers.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)`
              }
            },
            {
              id: "topic-resnet",
              title: "ResNet & Modern Arcs",
              type: "doc",
              icon: Network,
              description: "Solving vanishing gradients with Residual Connections.",
              details: {
                theory: `As networks got deeper, they became harder to train due to vanishing gradients. **ResNet** introduced "Skip Connections" that allow gradients to flow directly through the network.
                
Instead of learning a mapping $H(x)$, ResNet learns the residual function $F(x) = H(x) - x$, making it easier to learn the identity function.`,
                math: "y = F(x, \\{W_i\\}) + x",
                mathLabel: "Residual Block Output",
                code: `class ResidualBlock(nn.Module):
    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)`
              }
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
              details: {
                theory: `**YOLO (You Only Look Once)** treats object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.
                
Unlike multi-stage detectors (R-CNN) that propose regions first, YOLO splits the image into a grid. If an object's center falls into a grid cell, that cell is responsible for detecting it.`,
                math: "Loss = \\lambda_{coord} \\sum (x_i - \\hat{x}_i)^2 + \\sum (C_i - \\hat{C}_i)^2",
                mathLabel: "Simplified YOLO Loss",
                code: `model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model('image.jpg')`
              }
            },
            {
              id: "topic-segmentation",
              title: "Segmentation (U-Net)",
              type: "doc",
              icon: Layers,
              description: "Semantic vs Instance Segmentation.",
              details: {
                theory: `**Semantic Segmentation** classifies every pixel in an image.
                
**U-Net** is the standard architecture for this, especially in medical imaging. It features an encoder (contracting path) to capture context and a decoder (symmetric expanding path) that enables precise localization, connected by skip connections.`,
                math: "J(y, \\hat{y}) = - \\frac{1}{N} \\sum y_i \\log(\\hat{y}_i)",
                mathLabel: "Pixel-wise Cross Entropy",
                code: `# U-Net Architecture pattern
x1 = self.down1(x)
x2 = self.down2(x1)
x_up = self.up(x2)
x_out = torch.cat([x_up, x1], dim=1) # Skip Connection`
              }
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
              details: {
                theory: `Embeddings map discrete words to continuous vector spaces where geometric distance corresponds to semantic similarity.
                
**Word2Vec (Skip-gram)** trains a shallow network to predict context words given a target word. The hidden layer weights become the embeddings.`,
                math: "Vec(King) - Vec(Man) + Vec(Woman) \\approx Vec(Queen)",
                mathLabel: "Vector Arithmetic",
                code: `from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)`
              }
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
              details: {
                theory: `**BERT (Bidirectional Encoder Representations)** is an Encoder-only model trained on Masked Language Modeling (MLM). It sees the entire sentence at once, making it ideal for understanding tasks (classification, QA).
                
**GPT (Generative Pre-trained Transformer)** is a Decoder-only model trained on Causal Language Modeling (predict next word). It is auto-regressive, making it ideal for generation tasks.`,
                math: "P(w_t | w_{1:t-1})",
                mathLabel: "Autoregressive Objective",
                code: `from transformers import BertModel, GPT2Model
# BERT for understanding
bert = BertModel.from_pretrained('bert-base-uncased')
# GPT for generation
gpt = GPT2Model.from_pretrained('gpt2')`
              }
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
    """
    Args:
        q: Query matrix (batch, seq_len, d_k)
        k: Key matrix (batch, seq_len, d_k)
        v: Value matrix (batch, seq_len, d_v)
    Returns:
        values, attention_weights
    """
    d_k = q.shape[-1]
    
    # TODO: Implement step-by-step
    # 1. Matmul Q and K transpose
    # 2. Scale by sqrt(d_k)
    # 3. Softmax
    # 4. Matmul with V
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
              details: {
                theory: `Full fine-tuning of LLMs is prohibitively expensive. **PEFT (Parameter-Efficient Fine-Tuning)** methods like **LoRA** freeze the pre-trained weights $W$ and inject trainable rank decomposition matrices $A$ and $B$.
                
This reduces the number of trainable parameters by up to 10,000x while maintaining performance.`,
                math: "W' = W + \\frac{\\alpha}{r} BA",
                mathLabel: "LoRA Update Rule",
                code: `from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(base_model, config)`
              }
            },
            {
              id: "topic-rag",
              title: "RAG Systems",
              type: "doc",
              icon: Database,
              description: "Retrieval Augmented Generation with Vector DBs.",
              details: {
                theory: `**RAG** bridges the gap between an LLM's frozen knowledge and real-time private data.
                
1.  **Retrieve:** Convert user query to vector, search Vector DB for relevant context.
2.  **Augment:** Stuff retrieved context into the prompt.
3.  **Generate:** LLM answers based on the augmented prompt.`,
                math: "p(y|x) \\approx \\sum_{z \\in TopK(x)} p(y|x,z)p(z|x)",
                mathLabel: "RAG Probability Approximation",
                code: `docs = vector_db.similarity_search(query)
prompt = f"Context: {docs}\\nQuestion: {query}"
llm.predict(prompt)`
              }
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
              details: {
                theory: `**Generative Adversarial Networks (GANs)** consist of two networks playing a zero-sum game:
                
*   **Generator ($G$):** Tries to create fake images that look real.
*   **Discriminator ($D$):** Tries to distinguish between real and fake images.
                
They improve together until the Generator produces indistinguishable samples.`,
                math: "\\min_G \\max_D V(D, G) = \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]",
                mathLabel: "Minimax Loss",
                code: `# Adversarial Training Loop
loss_d = train_discriminator(real, fake)
loss_g = train_generator(fake) # labels flipped to 'real'`
              }
            },
            {
              id: "topic-diffusion",
              title: "Diffusion Models",
              type: "doc",
              icon: Box,
              description: "Stable Diffusion and Latent Space denoising.",
              details: {
                theory: `**Diffusion Models** learn to reverse a gradual noise-adding process.
                
1.  **Forward:** Slowly add Gaussian noise to an image until it is pure random noise.
2.  **Reverse:** Train a U-Net to predict the noise added at each step. By iteratively removing predicted noise, we can generate high-quality images from pure static.`,
                math: "L_{simple} = \\mathbb{E}_{t, x_0, \\epsilon} [ || \\epsilon - \\epsilon_\\theta(x_t, t) ||^2 ]",
                mathLabel: "Denoising Objective",
                code: `noise_pred = unet(latents, t, text_embeddings)
latents = scheduler.step(noise_pred, t, latents)`
              }
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
              details: {
                theory: `**DQN** successfully combined Q-Learning with Deep Neural Networks.
                
Key innovations to stabilize training:
*   **Experience Replay:** Storing transitions $(s, a, r, s')$ in a buffer and sampling randomly to break correlation.
*   **Target Network:** Using a frozen copy of the network to compute target values, updated periodically.`,
                math: "L = (r + \\gamma \\max_{a'} Q(s', a'; \\theta^-) - Q(s, a; \\theta))^2",
                mathLabel: "DQN Loss Function",
                code: `replay_buffer.push(state, action, reward, next_state)
batch = replay_buffer.sample(32)
loss = compute_td_error(batch)`
              }
            },
            {
              id: "topic-ppo",
              title: "Policy Gradients (PPO)",
              type: "doc",
              icon: TrendingUp,
              description: "Proximal Policy Optimization and Actor-Critic methods.",
              details: {
                theory: `**Proximal Policy Optimization (PPO)** is the industry standard for RL (used in RLHF for ChatGPT).
                
It is a policy gradient method that prevents "destructive updates" by clipping the probability ratio. This ensures the new policy doesn't deviate too wildly from the old policy in a single step, stabilizing training.`,
                math: "L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}_t [\\min(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t)]",
                mathLabel: "PPO Clipped Objective",
                code: `ratio = (new_log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
loss = -torch.min(surr1, surr2).mean()`
              }
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
              details: {
                theory: `Deploying a model means exposing it as a service.
                
*   **REST API (FastAPI):** Simple HTTP endpoints. Good for low traffic.
*   **ONNX (Open Neural Network Exchange):** A standard format to represent models. Allows training in PyTorch but running in highly optimized runtimes (like C++ or WebAssembly) for production speed.`,
                math: "\\text{Latency} = T_{network} + T_{preprocessing} + T_{inference} + T_{postprocessing}",
                mathLabel: "Inference Latency Components",
                code: `import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": x_numpy})`
              }
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
              details: {
                theory: `**Docker** packages code and dependencies into a container, ensuring "it works on my machine" means it works everywhere.
                
**Kubernetes (K8s)** orchestrates these containers at scale, handling auto-scaling (adding more model replicas during traffic spikes) and self-healing (restarting crashed containers).`,
                math: "N_{replicas} = \\lceil \\frac{\\text{Total RPS}}{\\text{RPS per Replica}} \\rceil",
                mathLabel: "Scaling Formula",
                code: `FROM python:3.9-slim
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]`
              }
            },
            {
              id: "topic-cicd",
              title: "CI/CD for ML",
              type: "doc",
              icon: Workflow,
              description: "Automated training pipelines and Experiment Tracking (MLflow).",
              details: {
                theory: `**CI/CD (Continuous Integration/Continuous Deployment)** applies to ML as "CT/CD" (Continuous Training).
                
1.  **Data Change:** Triggers a pipeline.
2.  **Retrain:** Model trains on new data.
3.  **Evaluate:** Auto-compares new model vs current production model.
4.  **Deploy:** If metrics improve, auto-deploy.`,
                math: "\\text{Drift}(P, Q) = KL(P || Q) = \\sum P(x) \\log \\frac{P(x)}{Q(x)}",
                mathLabel: "KL Divergence (Data Drift)",
                code: `import mlflow
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")`
              }
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
