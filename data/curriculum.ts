
import React from 'react';
import { BookOpen, Database, Cpu, BrainCircuit, Gamepad2, Server, FlaskConical, Terminal, Activity, Layers, GitBranch, Globe, Container, Swords, Network, TrendingUp, Search } from 'lucide-react';
import { Module } from '../types';

export const CURRICULUM: Module[] = [
  {
    id: 'module-1',
    title: 'Module 1: Foundations',
    icon: React.createElement(BookOpen, { size: 16 }),
    chapters: [
      {
        id: 'math-foundations',
        title: 'Mathematical Core',
        topics: [
          {
            id: 'foundations',
            title: 'Linear Algebra Primer',
            type: 'doc',
            description: 'The geometric interpretation of data: Vectors, Matrices, and Dot Products.',
          },
          {
            id: 'optimization',
            title: 'Optimization Engines',
            type: 'doc',
            description: 'Gradient Descent and Backtracking: How machines learn by minimizing error.',
          }
        ]
      },
      {
        id: 'data-engineering',
        title: 'Data Engineering',
        topics: [
          {
            id: 'foundations/python-advanced',
            title: 'High-Performance Python',
            type: 'doc',
            description: 'Optimizing Python for data throughput using vectorization and memory management.',
            details: {
              theory: 'Standard Python lists involve significant pointer overhead. In AI engineering, we leverage NumPy strides and contiguous memory blocks to achieve near C-level speeds. Understanding the Global Interpreter Lock (GIL) and multiprocessing is crucial for data loading pipelines.',
              math: '\\text{Speedup} = \\frac{1}{(1-P) + \\frac{P}{N}}',
              mathLabel: 'Amdahl\'s Law',
              code: 'import numpy as np\n# Vectorized operation (100x faster than loops)\ndata = np.random.rand(1000000)\nresult = np.log(data) * np.exp(data)',
              codeLanguage: 'python'
            }
          },
          {
            id: 'foundations/sql-pipelines',
            title: 'SQL for Data Engineers',
            type: 'lab',
            description: 'Constructing analytical datasets using Window Functions and CTEs.',
            labConfig: {
              initialCode: `import sqlite3\nimport pandas as pd\n\n# Create in-memory DB\nconn = sqlite3.connect(':memory:')\ndf = pd.DataFrame({'user': ['A','A','B','B'], 'val': [10, 20, 30, 40], 'date': [1,2,1,2]})\ndf.to_sql('metrics', conn, index=False)\n\n# TODO: Write a query to calculate cumulative sum per user\nquery = """\nSELECT \n    user,\n    val,\n    -- Add Window Function Here\n    SUM(val) OVER (PARTITION BY ... ORDER BY ...) as running_total\nFROM metrics\n"""\n\nprint(pd.read_sql(query, conn))`,
              solution: `query = "SELECT user, val, SUM(val) OVER (PARTITION BY user ORDER BY date) as running_total FROM metrics"`,
              hints: ['Use PARTITION BY user', 'Use ORDER BY date', 'SUM(val) OVER (...)']
            }
          }
        ]
      }
    ]
  },
  {
    id: 'module-2',
    title: 'Module 2: Supervised Learning',
    icon: React.createElement(TrendingUp, { size: 16 }),
    chapters: [
      {
        id: 'classical-ml',
        title: 'Classical Algorithms',
        topics: [
          {
            id: 'regression',
            title: 'Regression Analysis',
            type: 'doc',
            description: 'Predicting continuous variables using Linear, Ridge, Lasso, and Polynomial models.',
          },
          {
            id: 'classification',
            title: 'Classification Logic',
            type: 'doc',
            description: 'Decision boundaries: Logistic Regression, KNN, SVM, and Naive Bayes.',
          },
          {
            id: 'ensemble',
            title: 'Ensemble Methods',
            type: 'doc',
            description: 'Random Forests and Gradient Boosting: Combining weak learners for robustness.',
          },
          {
            id: 'battleground',
            title: 'Model Battleground',
            type: 'doc',
            description: 'Direct comparison of algorithm performance, training time, and interpretability.',
          }
        ]
      }
    ]
  },
  {
    id: 'module-3',
    title: 'Module 3: Neural Intelligence',
    icon: React.createElement(BrainCircuit, { size: 16 }),
    chapters: [
      {
        id: 'deep-architectures',
        title: 'Deep Architectures',
        topics: [
          {
            id: 'unsupervised',
            title: 'Unsupervised Learning',
            type: 'doc',
            description: 'Clustering and Dimensionality Reduction: K-Means, Hierarchical, and t-SNE.',
          },
          {
            id: 'deep-learning',
            title: 'Deep Neural Networks',
            type: 'doc',
            description: 'MLPs, CNNs for Vision, RNNs for Sequence, and Embeddings.',
          },
          {
            id: 'reinforcement',
            title: 'Reinforcement Learning',
            type: 'doc',
            description: 'Agents, Environments, Q-Learning, and Actor-Critic methods.',
          }
        ]
      },
      {
        id: 'generative-ai',
        title: 'Generative AI & LLMs',
        topics: [
          {
            id: 'deep-learning/attention-mechanism',
            title: 'The Attention Mechanism',
            type: 'doc',
            description: 'The mathematical heart of Transformers and LLMs.',
          },
          {
            id: 'genai/diffusion',
            title: 'Diffusion Models',
            type: 'doc',
            description: 'Generating data by reversing a gradual noise addition process.',
            details: {
              theory: 'Diffusion models learn to reverse a Markov chain that adds Gaussian noise to data. The forward process destroys information (image -> noise), and the reverse process creates information (noise -> image). Training involves predicting the noise added at each timestep.',
              math: 'L_t(\\theta) = ||\\epsilon - \\epsilon_\\theta(\\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon, t)||^2',
              mathLabel: 'Denoising Objective',
              code: '# Forward diffusion sample\nnoise = torch.randn_like(x_0)\nnoisy_image = sqrt_alpha_cumprod[t] * x_0 + sqrt_one_minus_alpha[t] * noise',
              codeLanguage: 'python'
            }
          },
          {
            id: 'genai/prompt-quiz',
            title: 'Checkpoint: LLM Concepts',
            type: 'quiz',
            description: 'Test your knowledge on Tokenization, Context Windows, and Temperature.',
            quizConfig: {
              questions: [
                {
                  id: 'q1',
                  text: 'What is the role of Temperature in LLM sampling?',
                  options: ['It controls the learning rate', 'It controls the randomness of predictions', 'It sets the context length', 'It adjusts the model weights'],
                  correctIndex: 1,
                  explanation: 'Higher temperature flattens the probability distribution, making lower probability tokens more likely to be sampled (more creative).'
                },
                {
                  id: 'q2',
                  text: 'Why do we use Byte-Pair Encoding (BPE) or WordPiece?',
                  options: ['To encrypt data', 'To handle out-of-vocabulary words efficiently', 'To remove stop words', 'To increase dataset size'],
                  correctIndex: 1,
                  explanation: 'Subword tokenization breaks unknown words into known sub-units, allowing the model to process rare terms.'
                },
                {
                  id: 'q3',
                  text: 'What limits the Context Window of a standard Transformer?',
                  options: ['Disk space', 'Quadratic memory complexity of Self-Attention', 'Number of layers', 'Vocabulary size'],
                  correctIndex: 1,
                  explanation: 'The attention matrix grows as N^2 with sequence length N, consuming massive GPU VRAM.'
                }
              ]
            }
          }
        ]
      }
    ]
  },
  {
    id: 'module-4',
    title: 'Module 4: MLOps & Engineering',
    icon: React.createElement(Server, { size: 16 }),
    chapters: [
      {
        id: 'deployment',
        title: 'Production Systems',
        topics: [
          {
            id: 'mlops/fastapi',
            title: 'Real-time Inference APIs',
            type: 'doc',
            description: 'Wrapping models in REST APIs using FastAPI and Pydantic.',
            details: {
              theory: 'For real-time serving, models are containerized and exposed via HTTP. FastAPI provides high-performance asynchronous handling. We must consider serialization overhead, latency budgets, and concurrency management (Uvicorn workers).',
              math: '\\text{Latency} = T_{net} + T_{deserialize} + T_{inference} + T_{serialize}',
              mathLabel: 'Request Latency Model',
              code: 'from fastapi import FastAPI\napp = FastAPI()\n\n@app.post("/predict")\nasync def predict(payload: InputSchema):\n    vector = preprocess(payload)\n    return {"pred": model.predict(vector).tolist()}',
              codeLanguage: 'python'
            }
          },
          {
            id: 'mlops/docker-lab',
            title: 'Lab: Dockerizing Models',
            type: 'lab',
            description: 'Write a Dockerfile to containerize a Python ML application.',
            labConfig: {
              initialCode: `# Complete the Dockerfile\n\n# 1. Base Image (python:3.9-slim)\nFROM \n\n# 2. Set work directory\nWORKDIR /app\n\n# 3. Copy requirements and install\nCOPY requirements.txt .\nRUN \n\n# 4. Copy app code\nCOPY . .\n\n# 5. Command to run app\nCMD ["python", "app.py"]`,
              solution: `FROM python:3.9-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD ["python", "app.py"]`,
              hints: ['Use python:3.9-slim', 'RUN pip install', 'CMD needs a list of strings']
            }
          },
          {
            id: 'mlops/drift',
            title: 'Data Drift Detection',
            type: 'doc',
            description: 'Detecting distributional shifts between training and production data.',
            details: {
              theory: 'Models degrade over time as the world changes (Concept Drift) or input data changes (Covariate Shift). We use statistical tests like Kolmogorov-Smirnov (KS) or Population Stability Index (PSI) to trigger retraining pipelines automatically.',
              math: 'D_{KL}(P || Q) = \\sum P(x) \\log \\frac{P(x)}{Q(x)}',
              mathLabel: 'Kullback-Leibler Divergence',
              code: 'from scipy.stats import ks_2samp\n\nstat, p_value = ks_2samp(train_data, prod_data)\nif p_value < 0.05:\n    trigger_retrain()',
              codeLanguage: 'python'
            }
          }
        ]
      },
      {
        id: 'certification',
        title: 'Certification',
        topics: [
          {
            id: 'lab',
            title: 'Capstone: Medical Case Study',
            type: 'lab',
            description: 'End-to-end project: EDA, Model Selection, and Performance Analysis on clinical heart disease data.',
          },
          {
            id: 'mlops/final-exam',
            title: 'Final Certification Exam',
            type: 'quiz',
            description: 'Comprehensive assessment covering Architecture, MLOps, and Theory.',
            quizConfig: {
              passingScore: 80,
              questions: [
                {
                  id: 'f1',
                  text: 'Which deployment strategy directs a small % of traffic to the new model?',
                  options: ['Blue/Green', 'Canary', 'Shadow Mode', 'A/B Testing'],
                  correctIndex: 1,
                  explanation: 'Canary deployment releases the change to a small subset of users to reduce risk before full rollout.'
                },
                {
                  id: 'f2',
                  text: 'What happens if you increase the batch size significantly without adjusting learning rate?',
                  options: ['Faster convergence', 'Generalization gap (worse test accuracy)', 'Model collapse', 'Gradient explosion'],
                  correctIndex: 1,
                  explanation: 'Large batch training tends to converge to sharp minimas, leading to poorer generalization unless heuristics like Linear Scaling Rule are used.'
                },
                {
                  id: 'f3',
                  text: 'In Docker, which instruction minimizes layer size?',
                  options: ['Using multiple RUN commands', 'Chaining commands with &&', 'Copying all files', 'Using Ubuntu base'],
                  correctIndex: 1,
                  explanation: 'Chaining commands prevents the creation of intermediate temporary layers, keeping the image small.'
                },
                {
                  id: 'f4',
                  text: 'What is the primary benefit of Feature Stores?',
                  options: ['Cheaper storage', 'Consistency between training and inference', 'Faster model training', 'Auto-ML'],
                  correctIndex: 1,
                  explanation: 'Feature Stores ensure the exact same feature logic serves both offline training and online inference, preventing training-serving skew.'
                }
              ]
            }
          }
        ]
      }
    ]
  }
];
