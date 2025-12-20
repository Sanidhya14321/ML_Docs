
import React from 'react';
import { BookOpen, Database, Cpu, BrainCircuit, Gamepad2, Server, FlaskConical, Terminal, Activity, Layers, GitBranch, Globe, Container } from 'lucide-react';
import { Module } from '../types';

export const CURRICULUM: Module[] = [
  {
    id: 'module-1',
    title: 'Module 1: AI Engineering Foundations',
    icon: React.createElement(Database, { size: 16 }),
    chapters: [
      {
        id: 'data-stack',
        title: 'The Modern Data Stack',
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
          },
          {
            id: 'foundations/math-quiz',
            title: 'Checkpoint: Linear Algebra',
            type: 'quiz',
            description: 'Validate understanding of vector spaces and matrix operations.',
            quizConfig: {
              questions: [
                {
                  id: 'q1',
                  text: 'What does the Rank of a matrix represent?',
                  options: ['The number of rows', 'The number of linearly independent columns', 'The total sum of elements', 'The determinant value'],
                  correctIndex: 1,
                  explanation: 'Rank is the dimension of the vector space generated (or spanned) by its columns.'
                },
                {
                  id: 'q2',
                  text: 'In PCA, what do the Eigenvectors represent?',
                  options: ['The principal directions of variance', 'The magnitude of variance', 'The cluster centroids', 'The loss function gradients'],
                  correctIndex: 0,
                  explanation: 'Eigenvectors point in the direction of the greatest variance in the data.'
                },
                {
                  id: 'q3',
                  text: 'Which matrix operation is non-commutative?',
                  options: ['Addition (A+B)', 'Matrix Multiplication (AB)', 'Scalar Multiplication (cA)', 'Transpose (A^T)'],
                  correctIndex: 1,
                  explanation: 'In general, AB does not equal BA. The order of transformation matters.'
                }
              ]
            }
          }
        ]
      },
      {
        id: 'etl-pipelines',
        title: 'ETL & Orchestration',
        topics: [
          {
            id: 'mlops/airflow',
            title: 'DAGs with Airflow',
            type: 'doc',
            description: 'Defining dependencies and scheduling workflows as code.',
            details: {
              theory: 'Directed Acyclic Graphs (DAGs) model the logical flow of data tasks. Airflow serves as the scheduler, ensuring tasks like data extraction, validation, and model training occur in the correct topological order. We handle retries, backfills, and SLA alerts programmatically.',
              math: 'G = (V, E) \\text{ where } \\forall v \\in V, (v, v) \\notin E^*',
              mathLabel: 'Acyclic Property',
              code: 'from airflow import DAG\nfrom airflow.operators.python import PythonOperator\n\nwith DAG("model_train", schedule_interval="@daily") as dag:\n    t1 = PythonOperator(task_id="extract", python_callable=extract_data)\n    t2 = PythonOperator(task_id="train", python_callable=train_model)\n    t1 >> t2',
              codeLanguage: 'python'
            }
          },
          {
            id: 'mlops/data-lab',
            title: 'Lab: Data Cleaning Pipeline',
            type: 'lab',
            description: 'Build a robust Pandas pipeline to handle missing values and outliers.',
            labConfig: {
              initialCode: `import pandas as pd\nimport numpy as np\n\ndata = pd.DataFrame({\n    'age': [25, 30, np.nan, 150, 40],\n    'income': [50000, 60000, 55000, -1000, 100000000]\n})\n\ndef clean_data(df):\n    # 1. Fill NA age with median\n    \n    # 2. Filter invalid income (< 0)\n    \n    # 3. Cap income outliers at 99th percentile\n    \n    return df\n\ncleaned = clean_data(data)\nprint(cleaned)`,
              solution: `def clean_data(df):\n    df['age'] = df['age'].fillna(df['age'].median())\n    df = df[df['income'] >= 0]\n    cap = df['income'].quantile(0.99)\n    df['income'] = df['income'].clip(upper=cap)\n    return df`,
              hints: ['Use fillna()', 'Boolean indexing for filtering', 'Use quantile() and clip()']
            }
          }
        ]
      }
    ]
  },
  {
    id: 'module-2',
    title: 'Module 2: Model Engineering',
    icon: React.createElement(BrainCircuit, { size: 16 }),
    chapters: [
      {
        id: 'deep-learning',
        title: 'Deep Learning Architectures',
        topics: [
          {
            id: 'dl/transformers-arch',
            title: 'Transformer Architecture',
            type: 'doc',
            description: 'The mechanism behind LLMs: Attention, Normalization, and Feed-Forward networks.',
            details: {
              theory: 'Transformers discard recurrence for pure attention. The Encoder processes the input context, while the Decoder generates output. Layer Normalization and Residual Connections are critical for gradient flow in deep stacks. Positional encodings inject sequence order information.',
              math: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
              mathLabel: 'Position-wise Feed-Forward',
              code: 'class TransformerBlock(nn.Module):\n    def __init__(self, embed_dim, num_heads):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(embed_dim, num_heads)\n        self.norm1 = nn.LayerNorm(embed_dim)\n        self.ffn = nn.Linear(embed_dim, embed_dim)',
              codeLanguage: 'python'
            }
          },
          {
            id: 'dl/pytorch-training-lab',
            title: 'Lab: Training Loop',
            type: 'lab',
            description: 'Implement a full PyTorch training loop with batches and optimization.',
            labConfig: {
              initialCode: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Dummy Data\nX = torch.randn(100, 10)\ny = torch.randint(0, 2, (100,)).float().unsqueeze(1)\n\nmodel = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())\n\n# TODO: Define Optimizer (SGD) and Loss (BCELoss)\noptimizer = \ncriterion = \n\n# Training Step\noptimizer.zero_grad()\npred = model(X)\nloss = criterion(pred, y)\nloss.backward()\noptimizer.step()\n\nprint(f"Loss: {loss.item()}")`,
              solution: `optimizer = optim.SGD(model.parameters(), lr=0.01)\ncriterion = nn.BCELoss()\noptimizer.zero_grad()\npred = model(X)\nloss = criterion(pred, y)\nloss.backward()\noptimizer.step()`,
              hints: ['Use optim.SGD', 'Use nn.BCELoss for binary classification', 'Remember zero_grad() before backward()']
            }
          }
        ]
      },
      {
        id: 'generative-ai',
        title: 'Generative AI & LLMs',
        topics: [
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
    id: 'module-3',
    title: 'Module 3: MLOps & Production',
    icon: React.createElement(Server, { size: 16 }),
    chapters: [
      {
        id: 'deployment',
        title: 'Model Deployment',
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
          }
        ]
      },
      {
        id: 'monitoring',
        title: 'Observability',
        topics: [
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
