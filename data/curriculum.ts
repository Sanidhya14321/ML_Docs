
import { 
  BrainCircuit, 
  TrendingUp, 
  Binary, 
  Network, 
  Layers, 
  Box, 
  Database,
  Calculator,
  Search,
  Zap,
  Swords,
  FlaskConical
} from 'lucide-react';
import { Course, ViewSection } from '../types';

export const CURRICULUM: Course = {
  id: "ai-codex-v3",
  title: "AI Engineering Certification",
  description: "Master the art of machine learning through interactive visualizations and rigorous mathematical foundations.",
  modules: [
    {
      id: "mod-foundations",
      title: "Foundations",
      icon: Calculator,
      chapters: [
        {
          id: "chap-math",
          title: "Mathematical Core",
          topics: [
            {
              id: ViewSection.FOUNDATIONS,
              title: "Linear Algebra & Calculus",
              type: "doc",
              icon: Calculator,
              description: "The mathematical language of vectors, matrices, and gradients."
            },
            {
              id: ViewSection.OPTIMIZATION,
              title: "Optimization Engines",
              type: "doc",
              icon: Zap,
              description: "Gradient descent and the search for minima."
            }
          ]
        }
      ]
    },
    {
      id: "mod-supervised",
      title: "Supervised Learning",
      icon: TrendingUp,
      chapters: [
        {
          id: "chap-regression",
          title: "Regression",
          topics: [
            {
              id: ViewSection.REGRESSION,
              title: "Linear & Poly Regression",
              type: "doc",
              description: "Predicting continuous values."
            }
          ]
        },
        {
          id: "chap-classification",
          title: "Classification",
          topics: [
            {
              id: ViewSection.CLASSIFICATION,
              title: "Classification Models",
              type: "doc",
              icon: Binary,
              description: "Logistic Regression, SVM, KNN, and Decision Trees."
            },
            {
              id: ViewSection.ENSEMBLE,
              title: "Ensemble Methods",
              type: "doc",
              icon: Layers,
              description: "Random Forests and Gradient Boosting."
            },
            {
              id: ViewSection.MODEL_COMPARISON,
              title: "The Battleground",
              type: "doc",
              icon: Swords,
              description: "Direct comparison of algorithms."
            }
          ]
        }
      ]
    },
    {
      id: "mod-unsupervised",
      title: "Unsupervised Learning",
      icon: Network,
      chapters: [
        {
          id: "chap-clustering",
          title: "Clustering & Dim Reduction",
          topics: [
            {
              id: ViewSection.UNSUPERVISED,
              title: "Clustering Algorithms",
              type: "doc",
              description: "K-Means, Hierarchical, and t-SNE."
            }
          ]
        }
      ]
    },
    {
      id: "mod-deep-learning",
      title: "Deep Learning",
      icon: BrainCircuit,
      chapters: [
        {
          id: "chap-nn",
          title: "Neural Architectures",
          topics: [
            {
              id: ViewSection.DEEP_LEARNING,
              title: "Neural Networks",
              type: "doc",
              description: "MLP, CNN, RNN, and Transformers."
            },
            {
              id: "deep-learning/attention-mechanism",
              title: "Attention Mechanism",
              type: "doc",
              icon: Search,
              description: "The engine behind modern LLMs.",
              details: {
                theory: `**Attention** allows models to focus on specific parts of the input sequence when generating output, rather than treating all inputs equally. It calculates a weighted sum of input vectors, where weights are determined by the compatibility (dot product) between a query and keys.`,
                math: "Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V",
                mathLabel: "Scaled Dot-Product Attention",
                code: `def attention(query, key, value):
    score = matmul(query, key.transpose()) / sqrt(d_k)
    p_attn = softmax(score)
    return matmul(p_attn, value)`,
                codeLanguage: "python"
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

### Data Pipeline
1.  **Retrieve:** Convert user query to vector, search Vector DB for relevant context.
2.  **Augment:** Stuff retrieved context into the prompt.
3.  **Generate:** LLM answers based on the augmented prompt.

### Vector Databases
Vector databases (like **Pinecone**, **Chroma**, **Milvus**) are specialized storage engines designed to index and search high-dimensional vectors.`,
                math: "p(y|x) \\approx \\sum_{z \\in TopK(x)} p(y|x,z)p(z|x)",
                mathLabel: "RAG Probability Approximation",
                code: `# Pseudo-code for Vector DB interaction
# 1. Convert query to vector embedding
query_vec = embedding_model.encode("How do I fix a leaking pipe?")

# 2. Semantic Search
results = vector_db.query(
    vector=query_vec, 
    top_k=3
)

# 3. Generate
llm.predict(f"Context: {results}\\nQuestion: {query}")`,
                codeLanguage: "python"
              }
            }
          ]
        }
      ]
    },
    {
      id: "mod-advanced",
      title: "Advanced Paradigms",
      icon: Box,
      chapters: [
        {
          id: "chap-rl",
          title: "Reinforcement Learning",
          topics: [
            {
              id: ViewSection.REINFORCEMENT,
              title: "RL Agents",
              type: "doc",
              description: "Q-Learning, Policy Gradients, and Actor-Critic."
            }
          ]
        }
      ]
    },
    {
      id: "mod-lab",
      title: "Project Lab",
      icon: FlaskConical,
      chapters: [
        {
          id: "chap-projects",
          title: "Applied Projects",
          topics: [
            {
              id: ViewSection.PROJECT_LAB,
              title: "Medical Diagnostic AI",
              type: "lab",
              description: "End-to-end case study on heart disease classification.",
              labConfig: {
                initialCode: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('heart_vitals.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test)}")`,
                solution: `print("Solution hidden")`,
                hints: ["Check for data scaling", "Use cross-validation"]
              }
            }
          ]
        }
      ]
    }
  ]
};
