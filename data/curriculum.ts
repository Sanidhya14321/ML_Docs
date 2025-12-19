
import React from 'react';
import { BookOpen, Database, Cpu, BrainCircuit, Gamepad2, Server, FlaskConical } from 'lucide-react';
import { Module } from '../types';

export const CURRICULUM: Module[] = [
  {
    id: 'math-foundations',
    title: 'Mathematical Foundations',
    icon: React.createElement(BookOpen, { size: 16 }),
    chapters: [
      {
        id: 'linear-algebra',
        title: 'Linear Algebra',
        topics: [
          {
            id: 'foundations',
            title: 'Vectors & Spaces',
            type: 'doc',
            description: 'Understanding the fundamental building blocks of linear algebra: vectors, vector spaces, and linear independence.'
          },
          {
            id: 'matrix-ops',
            title: 'Matrix Operations',
            type: 'doc',
            description: 'Deep dive into matrix multiplication, transformations, and determinants.'
          },
          {
            id: 'numpy-lab',
            title: 'Lab: NumPy Tensor Ops',
            type: 'lab',
            description: 'Hands-on practice with NumPy for high-performance vector computation.',
            labConfig: {
              initialCode: `import numpy as np\n\n# 1. Create a 3x3 Identity Matrix\nI = \n\n# 2. Create a vector v = [1, 2, 3]\nv = \n\n# 3. Compute dot product of v and itself\ndot_prod = \n\nprint(f"Identity:\\n{I}")\nprint(f"Dot Product: {dot_prod}")`,
              solution: `import numpy as np\nI = np.eye(3)\nv = np.array([1, 2, 3])\ndot_prod = np.dot(v, v)\nprint(f"Identity:\\n{I}")\nprint(f"Dot Product: {dot_prod}")`,
              hints: ['Use np.eye() for identity', 'Use np.dot() or @ operator']
            }
          },
          {
            id: 'linear-algebra-quiz',
            title: 'Checkpoint: Linear Algebra',
            type: 'quiz',
            description: 'Verify your understanding of vectors, matrices, and NumPy operations.',
            quizConfig: {
              questions: [
                {
                  id: 'q1',
                  text: 'What is the result of the dot product between v=[1, 2] and u=[3, 4]?',
                  options: ['10', '11', '7', '[3, 8]'],
                  correctIndex: 1,
                  explanation: 'The dot product is (1*3) + (2*4) = 3 + 8 = 11.'
                },
                {
                  id: 'q2',
                  text: 'Which NumPy function creates an Identity Matrix?',
                  options: ['np.identity()', 'np.eye()', 'np.ones()', 'np.diag()'],
                  correctIndex: 1,
                  explanation: 'np.eye(n) creates an n x n identity matrix.'
                },
                {
                  id: 'q3',
                  text: 'If matrix A is shape (3, 2) and B is shape (2, 4), what is the shape of A @ B?',
                  options: ['(3, 4)', '(2, 2)', '(3, 2)', 'Error'],
                  correctIndex: 0,
                  explanation: 'Matrix multiplication (m, n) x (n, p) results in (m, p).'
                }
              ]
            }
          }
        ]
      },
      {
        id: 'calculus',
        title: 'Calculus',
        topics: [
          { id: 'optimization', title: 'Gradient Descent', type: 'doc', description: 'The engine of learning: minimizing loss functions via gradients.' },
          { id: 'backprop-lab', title: 'Lab: Autograd from Scratch', type: 'lab', description: 'Implement a basic automatic differentiation engine.', labConfig: { initialCode: '# TODO: Implement Node class for autograd', solution: '', hints: [] } }
        ]
      }
    ]
  },
  {
    id: 'classical-ml',
    title: 'Classical Machine Learning',
    icon: React.createElement(Cpu, { size: 16 }),
    chapters: [
      {
        id: 'supervised',
        title: 'Supervised Learning',
        topics: [
          { id: 'regression', title: 'Linear Regression', type: 'doc', description: 'Predicting continuous values using least squares.' },
          { id: 'classification', title: 'Classification (SVM/LogReg)', type: 'doc', description: 'Separating classes with hyperplanes and probability.' },
          { id: 'ensemble', title: 'Ensemble Methods', type: 'doc', description: 'Combining weak learners: Random Forests and Gradient Boosting.' },
          { 
            id: 'sklearn-lab', 
            title: 'Lab: Scikit-Learn Pipeline', 
            type: 'lab',
            description: 'Build a production-ready training pipeline with preprocessing.',
            labConfig: {
              initialCode: `from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\n# TODO: Define steps\nsteps = []\n\npipeline = Pipeline(steps)`,
              solution: '',
              hints: []
            } 
          }
        ]
      },
      {
        id: 'unsupervised',
        title: 'Unsupervised Learning',
        topics: [
          { id: 'unsupervised', title: 'Clustering & Dim Reduction', type: 'doc', description: 'Finding hidden structures: K-Means, PCA, and t-SNE.' }
        ]
      }
    ]
  },
  {
    id: 'deep-learning',
    title: 'Deep Learning',
    icon: React.createElement(BrainCircuit, { size: 16 }),
    chapters: [
      {
        id: 'neural-nets',
        title: 'Neural Networks',
        topics: [
          { id: 'deep-learning', title: 'Perceptrons & MLP', type: 'doc', description: 'The architecture of artificial neurons and layers.' },
          { id: 'deep-learning/attention-mechanism', title: 'The Attention Mechanism', type: 'doc', description: 'The math behind Transformers and LLMs.' }
        ]
      }
    ]
  },
  {
    id: 'reinforcement',
    title: 'Reinforcement Learning',
    icon: React.createElement(Gamepad2, { size: 16 }),
    chapters: [
      {
        id: 'rl-basics',
        title: 'Foundations',
        topics: [
          { id: 'reinforcement', title: 'MDPs & Q-Learning', type: 'doc', description: 'Agents, environments, and reward maximization.' }
        ]
      }
    ]
  },
  {
    id: 'labs',
    title: 'Project Labs',
    icon: React.createElement(FlaskConical, { size: 16 }),
    chapters: [
      {
        id: 'capstones',
        title: 'Capstones',
        topics: [
          { id: 'battleground', title: 'Algorithm Battleground', type: 'doc', description: 'Interactive comparison of model performance.' },
          { id: 'lab', title: 'Clinical Case Study', type: 'doc', description: 'End-to-end medical diagnosis project.' }
        ]
      }
    ]
  }
];
