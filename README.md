
# AI Mastery Hub (AI Codex)

**AI Mastery Hub** is an advanced, interactive educational platform designed to teach Machine Learning and Artificial Intelligence engineering. Unlike static documentation, it features real-time algorithm visualizations, an interactive coding lab, and a gamified progress tracking system.

![AI Codex Banner](https://via.placeholder.com/1200x400/0f172a/6366f1?text=AI+Mastery+Hub)

## 🚀 Key Features

### 🧠 Interactive Learning Engine
*   **Algorithm Visualizations:** Real-time, interactive graphs for Gradient Descent, Neural Networks, K-Means Clustering, and more using `Recharts` and SVG.
*   **Deep Learning Simulations:** Visualizations for Backpropagation, CNN Convolutions, and Transformer Attention mechanisms.
*   **Math & Theory:** Integrated LaTeX rendering via `KaTeX` for rigorous mathematical explanations.

### 💻 Hands-On Project Lab
*   **In-Browser IDE:** A custom code editor with syntax highlighting (`PrismJS`) and a simulated Python runtime console.
*   **Case Studies:** End-to-end projects (e.g., Medical Heart Disease Classification) where users perform EDA, model selection, and training.
*   **Persistent Workspace:** Code changes are saved locally, allowing users to leave and return without losing progress.

### 🧭 Navigation & Discovery
*   **Smart Search:** `Cmd+K` global search modal to jump to any topic instantly.
*   **Dynamic Breadcrumbs:** Context-aware navigation trails.
*   **Progress Tracking:** LocalStorage-based tracking of completed topics, module progress bars, and "Resume Learning" functionality.

### 🎨 UI/UX
*   **Theming:** First-class Dark Mode support (default) with a toggle for Light Mode.
*   **Responsive Design:** Fully mobile-optimized with collapsible sidebars and touch-friendly controls.
*   **Animations:** Smooth transitions using `Framer Motion` for route changes, modals, and interactive elements.

---

## 🛠 Tech Stack

### Core Framework
*   **React 18:** Component-based UI architecture.
*   **TypeScript:** Type-safe development for robust code.
*   **Vite:** High-performance build tool and dev server.

### Styling & Animation
*   **Tailwind CSS:** Utility-first CSS framework (configured via CDN/JS for portability in this specific iteration).
*   **Framer Motion:** Complex animations, layout transitions, and gesture handling.
*   **Lucide React:** Consistent, clean iconography.

### Visualization & Data
*   **Recharts:** Composable charting library built on React components.
*   **KaTeX:** Fast LaTeX math typesetting.
*   **PrismJS:** Lightweight, robust syntax highlighting.

### State Management
*   **React Context API:** Used for Global Theme (`ThemeContext`) and Course Progress (`CourseContext`).
*   **Custom Hooks:** `useCodeRunner` (simulation), `useQuiz` (assessment logic), `useCourseProgress`.

---

## 📂 Project Structure

```bash
/
├── components/          # Reusable UI components
│   ├── AlgorithmCard    # Standard layout for ML topics
│   ├── CodeEditor       # Lab workspace editor
│   ├── Visualizations   # (NeuralNetworkViz, etc.)
│   └── ...
├── content/             # Lazy-loaded documentation modules
├── contexts/            # Global state providers
├── data/                # Static curriculum data (modules, chapters)
├── hooks/               # Custom React hooks
├── lib/                 # Utilities (navigation helpers, validation)
├── views/               # Main page views (Dashboard, Foundations, etc.)
├── App.tsx              # Main entry, Routing logic
├── index.html           # HTML entry point (Tailwind/KaTeX CDNs)
└── ...config files
```

---

## 🗺 Routing & Architecture

The application uses a custom **Hash-based Routing** system for simplicity and static hosting compatibility (e.g., GitHub Pages).

### Route Registry (`types.ts` & `App.tsx`)

| Route | View Component | Description |
| :--- | :--- | :--- |
| `#/dashboard` | `Dashboard` | Main hub showing overall progress and modules. |
| `#/foundations` | `FoundationsView` | Linear Algebra, Calculus, Probability. |
| `#/optimization` | `OptimizationView` | Gradient Descent, Convex Optimization. |
| `#/regression` | `RegressionView` | Linear/Poly Regression visualizations. |
| `#/classification` | `ClassificationView` | SVM, KNN, Decision Trees. |
| `#/unsupervised` | `UnsupervisedView` | K-Means, DBSCAN, PCA. |
| `#/deep-learning` | `DeepLearningView` | Neural Networks, Backprop, RNNs. |
| `#/lab/{topicId}` | `LabWorkspace` | Full-screen coding environment. |
| `#/battleground` | `ModelComparisonView` | Algorithm trade-off analysis tool. |

---

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/ai-mastery-hub.git
    cd ai-mastery-hub
    ```

2.  **Install Dependencies**
    ```bash
    npm install
    ```

3.  **Run Development Server**
    ```bash
    npm run dev
    ```

4.  **Build for Production**
    ```bash
    npm run build
    ```

---

## 🔄 Recent Updates

*   **Vercel Deployment Fix:** Migrated from browser-native ES modules to a standard Vite build pipeline (`package.json`, `vite.config.ts`).
*   **Robust Error Handling:** Added try/catch blocks and visual error states to the Sudoku/CSP visualization.
*   **Enhanced Loading State:** Replaced basic loading spinner with a sophisticated `PageLoader` featuring neural animations during Suspense fallbacks.
*   **Lab Environment:** Added `LabWorkspace` with console output simulation and auto-completion triggers.

---

## 📝 License

This project is open-source and available under the MIT License.
