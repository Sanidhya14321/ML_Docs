import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Task {
  id: string;
  title: string;
  completed: boolean;
}

interface TaskStore {
  tasks: Task[];
  addTask: (title: string) => void;
  toggleTask: (id: string) => void;
  deleteTask: (id: string) => void;
  reorderTasks: (startIndex: number, endIndex: number) => void;
}

export const useTaskStore = create<TaskStore>()(
  persist(
    (set) => ({
      tasks: [
        { id: '1', title: 'Complete Foundations Module', completed: false },
        { id: '2', title: 'Review Logistic Regression Math', completed: false },
        { id: '3', title: 'Finish Medical Case Study Lab', completed: false },
      ],
      addTask: (title) =>
        set((state) => ({
          tasks: [...state.tasks, { id: Date.now().toString(), title, completed: false }],
        })),
      toggleTask: (id) =>
        set((state) => ({
          tasks: state.tasks.map((t) =>
            t.id === id ? { ...t, completed: !t.completed } : t
          ),
        })),
      deleteTask: (id) =>
        set((state) => ({
          tasks: state.tasks.filter((t) => t.id !== id),
        })),
      reorderTasks: (startIndex, endIndex) =>
        set((state) => {
          const result = Array.from(state.tasks);
          const [removed] = result.splice(startIndex, 1);
          result.splice(endIndex, 0, removed);
          return { tasks: result };
        }),
    }),
    {
      name: 'ai-codex-tasks',
    }
  )
);
