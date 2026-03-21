import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useTaskStore, Task } from '../stores/useTaskStore';
import { CheckCircle, Circle, GripVertical, Plus, Trash2, ListTodo } from 'lucide-react';
import { cn } from '../lib/utils';

interface SortableTaskItemProps {
  task: Task;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
}

const SortableTaskItem: React.FC<SortableTaskItemProps> = ({ task, onToggle, onDelete }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: task.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 10 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "flex items-center gap-4 p-4 bg-surface border border-border-strong rounded-none group transition-colors",
        isDragging ? "opacity-50 shadow-xl border-brand" : "hover:border-border-subtle"
      )}
    >
      <div
        {...attributes}
        {...listeners}
        className="cursor-grab active:cursor-grabbing text-text-muted hover:text-brand transition-colors p-1"
      >
        <GripVertical size={16} />
      </div>

      <button
        onClick={() => onToggle(task.id)}
        className={cn(
          "shrink-0 transition-colors",
          task.completed ? "text-emerald-500" : "text-text-muted hover:text-brand"
        )}
      >
        {task.completed ? <CheckCircle size={20} /> : <Circle size={20} />}
      </button>

      <span
        className={cn(
          "flex-1 font-mono text-sm tracking-tight truncate transition-all",
          task.completed ? "text-text-muted line-through decoration-emerald-500/30" : "text-text-primary"
        )}
      >
        {task.title}
      </span>

      <button
        onClick={() => onDelete(task.id)}
        className="text-text-muted hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-all p-2"
        aria-label="Delete task"
      >
        <Trash2 size={16} />
      </button>
    </div>
  );
};

export const TasksView: React.FC = () => {
  const { tasks, addTask, toggleTask, deleteTask, reorderTasks } = useTaskStore();
  const [newTaskTitle, setNewTaskTitle] = useState('');

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5, // 5px movement required to start drag
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = tasks.findIndex((t) => t.id === active.id);
      const newIndex = tasks.findIndex((t) => t.id === over.id);
      reorderTasks(oldIndex, newIndex);
    }
  };

  const handleAddTask = (e: React.FormEvent) => {
    e.preventDefault();
    if (newTaskTitle.trim()) {
      addTask(newTaskTitle.trim());
      setNewTaskTitle('');
    }
  };

  const completedCount = tasks.filter(t => t.completed).length;
  const progress = tasks.length === 0 ? 0 : Math.round((completedCount / tasks.length) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto px-6 py-12 md:py-20"
    >
      <header className="mb-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 rounded-none border border-brand/30 bg-brand/5 flex items-center justify-center text-brand">
            <ListTodo size={32} />
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-heading font-black text-text-primary uppercase tracking-tight">
              Study Plan
            </h1>
            <p className="text-text-secondary mt-2 text-lg font-light">
              Manage your learning objectives and track your progress. Drag and drop to reorder tasks.
            </p>
          </div>
        </div>

        <div className="bg-surface border border-border-strong rounded-none p-6 flex items-center gap-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-16 h-16 bg-brand/5 rotate-45 translate-x-8 -translate-y-8" />
          <div className="flex-1 relative z-10">
            <div className="flex justify-between items-end mb-2">
              <span className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">TASK_COMPLETION</span>
              <span className="text-2xl font-heading font-black text-brand">{progress}%</span>
            </div>
            <div className="h-1 bg-border-subtle rounded-none overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="h-full bg-brand relative"
              >
                <div className="absolute top-0 right-0 w-1 h-full bg-white animate-pulse" />
              </motion.div>
            </div>
          </div>
        </div>
      </header>

      <form onSubmit={handleAddTask} className="mb-8 flex gap-4">
        <input
          type="text"
          value={newTaskTitle}
          onChange={(e) => setNewTaskTitle(e.target.value)}
          placeholder="Add a new task..."
          className="flex-1 bg-surface border border-border-strong p-4 text-sm font-mono text-text-primary focus:outline-none focus:border-brand transition-colors rounded-none placeholder:text-text-muted"
        />
        <button
          type="submit"
          disabled={!newTaskTitle.trim()}
          className="bg-brand text-app px-6 font-mono font-black text-xs uppercase tracking-widest hover:bg-brand/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          <Plus size={16} /> ADD_TASK
        </button>
      </form>

      <div className="space-y-2">
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          <SortableContext
            items={tasks.map(t => t.id)}
            strategy={verticalListSortingStrategy}
          >
            {tasks.map((task) => (
              <SortableTaskItem
                key={task.id}
                task={task}
                onToggle={toggleTask}
                onDelete={deleteTask}
              />
            ))}
          </SortableContext>
        </DndContext>

        {tasks.length === 0 && (
          <div className="text-center py-12 border border-dashed border-border-strong text-text-muted font-mono text-sm uppercase tracking-widest">
            NO_TASKS_FOUND
          </div>
        )}
      </div>
    </motion.div>
  );
};
