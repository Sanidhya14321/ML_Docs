
import React, { createContext, useState, useEffect, useCallback } from 'react';
import { CURRICULUM } from '../data/curriculum';

const STORAGE_KEY_PROGRESS = 'ai-codex-progress';
const STORAGE_KEY_ACTIVE = 'ai-codex-last-active';

interface CourseContextType {
  completedTopics: string[];
  lastActiveTopic: string;
  setLastActiveTopic: (id: string) => void;
  markAsCompleted: (topicId: string) => void;
  isCompleted: (topicId: string) => boolean;
  getModuleProgress: (moduleId: string) => number;
  getOverallProgress: () => number;
  resetProgress: () => void;
}

export const CourseContext = createContext<CourseContextType | undefined>(undefined);

export const CourseProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [completedTopics, setCompletedTopics] = useState<string[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem(STORAGE_KEY_PROGRESS);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error("Failed to parse progress", e);
      return [];
    }
  });
  
  const [lastActiveTopic, setLastActiveTopic] = useState<string>(() => {
    if (typeof window === 'undefined') return '';
    return localStorage.getItem(STORAGE_KEY_ACTIVE) || '';
  });

  // Persist changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_PROGRESS, JSON.stringify(completedTopics));
  }, [completedTopics]);

  useEffect(() => {
    if (lastActiveTopic) {
      localStorage.setItem(STORAGE_KEY_ACTIVE, lastActiveTopic);
    }
  }, [lastActiveTopic]);

  const markAsCompleted = useCallback((topicId: string) => {
    setCompletedTopics(prev => {
      if (prev.includes(topicId)) return prev;
      return [...prev, topicId];
    });
  }, []);

  const isCompleted = useCallback((topicId: string) => {
    return completedTopics.includes(topicId);
  }, [completedTopics]);

  const getModuleProgress = useCallback((moduleId: string) => {
    const module = CURRICULUM.modules.find(m => m.id === moduleId);
    if (!module) return 0;
    
    let total = 0;
    let completed = 0;
    
    module.chapters.forEach(c => {
      c.topics.forEach(t => {
        total++;
        if (completedTopics.includes(t.id)) completed++;
      });
    });
    
    return total === 0 ? 0 : Math.round((completed / total) * 100);
  }, [completedTopics]);
  
  const getOverallProgress = useCallback(() => {
     let total = 0;
     let completed = 0;
     CURRICULUM.modules.forEach(m => m.chapters.forEach(c => c.topics.forEach(t => {
         total++;
         if (completedTopics.includes(t.id)) completed++;
     })));
     return total === 0 ? 0 : Math.round((completed / total) * 100);
  }, [completedTopics]);

  const resetProgress = useCallback(() => {
    setCompletedTopics([]);
    setLastActiveTopic('');
    localStorage.removeItem(STORAGE_KEY_PROGRESS);
    localStorage.removeItem(STORAGE_KEY_ACTIVE);
    // Clear all lab persistence as well
    Object.keys(localStorage).forEach(key => {
        if (key.startsWith('ai-codex-lab-')) {
            localStorage.removeItem(key);
        }
    });
    window.location.reload();
  }, []);

  return (
    <CourseContext.Provider value={{ 
        completedTopics, 
        lastActiveTopic, 
        setLastActiveTopic, 
        markAsCompleted, 
        isCompleted, 
        getModuleProgress, 
        getOverallProgress, 
        resetProgress 
    }}>
      {children}
    </CourseContext.Provider>
  );
};
