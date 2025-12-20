
import { useState, useEffect, useCallback } from 'react';
import { CURRICULUM } from '../data/curriculum';

const STORAGE_KEY_PROGRESS = 'ai-codex-progress';
const STORAGE_KEY_ACTIVE = 'ai-codex-last-active';

export const useCourseProgress = () => {
  const [completedTopics, setCompletedTopics] = useState<string[]>(() => {
    if (typeof window === 'undefined') return [];
    const saved = localStorage.getItem(STORAGE_KEY_PROGRESS);
    return saved ? JSON.parse(saved) : [];
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
    const module = CURRICULUM.find(m => m.id === moduleId);
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
     CURRICULUM.forEach(m => m.chapters.forEach(c => c.topics.forEach(t => {
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
    window.location.reload(); // Force reload to clear all states
  }, []);

  return { 
    completedTopics, 
    lastActiveTopic, 
    setLastActiveTopic, 
    markAsCompleted, 
    isCompleted, 
    getModuleProgress,
    getOverallProgress,
    resetProgress
  };
};
