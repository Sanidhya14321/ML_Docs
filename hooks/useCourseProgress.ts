
import { useContext } from 'react';
import { CourseContext } from '../contexts/CourseContext';

export const useCourseProgress = () => {
  const context = useContext(CourseContext);
  if (context === undefined) {
    throw new Error('useCourseProgress must be used within a CourseProvider');
  }
  return context;
};
