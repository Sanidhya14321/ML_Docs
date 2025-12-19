
import { CURRICULUM } from '../data/curriculum';
import { Topic, Chapter, Module } from '../types';

// Flatten the curriculum into a linear list of topics for easy navigation
const getAllTopics = (): Topic[] => {
  const topics: Topic[] = [];
  CURRICULUM.forEach(module => {
    module.chapters.forEach(chapter => {
      chapter.topics.forEach(topic => {
        topics.push(topic);
      });
    });
  });
  return topics;
};

export const getTopicById = (id: string): Topic | undefined => {
  // Direct lookup
  for (const module of CURRICULUM) {
    for (const chapter of module.chapters) {
      const topic = chapter.topics.find(t => t.id === id);
      if (topic) return topic;
    }
  }
  return undefined;
};

export const getNextTopic = (currentId: string): string | null => {
  const allTopics = getAllTopics();
  const index = allTopics.findIndex(t => t.id === currentId);
  if (index !== -1 && index < allTopics.length - 1) {
    return allTopics[index + 1].id;
  }
  return null;
};

export const getPrevTopic = (currentId: string): string | null => {
  const allTopics = getAllTopics();
  const index = allTopics.findIndex(t => t.id === currentId);
  if (index > 0) {
    return allTopics[index - 1].id;
  }
  return null;
};

export const getBreadcrumbs = (topicId: string): { label: string; id?: string }[] => {
  for (const module of CURRICULUM) {
    for (const chapter of module.chapters) {
      const topic = chapter.topics.find(t => t.id === topicId);
      if (topic) {
        return [
          { label: module.title },
          { label: chapter.title },
          { label: topic.title, id: topic.id }
        ];
      }
    }
  }
  return [];
};
