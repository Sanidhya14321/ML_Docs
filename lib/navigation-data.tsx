
import { CURRICULUM } from '../data/curriculum';
import { NavigationItem } from '../types';

// Dynamically map the Master Database (CURRICULUM) into the Navigation Structure.
// This ensures Sidebar, Sitemap, and Breadcrumbs are always 100% in sync with the data.
export const NAV_ITEMS: NavigationItem[] = CURRICULUM.modules.map((module) => ({
  id: module.id,
  label: module.title,
  icon: module.icon,
  category: 'Module',
  items: module.chapters.map((chapter) => ({
    id: chapter.id,
    label: chapter.title,
    items: chapter.topics.map((topic) => ({
      id: topic.id,
      label: topic.title
    }))
  }))
}));
