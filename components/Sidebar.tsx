
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronDown, Circle } from 'lucide-react';
import { NavigationItem } from '../types';

interface SidebarProps {
  items: NavigationItem[];
  currentPath: string;
  onNavigate: (path: string) => void;
}

const SidebarItem: React.FC<{ 
  item: NavigationItem; 
  depth: number; 
  currentPath: string;
  onNavigate: (path: string) => void; 
}> = ({ item, depth, currentPath, onNavigate }) => {
  const isActive = currentPath === item.id;
  const hasChildren = item.items && item.items.length > 0;
  const [isHovered, setIsHovered] = useState(false);

  // Auto-expand if a child is active
  const isChildActive = (items: NavigationItem[] | undefined): boolean => {
      if (!items) return false;
      return items.some(child => child.id === currentPath || isChildActive(child.items));
  };

  const [isExpanded, setIsExpanded] = useState(isActive || isChildActive(item.items));
  
  useEffect(() => {
    if (isChildActive(item.items)) setIsExpanded(true);
  }, [currentPath]);

  return (
    <div className="select-none relative">
      <button
        onClick={() => {
            if (hasChildren) setIsExpanded(!isExpanded);
            onNavigate(item.id);
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`
          w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-200 group relative z-10
          ${isActive ? 'text-indigo-300' : 'text-slate-400 hover:text-slate-200'}
        `}
        style={{ paddingLeft: `${depth * 12 + 12}px` }}
      >
        {/* Magnetic Hover Effect */}
        {isHovered && !isActive && (
          <motion.div 
            layoutId="sidebar-hover"
            className="absolute inset-0 bg-slate-800/50 rounded-lg -z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
        )}

        {/* Active Background */}
        {isActive && (
          <motion.div 
            layoutId="sidebar-active"
            className="absolute inset-0 bg-indigo-500/10 border border-indigo-500/20 rounded-lg -z-10 shadow-[0_0_15px_rgba(99,102,241,0.1)]"
          />
        )}

        <span className={`${isActive ? "text-indigo-400" : "text-slate-600 group-hover:text-slate-400"}`}>
            {item.icon || <Circle size={4} className={isActive ? "fill-indigo-400" : "fill-slate-600"} />}
        </span>
        
        <span className="flex-1 text-left truncate">{item.label}</span>
        
        {hasChildren && (
            <span className="text-slate-600">
                {isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            </span>
        )}
      </button>

      <AnimatePresence>
        {hasChildren && isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            {item.items!.map(child => (
              <SidebarItem 
                key={child.id} 
                item={child} 
                depth={depth + 1} 
                currentPath={currentPath} 
                onNavigate={onNavigate} 
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({ items, currentPath, onNavigate }) => {
  const categories = ['Core', 'Supervised', 'Advanced', 'Lab'];

  return (
    <nav className="space-y-6 pb-12 px-2">
      {categories.map(category => {
        const categoryItems = items.filter(item => item.category === category);
        if (categoryItems.length === 0) return null;

        return (
          <div key={category}>
            <h3 className="px-4 text-[9px] font-black text-slate-600 uppercase tracking-[0.3em] mb-2">{category}</h3>
            <div className="space-y-0.5">
              {categoryItems.map(item => (
                <SidebarItem 
                  key={item.id} 
                  item={item} 
                  depth={0} 
                  currentPath={currentPath} 
                  onNavigate={onNavigate} 
                />
              ))}
            </div>
          </div>
        );
      })}
    </nav>
  );
};
