
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronDown, Circle, Hash } from 'lucide-react';
import { NavigationItem } from '../types';

interface SidebarProps {
  items: NavigationItem[]; // Kept as prop for flexibility, though usually comes from lib
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

  // Auto-expand checks
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
            if (hasChildren) {
              setIsExpanded(!isExpanded);
            } else {
              onNavigate(item.id);
            }
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`
          w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-200 group relative z-10
          ${isActive ? 'text-indigo-300' : 'text-slate-400 hover:text-slate-200'}
        `}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
      >
        {/* Hover Effect */}
        {isHovered && !isActive && (
          <motion.div 
            layoutId="sidebar-hover"
            className="absolute inset-0 bg-slate-800/50 rounded-lg -z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
        )}

        {/* Active Effect */}
        {isActive && (
          <motion.div 
            layoutId="sidebar-active"
            className="absolute inset-0 bg-indigo-500/10 border border-indigo-500/20 rounded-lg -z-10 shadow-[0_0_15px_rgba(99,102,241,0.1)]"
          />
        )}

        {/* Icon */}
        <span className={`${isActive ? "text-indigo-400" : "text-slate-600 group-hover:text-slate-400"}`}>
            {item.icon || (
              depth > 1 ? <div className={`w-1 h-1 rounded-full ${isActive ? "bg-indigo-400" : "bg-slate-600"}`} /> : <Circle size={4} className={isActive ? "fill-indigo-400" : "fill-slate-600"} />
            )}
        </span>
        
        <span className="flex-1 text-left truncate tracking-tight">{item.label}</span>
        
        {hasChildren && (
            <span className="text-slate-600 opacity-70">
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
            className="overflow-hidden relative"
          >
            {/* Thread Line for deep nesting */}
            {depth > 0 && (
              <div 
                className="absolute left-0 top-0 bottom-0 w-px bg-slate-800" 
                style={{ left: `${depth * 16 + 15}px` }}
              />
            )}
            
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
  return (
    <nav className="space-y-6 pb-24 px-2">
      <div className="space-y-1">
        {items.map(item => (
          <div key={item.id} className="mb-2">
            <SidebarItem 
              item={item} 
              depth={0} 
              currentPath={currentPath} 
              onNavigate={onNavigate} 
            />
          </div>
        ))}
      </div>
    </nav>
  );
};
