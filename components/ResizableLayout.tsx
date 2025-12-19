
import React, { useState, useEffect, useRef } from 'react';
import { GripVertical } from 'lucide-react';

interface ResizableLayoutProps {
  left: React.ReactNode;
  right: React.ReactNode;
  initialLeftWidth?: number; // Percentage
  isMobile?: boolean;
}

export const ResizableLayout: React.FC<ResizableLayoutProps> = ({ 
  left, 
  right, 
  initialLeftWidth = 40,
  isMobile = false
}) => {
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const startDragging = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;
      
      const containerRect = containerRef.current.getBoundingClientRect();
      const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
      
      // Constrain width between 20% and 80%
      if (newLeftWidth > 20 && newLeftWidth < 80) {
        setLeftWidth(newLeftWidth);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  // Mobile Layout: Vertical Stack
  if (isMobile) {
    return (
      <div className="flex flex-col h-full w-full overflow-hidden bg-[#020617]">
         <div className="h-1/2 overflow-y-auto border-b border-slate-800 relative z-10">
            {left}
         </div>
         <div className="h-1/2 overflow-hidden bg-[#1e1e1e] relative z-0">
            {right}
         </div>
      </div>
    );
  }

  // Desktop Layout: Resizable Horizontal Split
  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden relative">
      {/* Left Pane */}
      <div 
        style={{ width: `${leftWidth}%` }} 
        className="h-full overflow-y-auto custom-scrollbar relative"
      >
        {left}
        {/* Overlay to prevent iframe/selection interference during drag */}
        {isDragging && <div className="absolute inset-0 z-50 bg-transparent" />}
      </div>

      {/* Resizer Handle */}
      <div
        className={`w-4 -ml-2 h-full cursor-col-resize z-40 flex items-center justify-center group absolute`}
        style={{ left: `${leftWidth}%` }}
        onMouseDown={startDragging}
      >
        <div className={`w-[1px] h-full transition-colors ${isDragging ? 'bg-indigo-500' : 'bg-slate-800 group-hover:bg-indigo-500/50'}`} />
        <div className={`absolute p-1 rounded bg-slate-800 border border-slate-700 text-slate-500 transition-colors ${isDragging ? 'text-indigo-400 border-indigo-500' : 'group-hover:text-slate-300'}`}>
           <GripVertical size={12} />
        </div>
      </div>

      {/* Right Pane */}
      <div 
        style={{ width: `${100 - leftWidth}%` }} 
        className="h-full overflow-hidden bg-[#1e1e1e]"
      >
         {right}
         {/* Overlay to prevent interference during drag */}
         {isDragging && <div className="absolute inset-0 z-50 bg-transparent" />}
      </div>
    </div>
  );
};
