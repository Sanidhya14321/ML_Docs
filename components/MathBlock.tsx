import React from 'react';

interface MathBlockProps {
  children: React.ReactNode;
  label?: string;
}

export const MathBlock: React.FC<MathBlockProps> = ({ children, label }) => {
  return (
    <div className="my-6 p-6 bg-slate-900 border-l-4 border-indigo-500 rounded-r-lg shadow-md">
      {label && <div className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2">{label}</div>}
      <div className="math-serif text-xl text-slate-200 leading-relaxed tracking-wide text-center">
        {children}
      </div>
    </div>
  );
};