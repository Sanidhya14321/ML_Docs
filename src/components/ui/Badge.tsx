import React from 'react';
import { cn } from '../../lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'success' | 'warning' | 'destructive';
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({ 
  className, 
  variant = 'primary', 
  size = 'md', 
  children, 
  ...props 
}) => {
  const variants = {
    primary: 'bg-brand/10 text-brand border-brand/20',
    secondary: 'bg-surface-active text-text-secondary border-border-strong',
    outline: 'bg-transparent border-border-strong text-text-secondary',
    success: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
    warning: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
    destructive: 'bg-rose-500/10 text-rose-500 border-rose-500/20',
  };

  const sizes = {
    sm: 'px-1.5 py-0.5 text-[9px] font-mono',
    md: 'px-2 py-0.5 text-[10px] font-mono',
  };

  return (
    <div
      className={cn(
        'inline-flex items-center font-black uppercase tracking-widest border rounded-none transition-colors',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
