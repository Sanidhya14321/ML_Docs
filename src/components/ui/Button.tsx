import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends HTMLMotionProps<'button'> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive' | 'success' | 'warning';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, disabled, leftIcon, rightIcon, children, ...props }, ref) => {
    
    const variants = {
      primary: 'bg-text-primary text-app hover:bg-brand shadow-sm',
      secondary: 'bg-surface-active text-text-primary hover:bg-zinc-200 dark:hover:bg-zinc-700',
      outline: 'bg-transparent border border-border-strong text-text-primary hover:bg-surface-active hover:border-brand',
      ghost: 'bg-transparent text-text-secondary hover:bg-surface-active hover:text-text-primary',
      destructive: 'bg-rose-500 text-white hover:bg-rose-600 shadow-sm',
      success: 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-sm',
      warning: 'bg-amber-500 text-white hover:bg-amber-600 shadow-sm',
    };

    const sizes = {
      xs: 'h-6 px-2 text-[9px] font-mono font-black uppercase tracking-widest gap-1 rounded-none',
      sm: 'h-8 px-3 text-[10px] font-mono font-black uppercase tracking-widest gap-1.5 rounded-none',
      md: 'h-10 px-4 text-xs font-mono font-black uppercase tracking-widest gap-2 rounded-none',
      lg: 'h-12 px-6 text-sm font-mono font-black uppercase tracking-widest gap-2.5 rounded-none',
      xl: 'h-14 px-8 text-base font-mono font-black uppercase tracking-widest gap-3 rounded-none',
    };

    const isDisabled = disabled || isLoading;

    return (
      <motion.button
        ref={ref}
        whileTap={!isDisabled ? { scale: 0.98 } : undefined}
        disabled={isDisabled}
        className={cn(
          'inline-flex items-center justify-center font-medium transition-all duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {isLoading && <Loader2 className="w-4 h-4 animate-spin shrink-0" />}
        {!isLoading && leftIcon && <span className="shrink-0">{leftIcon}</span>}
        <span className="truncate">{children as React.ReactNode}</span>
        {!isLoading && rightIcon && <span className="shrink-0">{rightIcon}</span>}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';
