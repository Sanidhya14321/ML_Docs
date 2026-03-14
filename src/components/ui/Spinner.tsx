import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'primary' | 'secondary' | 'white';
}

export const Spinner: React.FC<SpinnerProps> = ({ 
  className, 
  size = 'md', 
  variant = 'primary',
  ...props 
}) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  };

  const variants = {
    primary: 'text-brand',
    secondary: 'text-text-secondary',
    white: 'text-white',
  };

  return (
    <div
      role="status"
      className={cn('flex items-center justify-center', className)}
      {...props}
    >
      <Loader2 className={cn('animate-spin', sizes[size], variants[variant])} />
      <span className="sr-only">Loading...</span>
    </div>
  );
};
