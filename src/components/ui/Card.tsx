import React from 'react';
import { cn } from '../../lib/utils';
import { motion, HTMLMotionProps } from 'framer-motion';

interface CardProps extends HTMLMotionProps<'div'> {
  variant?: 'default' | 'outline' | 'ghost' | 'glass';
  isHoverable?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', isHoverable, children, ...props }, ref) => {
    const variants = {
      default: 'bg-surface border-border-strong shadow-sm',
      outline: 'bg-transparent border-border-strong',
      ghost: 'bg-transparent border-transparent',
      glass: 'bg-surface/60 backdrop-blur-md border-border-strong shadow-lg',
    };

    return (
      <motion.div
        ref={ref}
        whileHover={isHoverable ? { y: -4, transition: { duration: 0.2 } } : undefined}
        className={cn(
          'rounded-none border transition-all duration-normal overflow-hidden',
          variants[variant],
          isHoverable && 'hover:shadow-md hover:border-brand/50',
          className
        )}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={cn('p-6 space-y-1.5', className)} {...props} />
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ className, ...props }) => (
  <h3 className={cn('text-xl font-heading font-black text-text-primary uppercase tracking-tight', className)} {...props} />
);

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({ className, ...props }) => (
  <p className={cn('text-sm text-text-muted font-mono', className)} {...props} />
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={cn('p-6 pt-0', className)} {...props} />
);

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={cn('flex items-center p-6 pt-0', className)} {...props} />
);

Card.displayName = 'Card';
