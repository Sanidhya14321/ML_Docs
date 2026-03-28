import React from 'react';
import { cn } from '../../lib/utils';
import { AlertCircle, CheckCircle2, Info, XCircle } from 'lucide-react';

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'info' | 'success' | 'warning' | 'destructive';
  title?: string;
  icon?: React.ReactNode;
}

export const Alert: React.FC<AlertProps> = ({ 
  className, 
  variant = 'info', 
  title, 
  icon, 
  children, 
  ...props 
}) => {
  const variants = {
    info: 'bg-brand/10 text-brand border-brand/20',
    success: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
    warning: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
    destructive: 'bg-rose-500/10 text-rose-500 border-rose-500/20',
  };

  const icons = {
    info: <Info size={18} />,
    success: <CheckCircle2 size={18} />,
    warning: <AlertCircle size={18} />,
    destructive: <XCircle size={18} />,
  };

  return (
    <div
      role="alert"
      className={cn(
        'flex gap-3 p-4 border rounded-none animate-fade-in',
        variants[variant],
        className
      )}
      {...props}
    >
      <div className="shrink-0 mt-0.5">
        {icon || icons[variant]}
      </div>
      <div className="space-y-1">
        {title && (
          <h5 className="font-heading font-black uppercase tracking-tight leading-none">
            {title}
          </h5>
        )}
        <div className="text-sm opacity-90 leading-relaxed font-mono">
          {children}
        </div>
      </div>
    </div>
  );
};
