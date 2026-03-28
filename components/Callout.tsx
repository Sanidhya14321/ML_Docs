
import React from 'react';
import { motion } from 'framer-motion';
import { Info, AlertTriangle, Lightbulb, Flame } from 'lucide-react';

type CalloutType = 'note' | 'warning' | 'tip' | 'danger';

interface CalloutProps {
  type?: CalloutType;
  title?: string;
  children: React.ReactNode;
}

const variants = {
  note: {
    icon: Info,
    color: 'text-brand',
    bg: 'bg-brand/5',
    border: 'border-brand/20',
    titleColor: 'text-brand'
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-amber-500',
    bg: 'bg-amber-500/5',
    border: 'border-amber-500/20',
    titleColor: 'text-amber-500'
  },
  tip: {
    icon: Lightbulb,
    color: 'text-emerald-500',
    bg: 'bg-emerald-500/5',
    border: 'border-emerald-500/20',
    titleColor: 'text-emerald-500'
  },
  danger: {
    icon: Flame,
    color: 'text-rose-500',
    bg: 'bg-rose-500/5',
    border: 'border-rose-500/20',
    titleColor: 'text-rose-500'
  }
};

export const Callout: React.FC<CalloutProps> = ({ type = 'note', title, children }) => {
  const style = variants[type];
  const Icon = style.icon;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      whileHover={{ scale: 1.01 }}
      className={`relative my-8 p-6 rounded-none border ${style.bg} ${style.border} overflow-hidden group`}
    >
      <div className={`absolute top-0 left-0 w-1 h-full ${style.bg.replace('/5', '/50')}`} />
      
      <div className="flex gap-4">
        <div className={`mt-0.5 p-2 rounded-none bg-surface border border-border-strong ${style.color} shadow-sm h-fit`}>
          <Icon size={18} />
        </div>
        <div className="flex-1">
          {title && (
            <h4 className={`text-[10px] font-mono font-black uppercase tracking-widest mb-2 ${style.titleColor}`}>
              {title}
            </h4>
          )}
          <div className="text-sm text-text-secondary leading-relaxed font-light">
            {children}
          </div>
        </div>
      </div>

      {/* Glossy Effect on Hover */}
      <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/5 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
    </motion.div>
  );
};
