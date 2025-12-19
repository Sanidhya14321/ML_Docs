
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
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    titleColor: 'text-blue-300'
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    titleColor: 'text-amber-300'
  },
  tip: {
    icon: Lightbulb,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    titleColor: 'text-emerald-300'
  },
  danger: {
    icon: Flame,
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
    titleColor: 'text-rose-300'
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
      className={`relative my-8 p-5 rounded-2xl border ${style.bg} ${style.border} overflow-hidden group`}
    >
      <div className={`absolute top-0 left-0 w-1 h-full ${style.bg.replace('/10', '/50')}`} />
      
      <div className="flex gap-4">
        <div className={`mt-0.5 p-2 rounded-lg bg-slate-950/50 ${style.color} shadow-sm h-fit`}>
          <Icon size={18} />
        </div>
        <div className="flex-1">
          {title && (
            <h4 className={`text-sm font-bold uppercase tracking-wide mb-2 ${style.titleColor}`}>
              {title}
            </h4>
          )}
          <div className="text-sm text-slate-300 leading-relaxed">
            {children}
          </div>
        </div>
      </div>

      {/* Glossy Effect on Hover */}
      <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/5 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
    </motion.div>
  );
};
