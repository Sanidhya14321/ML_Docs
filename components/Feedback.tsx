
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, MessageSquare, CheckCircle } from 'lucide-react';

export const Feedback: React.FC = () => {
  const [status, setStatus] = useState<'idle' | 'helpful' | 'unhelpful'>('idle');
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleFeedback = (type: 'helpful' | 'unhelpful') => {
    setStatus(type);
    setIsSubmitted(true);
    
    // Simulate Analytics Log
    console.log(`[ANALYTICS] User feedback recorded: ${type} on ${window.location.hash}`);
    
    // Reset after delay
    setTimeout(() => setIsSubmitted(false), 3000);
  };

  return (
    <div className="mt-16 pt-8 border-t border-slate-800">
      <div className="flex flex-col md:flex-row items-center justify-between gap-6 p-6 rounded-2xl bg-slate-900/50 border border-slate-800">
        <div>
          <h3 className="text-sm font-bold text-white mb-1">Was this documentation helpful?</h3>
          <p className="text-xs text-slate-500">Your feedback helps us improve the learning engine.</p>
        </div>

        <div className="flex items-center gap-3">
          <AnimatePresence mode="wait">
            {isSubmitted ? (
              <motion.div 
                key="thank-you"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-2 text-emerald-400 text-sm font-bold bg-emerald-500/10 px-4 py-2 rounded-lg border border-emerald-500/20"
              >
                <CheckCircle size={16} />
                <span>Feedback Recorded</span>
              </motion.div>
            ) : (
              <motion.div 
                key="buttons"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex gap-3"
              >
                <button 
                  onClick={() => handleFeedback('helpful')}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-400 text-xs font-bold transition-all hover:scale-105 active:scale-95"
                >
                  <ThumbsUp size={14} /> Yes
                </button>
                <button 
                  onClick={() => handleFeedback('unhelpful')}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-rose-600 hover:text-white text-slate-400 text-xs font-bold transition-all hover:scale-105 active:scale-95"
                >
                  <ThumbsDown size={14} /> No
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      
      <div className="mt-4 flex justify-center">
         <a href="https://github.com/ai-codex/docs/issues" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-[10px] text-slate-600 hover:text-indigo-400 transition-colors">
            <MessageSquare size={12} /> Report an issue or suggest an edit
         </a>
      </div>
    </div>
  );
};
