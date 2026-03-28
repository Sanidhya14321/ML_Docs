
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, Check, Twitter, Linkedin, Share2, BrainCircuit } from 'lucide-react';
import { DocMetadata } from '../types';

interface ShareDialogProps {
  isOpen: boolean;
  onClose: () => void;
  metadata: DocMetadata;
  url: string;
}

export const ShareDialog: React.FC<ShareDialogProps> = ({ isOpen, onClose, metadata, url }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-app/80 backdrop-blur-sm z-[150]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg z-[160] px-4"
          >
            <div className="bg-surface border border-border-strong rounded-none shadow-2xl overflow-hidden relative">
              <button onClick={onClose} className="absolute top-4 right-4 p-2 text-text-muted hover:text-text-primary hover:bg-surface-hover z-10 transition-colors">
                <X size={20} />
              </button>

              <div className="p-6 border-b border-border-strong">
                <h3 className="text-lg font-heading font-black text-text-primary uppercase tracking-tight flex items-center gap-2">
                  <Share2 size={18} className="text-brand" /> Share this article
                </h3>
              </div>

              {/* OG Card Preview (Simulating @vercel/og) */}
              <div className="p-6 bg-surface-active flex flex-col items-center">
                 <div className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest mb-3 w-full text-left">Social Card Preview</div>
                 <div className="w-full aspect-[1.91/1] bg-app border border-border-strong rounded-none overflow-hidden relative flex flex-col justify-between p-8 shadow-2xl select-none">
                    {/* Background Pattern */}
                    <div className="absolute inset-0 opacity-[0.03] pointer-events-none z-0" style={{ backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)', backgroundSize: '24px 24px' }}></div>
                    <div className="absolute top-0 right-0 w-64 h-64 bg-brand/10 rounded-full blur-3xl"></div>

                    <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-8 h-8 rounded-none bg-text-primary flex items-center justify-center">
                                <BrainCircuit size={18} className="text-app" />
                            </div>
                            <span className="font-heading font-black text-text-primary uppercase tracking-tighter">AI_CODEX</span>
                        </div>
                        <h1 className="text-3xl font-heading font-black text-text-primary uppercase tracking-tight leading-tight mb-2">
                            {metadata.title}
                        </h1>
                        <p className="text-text-secondary text-sm line-clamp-2 font-mono">
                            {metadata.description}
                        </p>
                    </div>

                    <div className="relative z-10 flex items-center gap-4 mt-4">
                        <div className="px-3 py-1 rounded-none bg-surface border border-brand/30 text-[10px] font-mono font-black text-brand uppercase tracking-widest">
                            {metadata.difficulty}
                        </div>
                        <div className="text-[10px] text-text-muted font-mono uppercase tracking-widest">
                            {metadata.readTimeMinutes} min read
                        </div>
                    </div>
                 </div>
              </div>

              <div className="p-6 space-y-4">
                 <div className="flex gap-2">
                    <div className="flex-1 bg-app border border-border-strong rounded-none px-3 py-2 text-xs text-text-secondary font-mono truncate">
                        {window.location.href}
                    </div>
                    <button 
                        onClick={handleCopy}
                        className="bg-text-primary hover:bg-brand text-app px-4 py-2 rounded-none text-[10px] font-mono font-black uppercase tracking-widest flex items-center gap-2 transition-all"
                    >
                        {copied ? <Check size={14} /> : <Copy size={14} />}
                        {copied ? 'Copied' : 'Copy'}
                    </button>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-3">
                    <a 
                        href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(metadata.title)}&url=${encodeURIComponent(window.location.href)}`} 
                        target="_blank" rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 bg-surface-hover border border-border-strong hover:border-brand text-text-secondary hover:text-brand py-2.5 rounded-none transition-all text-[10px] font-mono font-black uppercase tracking-widest"
                    >
                        <Twitter size={14} /> Twitter
                    </a>
                    <a 
                        href={`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(window.location.href)}`} 
                        target="_blank" rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 bg-surface-hover border border-border-strong hover:border-brand text-text-secondary hover:text-brand py-2.5 rounded-none transition-all text-[10px] font-mono font-black uppercase tracking-widest"
                    >
                        <Linkedin size={14} /> LinkedIn
                    </a>
                 </div>
              </div>

            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
