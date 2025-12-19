
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
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-[150]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg z-[160] px-4"
          >
            <div className="bg-[#020617] border border-slate-800 rounded-2xl shadow-2xl overflow-hidden relative">
              <button onClick={onClose} className="absolute top-4 right-4 p-2 text-slate-500 hover:text-white z-10">
                <X size={20} />
              </button>

              <div className="p-6 border-b border-slate-800">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Share2 size={18} className="text-indigo-400" /> Share this article
                </h3>
              </div>

              {/* OG Card Preview (Simulating @vercel/og) */}
              <div className="p-6 bg-slate-900/50 flex flex-col items-center">
                 <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 w-full text-left">Social Card Preview</div>
                 <div className="w-full aspect-[1.91/1] bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 rounded-xl overflow-hidden relative flex flex-col justify-between p-8 shadow-2xl select-none">
                    {/* Background Pattern */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
                    <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl"></div>

                    <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
                                <BrainCircuit size={18} className="text-white" />
                            </div>
                            <span className="font-serif font-bold text-white tracking-tight">AI<span className="text-indigo-500">.</span>Codex</span>
                        </div>
                        <h1 className="text-3xl font-serif font-bold text-white leading-tight mb-2">
                            {metadata.title}
                        </h1>
                        <p className="text-slate-400 text-sm line-clamp-2">
                            {metadata.description}
                        </p>
                    </div>

                    <div className="relative z-10 flex items-center gap-4 mt-4">
                        <div className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-[10px] font-mono text-indigo-400 uppercase tracking-wider">
                            {metadata.difficulty}
                        </div>
                        <div className="text-[10px] text-slate-500 font-mono">
                            {metadata.readTimeMinutes} min read
                        </div>
                    </div>
                 </div>
              </div>

              <div className="p-6 space-y-4">
                 <div className="flex gap-2">
                    <div className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-400 font-mono truncate">
                        {window.location.href}
                    </div>
                    <button 
                        onClick={handleCopy}
                        className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-all"
                    >
                        {copied ? <Check size={14} /> : <Copy size={14} />}
                        {copied ? 'Copied' : 'Copy'}
                    </button>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-3">
                    <a 
                        href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(metadata.title)}&url=${encodeURIComponent(window.location.href)}`} 
                        target="_blank" rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white py-2.5 rounded-xl transition-all text-xs font-bold"
                    >
                        <Twitter size={14} /> Twitter
                    </a>
                    <a 
                        href={`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(window.location.href)}`} 
                        target="_blank" rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white py-2.5 rounded-xl transition-all text-xs font-bold"
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
