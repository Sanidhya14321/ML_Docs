
import React from 'react';
import { motion } from 'framer-motion';
import { Award, CheckCircle, Printer, BrainCircuit } from 'lucide-react';
import { CURRICULUM } from '../data/curriculum';
import { Button } from '../components/Button';

export const CertificateView: React.FC = () => {
  const handlePrint = () => {
    window.print();
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="min-h-screen p-8 flex flex-col items-center justify-center relative overflow-hidden bg-app"
    >
      {/* Background Decor */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none z-0" 
           style={{ backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)', backgroundSize: '24px 24px' }} />
      
      <div className="mb-12 flex gap-4 print:hidden relative z-10">
        <Button 
          onClick={handlePrint}
          variant="primary"
          size="lg"
          leftIcon={<Printer size={18} />}
        >
          EXPORT_PDF
        </Button>
      </div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-[900px] bg-white text-slate-950 p-16 rounded-none shadow-2xl border border-slate-950 relative overflow-hidden"
        id="certificate-frame"
      >
        {/* Technical Border Pattern */}
        <div className="absolute top-0 left-0 w-full h-1 bg-slate-950" />
        <div className="absolute bottom-0 left-0 w-full h-1 bg-slate-950" />
        <div className="absolute top-0 left-0 h-full w-1 bg-slate-950" />
        <div className="absolute top-0 right-0 h-full w-1 bg-slate-950" />
        
        <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-slate-950" />
        <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-slate-950" />
        <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-slate-950" />
        <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-slate-950" />

        {/* Watermark */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-[0.02] pointer-events-none">
           <BrainCircuit size={500} />
        </div>

        <div className="text-center relative z-10 space-y-12">
           <div className="flex justify-center mb-8">
              <div className="w-24 h-24 bg-slate-950 rounded-none flex items-center justify-center text-white">
                 <Award size={56} />
              </div>
           </div>

           <div className="space-y-4">
              <h1 className="text-6xl font-heading font-black text-slate-950 tracking-tighter uppercase leading-none">CERTIFICATE</h1>
              <div className="flex items-center justify-center gap-4">
                 <div className="h-px w-12 bg-slate-950/20" />
                 <p className="text-sm font-mono font-black text-slate-500 uppercase tracking-[0.5em]">OF_COMPLETION</p>
                 <div className="h-px w-12 bg-slate-950/20" />
              </div>
           </div>

           <div className="py-12 space-y-8">
              <p className="text-slate-500 font-mono text-xs uppercase tracking-widest">THIS_CERTIFIES_THAT_NODE_ID:</p>
              <div className="text-5xl font-heading font-black text-slate-950 border-b-4 border-slate-950 pb-6 inline-block min-w-[400px] uppercase tracking-tight">
                 AI_ENGINEER_01
              </div>
              <p className="text-slate-500 font-mono text-xs uppercase tracking-widest leading-relaxed">
                HAS_SUCCESSFULLY_SYNCHRONIZED_WITH_THE_NEURAL_ARCHITECTURE_OF
              </p>
              <h2 className="text-3xl font-heading font-black text-slate-950 uppercase tracking-tight">{CURRICULUM.title}</h2>
           </div>

           <div className="grid grid-cols-2 gap-20 mt-16 pt-12 border-t border-slate-100">
              <div className="text-center space-y-2">
                 <div className="text-xl font-mono font-black text-slate-950 uppercase tracking-widest">
                    {new Date().toLocaleDateString()}
                 </div>
                 <div className="text-[9px] font-mono font-black text-slate-400 uppercase tracking-[0.3em]">TIMESTAMP</div>
              </div>
              <div className="text-center space-y-2">
                 <div className="text-xl font-mono font-black text-slate-950 uppercase tracking-widest">
                    AI_CODEX_CORE
                 </div>
                 <div className="text-[9px] font-mono font-black text-slate-400 uppercase tracking-[0.3em]">AUTHORITY</div>
              </div>
           </div>

           <div className="mt-12 flex justify-center">
              <div className="bg-emerald-50 px-6 py-3 rounded-none border border-emerald-100 flex items-center gap-3">
                 <CheckCircle size={16} className="text-emerald-500" />
                 <span className="text-[10px] font-mono font-black text-emerald-600 uppercase tracking-[0.3em]">VERIFIED_CREDENTIAL_HASH: {Math.random().toString(36).substring(2, 15).toUpperCase()}</span>
              </div>
           </div>
        </div>
      </motion.div>
    </motion.div>
  );
};
