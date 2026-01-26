
import React from 'react';
import { motion } from 'framer-motion';
import { Award, CheckCircle, Printer, BrainCircuit } from 'lucide-react';
import { CURRICULUM } from '../data/curriculum';

export const CertificateView: React.FC = () => {
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center animate-fade-in relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-500/10 via-slate-950 to-slate-950 -z-10"></div>
      
      <div className="mb-8 flex gap-4 print:hidden">
        <button 
          onClick={handlePrint}
          className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold shadow-lg shadow-indigo-600/20 transition-all"
        >
          <Printer size={16} /> Print / Save PDF
        </button>
      </div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-[800px] bg-white text-slate-900 p-12 rounded-lg shadow-2xl border-[16px] border-double border-slate-200 relative overflow-hidden"
        id="certificate-frame"
      >
        {/* Certificate Border Pattern */}
        <div className="absolute inset-0 border-[2px] border-slate-900/5 m-4 pointer-events-none"></div>
        <div className="absolute inset-0 border-[1px] border-slate-900/10 m-6 pointer-events-none"></div>

        {/* Watermark */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-[0.03] pointer-events-none">
           <BrainCircuit size={400} />
        </div>

        <div className="text-center relative z-10 space-y-8">
           <div className="flex justify-center mb-6">
              <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center text-white">
                 <Award size={48} />
              </div>
           </div>

           <div className="space-y-2">
              <h1 className="text-5xl font-serif font-bold text-slate-900 tracking-tight uppercase">Certificate</h1>
              <p className="text-xl font-light text-slate-500 uppercase tracking-[0.4em]">Of Completion</p>
           </div>

           <div className="py-8">
              <p className="text-slate-500 italic mb-4">This certifies that</p>
              <div className="text-4xl font-serif font-bold text-indigo-900 border-b-2 border-indigo-900/20 pb-4 inline-block min-w-[300px]">
                 AI Engineer
              </div>
              <p className="text-slate-500 italic mt-6 mb-2">has successfully completed the comprehensive curriculum</p>
              <h2 className="text-2xl font-bold text-slate-800">{CURRICULUM.title}</h2>
           </div>

           <div className="grid grid-cols-2 gap-12 mt-12 pt-8 border-t border-slate-100">
              <div className="text-center">
                 <div className="text-lg font-bold text-slate-900 font-serif">
                    {new Date().toLocaleDateString()}
                 </div>
                 <div className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Date</div>
              </div>
              <div className="text-center">
                 <div className="text-lg font-bold text-slate-900 font-serif font-signature">
                    The Neural Codex Team
                 </div>
                 <div className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Instructor</div>
              </div>
           </div>

           <div className="mt-8 flex justify-center">
              <div className="bg-slate-50 px-4 py-2 rounded border border-slate-100 flex items-center gap-2">
                 <CheckCircle size={14} className="text-emerald-500" />
                 <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Verified Credential</span>
              </div>
           </div>
        </div>
      </motion.div>
    </div>
  );
};
