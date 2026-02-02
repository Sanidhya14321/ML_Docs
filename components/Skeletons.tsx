
import React from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit } from 'lucide-react';

interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className }) => (
  <div className={`bg-slate-200 dark:bg-slate-800 animate-pulse rounded ${className}`} />
);

export const DashboardSkeleton: React.FC = () => (
  <div className="pb-20 space-y-12">
    {/* Advanced Hero Skeleton */}
    <div className="relative rounded-3xl bg-slate-900/50 border border-slate-800 p-8 md:p-12 h-[320px] overflow-hidden">
        {/* Shimmer Overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-indigo-500/5 to-transparent -translate-x-full animate-[shimmer_2s_infinite] z-0 pointer-events-none" />
        
        {/* Background Decor */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/3"></div>

        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start gap-8 h-full">
            <div className="space-y-8 w-full max-w-lg">
                <div className="flex items-center gap-3 opacity-50">
                    <div className="w-5 h-5 rounded bg-indigo-500/20 animate-pulse" />
                    <div className="w-32 h-3 rounded bg-slate-700/50 animate-pulse" />
                </div>
                <div className="space-y-4">
                    <div className="w-3/4 h-12 rounded-lg bg-slate-800/80 animate-pulse" />
                    <div className="w-1/2 h-12 rounded-lg bg-slate-800/50 animate-pulse" />
                </div>
                <div className="space-y-2 pt-6">
                    <div className="w-full h-3 rounded bg-slate-800/30 animate-pulse" />
                    <div className="w-5/6 h-3 rounded bg-slate-800/30 animate-pulse" />
                </div>
            </div>
            
            {/* Right Side Stats Placeholder */}
            <div className="w-full md:w-80 h-full bg-slate-950/80 rounded-2xl border border-slate-800/50 p-6 flex flex-col justify-between backdrop-blur-sm shadow-xl">
                <div className="flex justify-between items-end">
                    <div className="w-24 h-3 bg-slate-800 rounded animate-pulse" />
                    <div className="w-12 h-8 bg-slate-800 rounded animate-pulse" />
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden relative">
                     <div className="absolute left-0 top-0 bottom-0 bg-indigo-500/20 w-1/3 animate-[pulse_2s_cubic-bezier(0.4,0,0.6,1)_infinite]" />
                </div>
                <div className="w-full h-12 bg-slate-800/80 rounded-lg animate-pulse" />
            </div>
        </div>

        {/* Technical Loading Indicator */}
        <div className="absolute bottom-6 left-12 font-mono text-[10px] text-indigo-400/60 uppercase tracking-widest flex items-center gap-3">
            <div className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </div>
            <span>Synchronizing Neural Lattice...</span>
        </div>
    </div>

    {/* Staggered Grid Skeleton */}
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {[1, 2, 3, 4].map((i) => (
            <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                className="bg-slate-900/30 border border-slate-800 rounded-2xl p-6 h-[280px] flex flex-col gap-6 relative overflow-hidden group"
            >
                {/* Individual Card Shimmer */}
                <div 
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-slate-700/5 to-transparent -translate-x-full animate-[shimmer_2.5s_infinite]" 
                    style={{ animationDelay: `${i * 0.15}s` }} 
                />
                
                <div className="flex justify-between items-center pb-6 border-b border-slate-800/50">
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl bg-slate-800/80 flex items-center justify-center">
                            <BrainCircuit className="text-slate-700 w-5 h-5 animate-pulse" />
                        </div>
                        <div className="space-y-2">
                            <div className="w-40 h-5 bg-slate-800 rounded animate-pulse" />
                            <div className="w-24 h-3 bg-slate-800/50 rounded animate-pulse" />
                        </div>
                    </div>
                </div>
                <div className="space-y-3 flex-1 pt-2">
                    {[1, 2, 3].map(j => (
                        <div key={j} className="flex items-center gap-3 px-2 py-1">
                            <div className="w-4 h-4 rounded-full bg-slate-800/60 animate-pulse" />
                            <div className="w-full h-8 bg-slate-800/20 rounded-lg animate-pulse" />
                        </div>
                    ))}
                </div>
            </motion.div>
        ))}
    </div>
  </div>
);

export const AlgorithmSkeleton: React.FC = () => (
  <div className="bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-3xl overflow-hidden mb-16 shadow-xl">
      <div className="p-8 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center">
          <div className="flex items-center gap-4">
              <Skeleton className="w-12 h-12 rounded-2xl" />
              <div className="space-y-2">
                  <Skeleton className="w-48 h-8" />
                  <Skeleton className="w-24 h-5 rounded-full" />
              </div>
          </div>
          <Skeleton className="w-32 h-10 rounded-xl" />
      </div>
      
      <div className="p-8 md:p-10 space-y-10">
          <div className="space-y-3">
              <Skeleton className="w-48 h-4 mb-2" />
              <Skeleton className="w-full h-4" />
              <Skeleton className="w-full h-4" />
              <Skeleton className="w-5/6 h-4" />
          </div>

          <Skeleton className="w-full h-72 rounded-3xl" />

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
              <div className="lg:col-span-5 space-y-8">
                  <Skeleton className="w-full h-32 rounded-3xl" />
                  <div className="space-y-4">
                      <Skeleton className="w-full h-24 rounded-2xl" />
                      <Skeleton className="w-full h-24 rounded-2xl" />
                  </div>
              </div>
              <div className="lg:col-span-7 space-y-4">
                  <Skeleton className="w-40 h-4" />
                  <Skeleton className="w-full h-48 rounded-xl" />
              </div>
          </div>
      </div>
  </div>
);

export const DocSkeleton: React.FC = () => (
    <div className="max-w-4xl mx-auto pb-24 space-y-12 animate-pulse">
        <div className="space-y-8 pb-8 border-b border-slate-800">
            <div className="flex gap-3">
                <Skeleton className="w-20 h-6 rounded" />
                <Skeleton className="w-24 h-6 rounded" />
            </div>
            <Skeleton className="w-3/4 h-14" />
            <div className="flex gap-6">
                <Skeleton className="w-24 h-4" />
                <Skeleton className="w-24 h-4" />
                <Skeleton className="w-32 h-6 rounded-full" />
            </div>
            <div className="space-y-3 pt-4">
                <Skeleton className="w-full h-5" />
                <Skeleton className="w-full h-5" />
                <Skeleton className="w-5/6 h-5" />
            </div>
        </div>
        
        <div className="space-y-8">
             <div className="space-y-4">
                <Skeleton className="w-1/3 h-8" />
                <Skeleton className="w-full h-4" />
                <Skeleton className="w-full h-4" />
                <Skeleton className="w-full h-4" />
             </div>
             
             <Skeleton className="w-full h-32 rounded-xl" />
             
             <div className="space-y-4">
                <Skeleton className="w-1/4 h-8" />
                <Skeleton className="w-full h-48 rounded-xl" />
             </div>
        </div>
    </div>
);
