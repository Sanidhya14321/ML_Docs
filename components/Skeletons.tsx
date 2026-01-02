
import React from 'react';

interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className }) => (
  <div className={`bg-slate-200 dark:bg-slate-800 animate-pulse rounded ${className}`} />
);

export const DashboardSkeleton: React.FC = () => (
  <div className="pb-20 space-y-12">
    {/* Hero Skeleton */}
    <div className="rounded-3xl bg-slate-900 border border-slate-800 p-8 md:p-12 h-[300px] relative overflow-hidden">
        <div className="flex flex-col md:flex-row justify-between items-start gap-8 h-full">
            <div className="space-y-6 w-full max-w-lg">
                <Skeleton className="w-32 h-4" />
                <Skeleton className="w-3/4 h-12" />
                <div className="space-y-2">
                    <Skeleton className="w-full h-4" />
                    <Skeleton className="w-2/3 h-4" />
                </div>
            </div>
            <div className="w-full md:w-80 h-full">
                <Skeleton className="w-full h-full rounded-2xl" />
            </div>
        </div>
    </div>

    {/* Grid Skeleton */}
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 h-[400px] flex flex-col gap-6">
                <div className="flex justify-between items-center pb-6 border-b border-slate-800/50">
                    <div className="flex items-center gap-4">
                        <Skeleton className="w-12 h-12 rounded-xl" />
                        <div className="space-y-2">
                            <Skeleton className="w-40 h-6" />
                            <Skeleton className="w-24 h-3" />
                        </div>
                    </div>
                </div>
                <div className="space-y-4 flex-1">
                    <Skeleton className="w-32 h-4 mb-2" />
                    {[1, 2, 3].map(j => (
                        <Skeleton key={j} className="w-full h-10 rounded-lg" />
                    ))}
                </div>
            </div>
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
