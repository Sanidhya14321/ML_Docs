
import React from 'react';
import { motion } from 'framer-motion';
import { NavigationItem, ViewSection } from '../types';
import { FileText, Folder } from 'lucide-react';
import { CodeBlock } from '../components/CodeBlock';

// In a real app, this would be imported from a central config
const BASE_URL = 'https://ai-codex.dev';

interface SitemapViewProps {
  navItems: NavigationItem[];
  onNavigate: (path: string) => void;
}

const generateSitemapXML = (items: NavigationItem[]): string => {
  const urls: string[] = [];
  
  const traverse = (nodes: NavigationItem[]) => {
    nodes.forEach(node => {
      urls.push(`  <url>
    <loc>${BASE_URL}/#/${node.id}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>${node.items ? '0.8' : '1.0'}</priority>
  </url>`);
      
      if (node.items) traverse(node.items);
    });
  };

  traverse(items);

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.join('\n')}
</urlset>`;
};

const robotsTxt = `User-agent: *
Allow: /
Sitemap: ${BASE_URL}/sitemap.xml`;

export const SitemapView: React.FC<SitemapViewProps> = ({ navItems, onNavigate }) => {
  const xmlString = generateSitemapXML(navItems);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-12 pb-20"
    >
      <header className="border-b border-slate-800 pb-12">
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-serif font-bold text-white mb-6"
        >
          Site Architecture
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-slate-400 text-xl max-w-3xl leading-relaxed font-light"
        >
          A complete index of the Neural Codex documentation. This page serves both as a user directory and a DevOps verification tool for crawler accessibility.
        </motion.p>
      </header>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.1
            }
          }
        }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-12"
      >
        {/* Visual Directory */}
        <section>
          <div className="flex items-center gap-3 mb-6">
             <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
                <Folder size={16} />
             </div>
             <h2 className="text-xl font-bold text-white">Directory Tree</h2>
          </div>
          
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 space-y-2">
            {navItems.map(item => (
               <div key={item.id} className="space-y-2">
                 <button onClick={() => onNavigate(item.id)} className="flex items-center gap-3 text-slate-300 hover:text-indigo-400 transition-colors group">
                    <span className="text-slate-600 group-hover:text-indigo-500/50 font-mono text-xs">00</span>
                    <span className="font-bold">{item.label}</span>
                 </button>
                 {item.items && (
                    <div className="ml-4 pl-4 border-l border-slate-800 space-y-2">
                       {item.items.map((sub, idx) => (
                          <button key={sub.id} onClick={() => onNavigate(sub.id)} className="flex items-center gap-3 text-sm text-slate-400 hover:text-indigo-300 transition-colors w-full text-left py-1">
                             <FileText size={12} className="text-slate-600" />
                             <span>{sub.label}</span>
                          </button>
                       ))}
                    </div>
                 )}
               </div>
            ))}
          </div>
        </section>

        {/* DevOps Artifacts */}
        <section className="space-y-8">
           <div className="flex items-center gap-3 mb-6">
             <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
                <FileText size={16} />
             </div>
             <h2 className="text-xl font-bold text-white">DevOps Artifacts</h2>
          </div>

          <div className="space-y-4">
             <div className="flex justify-between items-end">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">sitemap.xml</label>
             </div>
             <CodeBlock code={xmlString} language="xml" filename="public/sitemap.xml" />
          </div>

          <div className="space-y-4">
             <div className="flex justify-between items-end">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">robots.txt</label>
             </div>
             {/* Switched to bash for safer highlighting support */}
             <CodeBlock code={robotsTxt} language="bash" filename="public/robots.txt" />
          </div>
        </section>
      </motion.div>
    </motion.div>
  );
};
