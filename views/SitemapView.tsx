
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
      className="space-y-16 pb-24"
    >
      <header className="border-b border-border-strong pb-12">
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-heading font-black text-text-primary mb-6 uppercase tracking-tighter"
        >
          SITE_ARCHITECTURE
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-text-secondary text-xl max-w-3xl leading-relaxed font-light"
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
        className="grid grid-cols-1 lg:grid-cols-2 gap-16"
      >
        {/* Visual Directory */}
        <section>
          <div className="flex items-center gap-4 mb-8">
             <div className="w-10 h-10 rounded-none bg-brand/10 flex items-center justify-center text-brand">
                <Folder size={20} />
             </div>
             <h2 className="text-xl font-heading font-black text-text-primary uppercase tracking-tight">DIRECTORY_TREE</h2>
          </div>
          
          <div className="bg-surface border border-border-strong rounded-none p-10 space-y-4">
            {navItems.map((item, idx) => (
               <div key={item.id} className="space-y-4">
                 <button onClick={() => onNavigate(item.id)} className="flex items-center gap-4 text-text-primary hover:text-brand transition-all group w-full text-left">
                    <span className="text-text-muted group-hover:text-brand/50 font-mono text-[10px] font-black uppercase tracking-widest">{String(idx).padStart(2, '0')}</span>
                    <span className="font-mono font-black uppercase tracking-widest text-xs">{item.label}</span>
                 </button>
                 {item.items && (
                    <div className="ml-6 pl-6 border-l border-border-strong space-y-3">
                       {item.items.map((sub, subIdx) => (
                          <button key={sub.id} onClick={() => onNavigate(sub.id)} className="flex items-center gap-4 text-[11px] font-mono font-bold uppercase tracking-tight text-text-secondary hover:text-brand transition-all w-full text-left py-1">
                             <FileText size={12} className="text-text-muted" />
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
        <section className="space-y-12">
           <div className="flex items-center gap-4 mb-8">
             <div className="w-10 h-10 rounded-none bg-brand/10 flex items-center justify-center text-brand">
                <FileText size={20} />
             </div>
             <h2 className="text-xl font-heading font-black text-text-primary uppercase tracking-tight">DEVOPS_ARTIFACTS</h2>
          </div>

          <div className="space-y-6">
             <div className="flex justify-between items-end">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-[0.4em]">SITEMAP_XML</label>
             </div>
             <CodeBlock code={xmlString} language="xml" filename="public/sitemap.xml" />
          </div>

          <div className="space-y-6">
             <div className="flex justify-between items-end">
                <label className="text-[10px] font-mono font-black text-text-muted uppercase tracking-[0.4em]">ROBOTS_TXT</label>
             </div>
             <CodeBlock code={robotsTxt} language="bash" filename="public/robots.txt" />
          </div>
        </section>
      </motion.div>
    </motion.div>
  );
};
