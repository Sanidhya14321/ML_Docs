
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { GoogleGenAI } from "@google/genai";
import { Newspaper, ExternalLink, Calendar, Tag, RefreshCw, AlertCircle, Loader2 } from 'lucide-react';

interface NewsItem {
  title: string;
  summary: string;
  category: 'Research' | 'Industry' | 'Tooling' | 'General';
  date: string;
  source: string;
  url?: string;
}

interface NewsFeedResponse {
  news: NewsItem[];
}

export const NewsFeedView: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchNews = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY is not set");
      }

      const ai = new GoogleGenAI({ apiKey });
      
      const prompt = `
        You are an expert AI news aggregator. Find the most significant and recent advancements, research papers, and industry news in machine learning and artificial intelligence from the last 7 days.
        
        Focus on:
        1. Breakthrough research papers (e.g., from Arxiv, Nature, conferences).
        2. Major industry announcements (e.g., from Google, OpenAI, Meta, Anthropic).
        3. New significant open-source tools or models.

        Return a JSON object with a 'news' array containing at least 5-7 items. 
        Each item must have:
        - title: A concise, catchy headline.
        - summary: A clear, technical summary (2-3 sentences).
        - category: One of 'Research', 'Industry', 'Tooling', 'General'.
        - date: The date of the event (YYYY-MM-DD).
        - source: The primary source (e.g., "Google DeepMind", "Arxiv", "TechCrunch").
        - url: A relevant URL if found (optional).
      `;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
          responseMimeType: "application/json"
        }
      });

      const responseText = response.text;
      if (!responseText) {
        throw new Error("No response from AI");
      }

      const data: NewsFeedResponse = JSON.parse(responseText);
      
      // Sort by date descending if possible, otherwise keep order
      const sortedNews = data.news.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
      
      setNews(sortedNews);
      setLastUpdated(new Date());

    } catch (err: any) {
      console.error("Failed to fetch news:", err);
      setError(err.message || "Failed to load news feed. Please try again.");
      
      // Fallback mock data for demo purposes if API fails (e.g. quota or network)
      // Only use if we really can't get data, but better to show error in a real app.
      // For this demo, I'll leave the error state visible.
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Research': return 'bg-purple-100 text-purple-700 dark:bg-purple-500/10 dark:text-purple-400 border-purple-200 dark:border-purple-500/20';
      case 'Industry': return 'bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400 border-blue-200 dark:border-blue-500/20';
      case 'Tooling': return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20';
      default: return 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400 border-slate-200 dark:border-slate-700';
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-5xl mx-auto pb-24 px-4 md:px-8"
    >
      <header className="mb-12 pt-8 border-b border-slate-200 dark:border-slate-800 pb-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 mb-2">
              <Newspaper size={20} />
              <span className="text-xs font-black uppercase tracking-widest">Live Intelligence</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-serif font-bold text-slate-900 dark:text-white leading-tight">
              Machine Learning News Feed
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2 max-w-2xl">
              Real-time updates on the latest research papers, industry shifts, and tooling advancements, curated by Gemini.
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {lastUpdated && (
              <span className="text-xs text-slate-500 font-mono hidden md:inline-block">
                Updated: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button 
              onClick={fetchNews}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors disabled:opacity-50"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
              Refresh
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="mb-8 p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/20 rounded-xl flex items-start gap-3 text-rose-700 dark:text-rose-400">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <div>
            <h3 className="font-bold text-sm">Unable to fetch news</h3>
            <p className="text-sm opacity-90">{error}</p>
          </div>
        </div>
      )}

      {isLoading && news.length === 0 ? (
        <div className="space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 animate-pulse">
              <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/4 mb-4"></div>
              <div className="h-8 bg-slate-200 dark:bg-slate-800 rounded w-3/4 mb-4"></div>
              <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-full mb-2"></div>
              <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {news.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="group relative bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 hover:border-indigo-500/30 transition-all hover:shadow-lg hover:shadow-indigo-500/5"
            >
              <div className="flex flex-col md:flex-row md:items-start gap-6">
                <div className="flex-1">
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
                      <Calendar size={12} />
                      {item.date}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
                      <Tag size={12} />
                      {item.source}
                    </span>
                  </div>
                  
                  <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                    {item.title}
                  </h2>
                  
                  <p className="text-slate-600 dark:text-slate-400 leading-relaxed text-sm">
                    {item.summary}
                  </p>
                </div>

                {item.url && (
                  <div className="shrink-0 mt-4 md:mt-0">
                    <a 
                      href={item.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center justify-center w-10 h-10 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 hover:bg-indigo-600 hover:text-white dark:hover:bg-indigo-500 transition-all"
                      title="Read Source"
                    >
                      <ExternalLink size={18} />
                    </a>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
};
