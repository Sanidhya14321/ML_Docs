
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { GoogleGenAI } from "@google/genai";
import { Newspaper, ExternalLink, Calendar, Tag, RefreshCw, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '../components/Button';

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
      const MOCK_NEWS: NewsItem[] = [
        {
          title: "Gemini 4.0 'Ultra' Achieves 99.8% on MMLU-Pro",
          summary: "Google DeepMind's latest multimodal model shatters benchmarks, demonstrating near-perfect reasoning capabilities across STEM fields and creative writing tasks.",
          category: "Industry",
          date: "2026-02-28",
          source: "Google DeepMind Blog",
          url: "https://blog.google/technology/ai/gemini-4-ultra"
        },
        {
          title: "Self-Correcting Code Agents: The End of Unit Tests?",
          summary: "A new paper from Stanford and MIT introduces 'Reflexion-X', an agentic framework that autonomously writes, tests, and fixes code with zero human intervention, achieving SOTA on SWE-bench.",
          category: "Research",
          date: "2026-02-26",
          source: "Arxiv",
          url: "https://arxiv.org/abs/2602.12345"
        },
        {
          title: "OpenAI Releases 'Sora 3' with Real-Time Video Generation",
          summary: "The new video generation model can create 4K, 60fps video in real-time, enabling interactive movie experiences and dynamic game environments.",
          category: "Industry",
          date: "2026-02-24",
          source: "OpenAI",
          url: "https://openai.com/sora-3"
        },
        {
          title: "PyTorch 3.0: Native Distributed Training on Edge Devices",
          summary: "Meta's latest PyTorch release optimizes distributed training for edge computing, allowing massive models to be fine-tuned on clusters of consumer hardware.",
          category: "Tooling",
          date: "2026-02-20",
          source: "PyTorch Foundation",
          url: "https://pytorch.org/blog/pytorch-3-release"
        },
        {
          title: "The 'Reasoning Gap': Why LLMs Still Struggle with Causal Inference",
          summary: "A critical analysis published in Nature Machine Intelligence argues that despite scaling, current transformer architectures fundamentally lack causal reasoning abilities.",
          category: "Research",
          date: "2026-02-18",
          source: "Nature",
          url: "https://nature.com/articles/s42256-026-00123-x"
        }
      ];
      
      setNews(MOCK_NEWS);
      setLastUpdated(new Date());
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Research': return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      case 'Industry': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'Tooling': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      default: return 'bg-surface-active text-text-muted border-border-strong';
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-6xl mx-auto pb-24 px-6 md:px-10"
    >
      <header className="mb-16 pt-12 border-b border-border-strong pb-12 transition-all duration-300">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-3 text-brand">
              <Newspaper size={20} />
              <span className="text-[10px] font-mono font-black uppercase tracking-[0.4em]">LIVE_INTELLIGENCE_STREAM</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-display font-black text-text-primary leading-none uppercase tracking-tighter">
              NEURAL_FEED
            </h1>
            <p className="text-text-secondary mt-4 max-w-2xl text-lg font-light leading-relaxed">
              Real-time synchronization with global AI advancements, research breakthroughs, and industry shifts.
            </p>
          </div>
          
          <div className="flex items-center gap-6">
            {lastUpdated && (
              <span className="text-[10px] text-text-muted font-mono font-black uppercase tracking-widest hidden md:inline-block">
                LAST_SYNC: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <Button 
              onClick={fetchNews}
              disabled={isLoading}
              variant="outline"
              size="sm"
              leftIcon={isLoading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            >
              REFRESH_STREAM
            </Button>
          </div>
        </div>
      </header>

      {error && (
        <div className="mb-12 p-6 bg-rose-500/5 border border-rose-500/20 rounded-none flex items-start gap-4 text-rose-500">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <div>
            <h3 className="font-mono font-black text-xs uppercase tracking-widest mb-1">SYNC_ERROR_DETECTED</h3>
            <p className="text-sm opacity-80 font-mono uppercase tracking-tight">{error}</p>
          </div>
        </div>
      )}

      {isLoading && news.length === 0 ? (
        <div className="space-y-8">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-surface border border-border-strong rounded-none p-8 animate-pulse">
              <div className="h-3 bg-border-strong rounded-none w-1/4 mb-6"></div>
              <div className="h-10 bg-border-strong rounded-none w-3/4 mb-6"></div>
              <div className="h-4 bg-border-strong rounded-none w-full mb-3"></div>
              <div className="h-4 bg-border-strong rounded-none w-2/3"></div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-8">
          {news.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="group relative bg-surface border border-border-strong rounded-none p-8 hover:border-brand transition-all duration-300"
            >
              <div className="flex flex-col md:flex-row md:items-start gap-10">
                <div className="flex-1 space-y-6">
                  <div className="flex flex-wrap items-center gap-4">
                    <span className={`px-2.5 py-1 rounded-none text-[9px] font-mono font-black uppercase tracking-widest border transition-colors duration-300 ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                    <span className="flex items-center gap-2 text-[10px] text-text-muted font-mono font-black uppercase tracking-widest">
                      <Calendar size={12} />
                      {item.date}
                    </span>
                    <span className="flex items-center gap-2 text-[10px] text-text-muted font-mono font-black uppercase tracking-widest">
                      <Tag size={12} />
                      {item.source}
                    </span>
                  </div>
                  
                  <h2 className="text-2xl font-display font-black text-text-primary uppercase tracking-tight group-hover:text-brand transition-colors leading-tight">
                    {item.title}
                  </h2>
                  
                  <p className="text-text-secondary leading-relaxed text-base font-light">
                    {item.summary}
                  </p>
                </div>

                {item.url && (
                  <div className="shrink-0 mt-4 md:mt-0">
                    <a 
                      href={item.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center justify-center w-12 h-12 rounded-none bg-app border border-border-strong text-text-muted hover:bg-brand hover:text-app hover:border-brand transition-all"
                      title="READ_SOURCE"
                    >
                      <ExternalLink size={20} />
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
