
import React, { useEffect } from 'react';
import { DocMetadata } from '../types';

interface SEOHeadProps {
  metadata: DocMetadata;
  path: string;
}

export const SEOHead: React.FC<SEOHeadProps> = ({ metadata, path }) => {
  useEffect(() => {
    // 1. Update Title
    document.title = `${metadata.title} | AI Mastery Hub`;

    // 2. Update Meta Description
    let metaDescription = document.querySelector("meta[name='description']");
    if (!metaDescription) {
      metaDescription = document.createElement('meta');
      metaDescription.setAttribute('name', 'description');
      document.head.appendChild(metaDescription);
    }
    metaDescription.setAttribute('content', metadata.description);

    // 3. Update Open Graph Tags (Simulated)
    const updateMeta = (property: string, content: string) => {
      let tag = document.querySelector(`meta[property='${property}']`);
      if (!tag) {
        tag = document.createElement('meta');
        tag.setAttribute('property', property);
        document.head.appendChild(tag);
      }
      tag.setAttribute('content', content);
    };

    updateMeta('og:title', metadata.title);
    updateMeta('og:description', metadata.description);
    updateMeta('og:type', 'article');
    updateMeta('og:url', `https://ai-codex.dev/#/${path}`);
    // In a real Next.js app, this would point to the /api/og route
    updateMeta('og:image', `https://ai-codex.dev/api/og?title=${encodeURIComponent(metadata.title)}`);
    
  }, [metadata, path]);

  return null;
};
