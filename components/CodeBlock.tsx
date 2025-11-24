import React from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'python' }) => {
  return (
    <div className="my-6 rounded-lg overflow-hidden border border-slate-700 shadow-xl bg-slate-900">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
        </div>
        <span className="text-xs text-slate-400 font-mono lowercase">{language}</span>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="font-mono text-sm leading-6 text-slate-300">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};