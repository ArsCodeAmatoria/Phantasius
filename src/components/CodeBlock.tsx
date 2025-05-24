'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { Copy, Check } from 'lucide-react';
import { useState } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
}

export function CodeBlock({ code, language = 'haskell', filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-8 rounded-lg overflow-hidden border border-slate-700 shadow-2xl bg-[#282a36]">
      {/* Window Header */}
      <div className="flex items-center justify-between bg-[#44475a] px-4 py-3 border-b border-slate-600">
        <div className="flex items-center space-x-3">
          {/* Traffic Light Buttons */}
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-[#ff5555]"></div>
            <div className="w-3 h-3 rounded-full bg-[#f1fa8c]"></div>
            <div className="w-3 h-3 rounded-full bg-[#50fa7b]"></div>
          </div>
          {/* Filename or Language */}
          <span className="text-sm font-medium text-[#f8f8f2] font-mono">
            {filename || `${language}.${language === 'haskell' ? 'hs' : language}`}
          </span>
        </div>
        
        {/* Copy Button */}
        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 px-2 py-1 rounded text-xs text-[#f8f8f2] hover:bg-[#6272a4] transition-colors duration-200"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3" />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Content */}
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={dracula}
          customStyle={{
            margin: 0,
            padding: '1.5rem',
            backgroundColor: '#282a36',
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={true}
          lineNumberStyle={{
            minWidth: '3em',
            paddingRight: '1em',
            color: '#6272a4',
            fontSize: '0.8rem',
          }}
          wrapLines={true}
          wrapLongLines={true}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
} 