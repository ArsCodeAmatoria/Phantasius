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

// Function to get proper file extension
function getFileExtension(language: string): string {
  const extensions: Record<string, string> = {
    javascript: 'js',
    typescript: 'ts',
    python: 'py',
    haskell: 'hs',
    rust: 'rs',
    cpp: 'cpp',
    c: 'c',
    java: 'java',
    go: 'go',
    php: 'php',
    ruby: 'rb',
    bash: 'sh',
    shell: 'sh',
    sql: 'sql',
    css: 'css',
    html: 'html',
    json: 'json',
    yaml: 'yml',
    xml: 'xml',
    markdown: 'md'
  };
  
  return extensions[language.toLowerCase()] || language;
}

export function CodeBlock({ code, language = 'haskell', filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const fileExtension = getFileExtension(language);
  const displayFilename = filename || `${language}.${fileExtension}`;

  return (
    <div className="my-8 rounded-lg overflow-hidden shadow-2xl">
      {/* Window Header */}
      <div className="flex items-center justify-between bg-slate-700 px-4 py-3 border-b border-slate-600">
        <div className="flex items-center space-x-3">
          {/* Traffic Light Buttons */}
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-[#ff5555]"></div>
            <div className="w-3 h-3 rounded-full bg-[#f1fa8c]"></div>
            <div className="w-3 h-3 rounded-full bg-[#50fa7b]"></div>
          </div>
          {/* Filename */}
          <span className="text-sm font-medium text-slate-200 font-mono">
            {displayFilename}
          </span>
        </div>
        
        {/* Copy Button */}
        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 px-2 py-1 rounded text-xs text-slate-200 hover:bg-slate-600 transition-colors duration-200"
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

      {/* Code Content - SyntaxHighlighter with background overrides */}
      <div 
        className="relative code-block-override"
        style={{ 
          backgroundColor: '#2d2d2d !important',
          padding: 0,
          fontSize: '0.875rem',
          lineHeight: '1.5',
        }}
      >
        <SyntaxHighlighter
          language={language}
          style={dracula}
          className="code-pre-override"
          customStyle={{
            backgroundColor: '#2d2d2d !important',
            color: '#f8f8f2 !important',
            margin: '0 !important',
            padding: '1.5rem !important',
            border: 'none !important',
            borderRadius: '0 !important',
            boxShadow: 'none !important',
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={true}
          lineNumberStyle={{
            minWidth: '3em',
            paddingRight: '1em',
            color: '#9ca3af',
            fontSize: '0.8rem',
            backgroundColor: '#2d2d2d !important',
          }}
          wrapLines={true}
          wrapLongLines={true}
          CodeTag={({ children, ...props }) => (
            <code 
              {...props} 
              className="code-element-override"
              style={{ 
                backgroundColor: '#2d2d2d !important',
                color: 'inherit !important',
                padding: '0 !important',
                margin: '0 !important',
                border: 'none !important',
                borderRadius: '0 !important',
                fontSize: 'inherit !important',
              }}
            >
              {children}
            </code>
          )}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
} 