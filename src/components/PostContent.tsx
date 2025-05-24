'use client';

import { useEffect, useRef } from 'react';
import { CodeBlock } from './CodeBlock';
import { createRoot } from 'react-dom/client';

interface PostContentProps {
  content: string;
}

export function PostContent({ content }: PostContentProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!contentRef.current) return;

    // Find all pre > code elements and replace them with our CodeBlock component
    const preElements = contentRef.current.querySelectorAll('pre code');
    
    preElements.forEach((codeElement) => {
      const preElement = codeElement.parentElement;
      if (!preElement) return;

      const codeText = codeElement.textContent || '';
      const className = codeElement.className || '';
      
      // Extract language from className (e.g., "language-haskell" -> "haskell")
      const languageMatch = className.match(/language-(\w+)/);
      const language = languageMatch ? languageMatch[1] : 'haskell';

      // Create a container for our React component
      const container = document.createElement('div');
      preElement.parentNode?.replaceChild(container, preElement);

      // Render our CodeBlock component
      const root = createRoot(container);
      root.render(<CodeBlock code={codeText} language={language} />);
    });
  }, [content]);

  return (
    <div 
      ref={contentRef}
      className="prose prose-lg prose-philosophy max-w-none"
      dangerouslySetInnerHTML={{ __html: content }}
    />
  );
} 