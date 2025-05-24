'use client';

import { CodeBlock } from './CodeBlock';
import type { ContentBlock } from '@/lib/posts';

interface PostContentProps {
  contentBlocks: ContentBlock[];
}

export function PostContent({ contentBlocks }: PostContentProps) {
  return (
    <div className="prose prose-lg prose-philosophy max-w-none">
      {contentBlocks.map((block, index) => {
        if (block.type === 'code') {
          return (
            <CodeBlock
              key={index}
              code={block.content}
              language={block.language}
            />
          );
        } else {
          return (
            <div
              key={index}
              dangerouslySetInnerHTML={{ __html: block.content }}
            />
          );
        }
      })}
    </div>
  );
} 