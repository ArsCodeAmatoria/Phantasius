import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

const postsDirectory = path.join(process.cwd(), 'posts');

export interface CodeBlock {
  type: 'code';
  content: string;
  language: string;
}

export interface HtmlBlock {
  type: 'html';
  content: string;
}

export type ContentBlock = CodeBlock | HtmlBlock;

export interface PostData {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  tags: string[];
  content: string;
  contentBlocks: ContentBlock[];
}

export interface PostMeta {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  tags: string[];
}

export async function getPostData(slug: string): Promise<PostData> {
  const fullPath = path.join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');

  // Use gray-matter to parse the post metadata section
  const matterResult = matter(fileContents);

  // ROBUST FIX: Properly tokenize math expressions to prevent greedy parsing
  let processedMarkdown = matterResult.content;
  
  // Step 1: Replace & with placeholder in existing math expressions
  processedMarkdown = processedMarkdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  processedMarkdown = processedMarkdown.replace(/\$([^$\n]+?)\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });
  
  // Step 2: FORCE PROPER TOKENIZATION by replacing $$ with unique markers
  const mathBlocks: string[] = [];
  let blockIndex = 0;
  
  // Extract all display math blocks and replace with unique markers
  processedMarkdown = processedMarkdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, content) => {
    mathBlocks.push(content);
    return `MATH_BLOCK_${blockIndex++}_MARKER`;
  });
  
  // Now put them back one by one with proper spacing
  for (let i = 0; i < mathBlocks.length; i++) {
    const marker = `MATH_BLOCK_${i}_MARKER`;
    processedMarkdown = processedMarkdown.replace(marker, `$$${mathBlocks[i]}$$`);
  }

  // Use unified pipeline to convert markdown to HTML with KaTeX math support
  const processedContent = await unified()
    .use(remarkParse) // Parse markdown
    .use(remarkGfm) // GitHub Flavored Markdown for tables
    .use(remarkMath) // Parse math expressions
    .use(remarkRehype, { 
      allowDangerousHtml: true,
      entities: {
        useShortestReferences: true,
        useNamedReferences: true
      }
    }) // Convert to HTML with better entity handling
    .use(rehypeKatex, {
      strict: false,
      trust: true
    }) // Render math with KaTeX (less strict)
    .use(rehypeStringify, { 
      allowDangerousHtml: true,
      entities: {
        useShortestReferences: true,
        useNamedReferences: true
      }
    }) // Convert to string
    .process(processedMarkdown);
  let contentHtml = processedContent.toString();
  
  // Post-process: Restore & characters
  contentHtml = contentHtml.replace(/XAMPERSANDX/g, '&');

  // Extract code blocks using regex (more reliable than DOM parsing for this case)
  const contentBlocks: ContentBlock[] = [];
  
  // Regex to match <pre><code class="language-xxx">content</code></pre>
  const codeBlockRegex = /<pre><code(?:\s+class="language-(\w+)")?>([\s\S]*?)<\/code><\/pre>/g;
  
  let lastIndex = 0;
  let match;
  
  while ((match = codeBlockRegex.exec(contentHtml)) !== null) {
    // Add HTML content before this code block
    if (match.index > lastIndex) {
      const htmlContent = contentHtml.substring(lastIndex, match.index).trim();
      if (htmlContent) {
        contentBlocks.push({
          type: 'html',
          content: htmlContent
        });
      }
    }
    
    // Add the code block
    const language = match[1] || 'haskell'; // Default to haskell if no language specified
    const codeContent = match[2];
    
    // Decode HTML entities in the code content
    const decodedCode = codeContent
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
    
    contentBlocks.push({
      type: 'code',
      content: decodedCode,
      language: language
    });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add any remaining HTML content after the last code block
  if (lastIndex < contentHtml.length) {
    const remainingHtml = contentHtml.substring(lastIndex).trim();
    if (remainingHtml) {
      contentBlocks.push({
        type: 'html',
        content: remainingHtml
      });
    }
  }

  // Combine the data with the slug and contentHtml
  return {
    slug,
    content: contentHtml,
    contentBlocks,
    title: matterResult.data.title,
    date: matterResult.data.date,
    excerpt: matterResult.data.excerpt,
    tags: matterResult.data.tags || [],
  };
}

export function getAllPostSlugs() {
  const fileNames = fs.readdirSync(postsDirectory);
  return fileNames
    .filter((name) => name.endsWith('.md'))
    .map((name) => name.replace(/\.md$/, ''));
}

export function getSortedPostsData(): PostMeta[] {
  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames
    .filter((name) => name.endsWith('.md'))
    .map((fileName) => {
      // Remove ".md" from file name to get slug
      const slug = fileName.replace(/\.md$/, '');

      // Read markdown file as string
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, 'utf8');

      // Use gray-matter to parse the post metadata section
      const matterResult = matter(fileContents);

      // Combine the data with the slug
      return {
        slug,
        title: matterResult.data.title,
        date: matterResult.data.date,
        excerpt: matterResult.data.excerpt,
        tags: matterResult.data.tags || [],
      };
    });

  // Sort posts by date
  return allPostsData.sort((a, b) => {
    if (a.date < b.date) {
      return 1;
    } else {
      return -1;
    }
  });
} 