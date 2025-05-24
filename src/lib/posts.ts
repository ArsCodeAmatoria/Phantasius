import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
import remarkGfm from 'remark-gfm';
import { parse, HTMLElement, Node } from 'node-html-parser';

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

  // Use remark to convert markdown into HTML string with GitHub Flavored Markdown
  const processedContent = await remark()
    .use(remarkGfm) // GitHub Flavored Markdown for tables
    .use(html, { sanitize: false })
    .process(matterResult.content);
  const contentHtml = processedContent.toString();

  // Parse HTML and extract code blocks
  const root = parse(contentHtml);
  const contentBlocks: ContentBlock[] = [];
  let currentHtml = '';

  const processNode = (node: Node) => {
    if (node instanceof HTMLElement && node.tagName === 'PRE') {
      const codeElement = node.querySelector('code');
      if (codeElement) {
        // Save any accumulated HTML
        if (currentHtml.trim()) {
          contentBlocks.push({ type: 'html', content: currentHtml.trim() });
          currentHtml = '';
        }

        // Extract code content and language
        const codeText = codeElement.text || '';
        const className = codeElement.getAttribute('class') || '';
        const languageMatch = className.match(/language-(\w+)/);
        const language = languageMatch ? languageMatch[1] : 'haskell';

        contentBlocks.push({
          type: 'code',
          content: codeText,
          language: language
        });
        return; // Skip adding this to HTML
      }
    }
    
    // Add regular content to HTML
    if (node instanceof HTMLElement) {
      currentHtml += node.outerHTML;
    } else {
      currentHtml += node.toString();
    }
  };

  // Process all child nodes
  root.childNodes.forEach(processNode);

  // Add any remaining HTML
  if (currentHtml.trim()) {
    contentBlocks.push({ type: 'html', content: currentHtml.trim() });
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