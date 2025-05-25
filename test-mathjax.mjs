import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeMathjax from 'rehype-mathjax';
import rehypeStringify from 'rehype-stringify';

async function testMathJax() {
  // Test just the problematic matrix section
  const matrixContent = `## Matrix Test

A simple matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

Identity matrix:

$$I = \\begin{pmatrix}
1 & 0 \\\\
0 & 1
\\end{pmatrix}$$`;

  console.log('=== Testing with MathJax ===');
  console.log(matrixContent);
  
  // Preprocess: Replace & with placeholder in math expressions
  let processedMarkdown = matrixContent;
  
  processedMarkdown = processedMarkdown.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });

  try {
    const result = await unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkMath)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeMathjax) // Using MathJax instead of KaTeX
      .use(rehypeStringify, { allowDangerousHtml: true })
      .process(processedMarkdown);
    
    let html = result.toString();
    html = html.replace(/XAMPERSANDX/g, '&');
    
    console.log('\\n=== MathJax HTML Output ===');
    console.log(html);
    
    // Check for math rendering
    if (html.includes('class="MathJax"') || html.includes('data-mjx')) {
      console.log('\\n✅ MathJax rendering detected!');
    } else if (html.includes('math-error')) {
      console.log('\\n❌ MathJax errors detected!');
    } else {
      console.log('\\n⚠️  Math content found but no specific MathJax markers');
    }
    
  } catch (error) {
    console.error('Error processing MathJax:', error);
  }
}

testMathJax(); 