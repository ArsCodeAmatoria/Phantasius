import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import remarkFrontmatter from 'remark-frontmatter';
import { readFileSync } from 'fs';

async function testCorrectedMath() {
  const markdown = readFileSync('corrected-math-example.md', 'utf8');
  
  console.log('=== Original content ===');
  console.log(markdown);
  
  // Preprocess: Temporarily replace & with a placeholder in math expressions
  let processedMarkdown = markdown;
  
  // Replace & with placeholder in display math ($$...$$)
  processedMarkdown = processedMarkdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  // Replace & with placeholder in inline math ($...$)
  processedMarkdown = processedMarkdown.replace(/\$([^$\n]+?)\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });

  try {
    const result = await unified()
      .use(remarkParse)
      .use(remarkFrontmatter, ['yaml']) // Handle YAML front matter
      .use(remarkGfm)
      .use(remarkMath)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeKatex, { strict: false, trust: true })
      .use(rehypeStringify, { allowDangerousHtml: true })
      .process(processedMarkdown);
    
    let html = result.toString();
    
    // Post-process: Restore & characters in the final HTML
    html = html.replace(/XAMPERSANDX/g, '&');
    
    console.log('\n=== Rendered HTML ===');
    console.log(html);
    
    // Check if KaTeX classes are present
    if (html.includes('class="katex"')) {
      console.log('\n✅ KaTeX rendering successful!');
    } else {
      console.log('\n❌ No KaTeX rendering found');
    }
    
  } catch (error) {
    console.error('Error processing KaTeX:', error);
  }
}

testCorrectedMath(); 