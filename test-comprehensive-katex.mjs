import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import remarkFrontmatter from 'remark-frontmatter';
import { readFileSync } from 'fs';

async function testComprehensiveKatex() {
  const markdown = readFileSync('posts/test-katex.md', 'utf8');
  
  console.log('=== Testing Updated KaTeX Page ===');
  console.log(`Content length: ${markdown.length} characters`);
  
  // Count math expressions with CORRECT regex
  const displayMathCount = (markdown.match(/\$\$[\s\S]*?\$\$/g) || []).length;
  const inlineMathCount = (markdown.match(/\$[^$\n]+?\$/g) || []).length;
  
  console.log(`Display math expressions: ${displayMathCount}`);
  console.log(`Inline math expressions: ${inlineMathCount}`);
  
  // Preprocess: Replace & with placeholder in math expressions
  let processedMarkdown = markdown;
  
  // Replace & with placeholder in display math ($$...$$) - CORRECTED REGEX
  processedMarkdown = processedMarkdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  // Replace & with placeholder in inline math ($...$) - CORRECTED REGEX
  processedMarkdown = processedMarkdown.replace(/\$([^$\n]+?)\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });

  try {
    const result = await unified()
      .use(remarkParse)
      .use(remarkFrontmatter, ['yaml'])
      .use(remarkGfm)
      .use(remarkMath)
      .use(remarkRehype, { 
        allowDangerousHtml: true,
        entities: {
          useShortestReferences: true,
          useNamedReferences: true
        }
      })
      .use(rehypeKatex, {
        strict: false,
        trust: true
      })
      .use(rehypeStringify, { 
        allowDangerousHtml: true,
        entities: {
          useShortestReferences: true,
          useNamedReferences: true
        }
      })
      .process(processedMarkdown);
    
    let html = result.toString();
    
    // Post-process: Restore & characters in the final HTML
    html = html.replace(/XAMPERSANDX/g, '&');
    
    // Analyze results
    const katexCount = (html.match(/class="katex"/g) || []).length;
    const errorCount = (html.match(/katex-error/g) || []).length;
    
    console.log(`\n=== Results ===`);
    console.log(`✅ KaTeX elements rendered: ${katexCount}`);
    
    if (errorCount > 0) {
      console.log(`❌ KaTeX errors found: ${errorCount}`);
      
      // Extract error details
      const errorMatches = html.match(/class="katex-error"[^>]*title="([^"]*)"/g) || [];
      errorMatches.forEach((error, index) => {
        const titleMatch = error.match(/title="([^"]*)"/);
        if (titleMatch) {
          console.log(`   Error ${index + 1}: ${titleMatch[1].substring(0, 100)}...`);
        }
      });
    } else {
      console.log(`✅ No KaTeX errors detected!`);
    }
    
    // Check if all expected math expressions were rendered
    const expectedTotal = displayMathCount + inlineMathCount;
    if (katexCount >= expectedTotal) {
      console.log(`✅ All ${expectedTotal} math expressions appear to be rendered successfully!`);
    } else {
      console.log(`⚠️  Expected ${expectedTotal} math expressions, but only ${katexCount} were rendered.`);
    }
    
  } catch (error) {
    console.error('Error processing KaTeX:', error);
  }
}

testComprehensiveKatex(); 