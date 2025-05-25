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

async function testActualProcessing() {
  // Read the actual test-katex.md file
  const fullPath = path.join(process.cwd(), 'posts', 'test-katex.md');
  const fileContents = fs.readFileSync(fullPath, 'utf8');

  // Use gray-matter to parse the post metadata section (same as posts.ts)
  const matterResult = matter(fileContents);

  // Enhanced preprocessing to handle math expressions properly
  let processedMarkdown = matterResult.content;
  
  // Step 1: Replace & with placeholder in math expressions
  processedMarkdown = processedMarkdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  processedMarkdown = processedMarkdown.replace(/\$([^$\n]+?)\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });
  
  // Step 2: FORCE PROPER TOKENIZATION by replacing $$ with unique markers
  const mathBlocks = [];
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

  console.log('=== First 500 chars of original content ===');
  console.log(processedMarkdown.substring(0, 500));

  // Use unified pipeline to convert markdown to HTML with KaTeX math support (same as posts.ts)
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
  console.log('\n=== Before ampersand restoration (first occurrence) ===');
  const xampIndex = contentHtml.indexOf('XAMPERSANDX');
  if (xampIndex !== -1) {
    console.log(contentHtml.substring(xampIndex - 20, xampIndex + 50));
  }
  
  contentHtml = contentHtml.replace(/XAMPERSANDX/g, '&');
  
  console.log('\n=== After ampersand restoration (same position) ===');
  if (xampIndex !== -1) {
    console.log(contentHtml.substring(xampIndex - 20, xampIndex + 30));
  }

  // Check specifically for the matrix section
  const matrixSectionStart = contentHtml.indexOf('Matrix and Vector Notation');
  if (matrixSectionStart !== -1) {
    const matrixSection = contentHtml.substring(matrixSectionStart, matrixSectionStart + 1000);
    console.log('\n=== Matrix section ===');
    console.log(matrixSection);
  }
  
  // Count math expressions and errors
  const katexCount = (contentHtml.match(/class="katex"/g) || []).length;
  const errorCount = (contentHtml.match(/katex-error/g) || []).length;
  
  console.log(`\n=== Results ===`);
  console.log(`✅ KaTeX elements rendered: ${katexCount}`);
  
  if (errorCount > 0) {
    console.log(`❌ KaTeX errors found: ${errorCount}`);
    
    // Extract error details
    const errorMatches = contentHtml.match(/class="katex-error"[^>]*title="([^"]*)"/g) || [];
    errorMatches.forEach((error, index) => {
      const titleMatch = error.match(/title="([^"]*)"/);
      if (titleMatch) {
        console.log(`   Error ${index + 1}: ${titleMatch[1].substring(0, 100)}...`);
      }
    });
  } else {
    console.log(`✅ No KaTeX errors detected!`);
  }
}

testActualProcessing(); 