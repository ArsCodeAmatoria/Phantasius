import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

async function testWorkingApproach() {
  let markdown = `# Test Single Matrix

A simple matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

End of content.`;

  console.log('=== Original markdown ===');
  console.log(markdown);
  
  // Use the EXACT same preprocessing as your working version
  markdown = markdown.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  markdown = markdown.replace(/\\$([^$\\n]+?)\\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });

  console.log('\\n=== Preprocessed markdown ===');
  console.log(markdown);

  try {
    const result = await unified()
      .use(remarkParse) 
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
      .process(markdown);
    
    let html = result.toString();
    
    // Post-process: Restore & characters in the final HTML
    html = html.replace(/XAMPERSANDX/g, '&');
    
    console.log('\\n=== Final HTML ===');
    console.log(html);
    
    if (html.includes('class="katex"')) {
      console.log('\\n✅ KaTeX rendering detected!');
    } else {
      console.log('\\n❌ No KaTeX rendering found');
    }
    
  } catch (error) {
    console.error('Error processing KaTeX:', error);
  }
}

testWorkingApproach(); 