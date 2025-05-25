import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

async function testSingleMatrix() {
  const correctMatrix = `A simple matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

End of content.`;

  console.log('=== Testing Single Correct Matrix ===');
  console.log(correctMatrix);

  // Replace & with placeholder
  let processedContent = correctMatrix.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });

  try {
    const result = await unified()
      .use(remarkParse)
      .use(remarkMath)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeKatex, { strict: false, trust: true })
      .use(rehypeStringify, { allowDangerousHtml: true })
      .process(processedContent);
    
    let html = result.toString();
    html = html.replace(/XAMPERSANDX/g, '&');
    
    console.log('\\n=== Rendered HTML ===');
    console.log(html);
    
    if (html.includes('class="katex"')) {
      console.log('\\n✅ KaTeX rendering successful!');
    } else if (html.includes('katex-error')) {
      console.log('\\n❌ KaTeX parsing error detected');
      const errorMatch = html.match(/title="([^"]*)"/);
      if (errorMatch) {
        console.log('Error details:', errorMatch[1]);
      }
    } else {
      console.log('\\n❌ No KaTeX rendering found');
    }
    
  } catch (error) {
    console.error('Error:', error.message);
  }
}

testSingleMatrix(); 