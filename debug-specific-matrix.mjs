import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

async function debugSpecificMatrix() {
  // Test the matrix multiplication example that might be causing issues
  const matrixMultiplication = `Matrix multiplication example:

$$\\begin{pmatrix}
1 & 2 \\\\
3 & 4
\\end{pmatrix}
\\begin{pmatrix}
x \\\\
y
\\end{pmatrix}
=
\\begin{pmatrix}
x + 2y \\\\
3x + 4y
\\end{pmatrix}$$

End of content.`;

  console.log('=== Testing Matrix Multiplication ===');
  console.log(matrixMultiplication);

  // Apply preprocessing with CORRECT regex
  let processedContent = matrixMultiplication.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    console.log('Processing math block:', JSON.stringify(mathContent.substring(0, 100)));
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });

  console.log('\\n=== After preprocessing ===');
  console.log(processedContent.substring(0, 300));

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

debugSpecificMatrix(); 