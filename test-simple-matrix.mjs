import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

async function testSimpleMatrix() {
  // Test case 1: Your original problematic content
  const problematicContent = `A simple matrix:

a & b \\\\ c & d \\end{pmatrix}$$

The identity matrix:

$$I = \\begin{pmatrix}
1 & 0 & 0 \\\\
0 & 1 & 0 \\\\
0 & 0 & 1
\\end{pmatrix}$$

A vector in quantum mechanics:

$$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$

Where $|\\alpha|^2 + |\\beta|^2 = 1$.`;

  // Test case 2: Corrected content
  const correctedContent = `A simple matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

The identity matrix:

$$I = \\begin{pmatrix}
1 & 0 & 0 \\\\
0 & 1 & 0 \\\\
0 & 0 & 1
\\end{pmatrix}$$

A vector in quantum mechanics:

$$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$

Where $|\\alpha|^2 + |\\beta|^2 = 1$.`;

  for (const [name, content] of [['Problematic', problematicContent], ['Corrected', correctedContent]]) {
    console.log(`\n=== Testing ${name} Content ===`);
    console.log(content.substring(0, 200) + '...');

    // Replace & with placeholder in math expressions
    let processedContent = content.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (match, mathContent) => {
      return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
    });
    
    processedContent = processedContent.replace(/\\$([^$\\n]+?)\\$/g, (match, mathContent) => {
      return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
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
      
      if (html.includes('class="katex"')) {
        console.log('✅ KaTeX rendering successful!');
      } else if (html.includes('katex-error')) {
        console.log('❌ KaTeX parsing error detected');
        console.log('Error snippet:', html.match(/class="katex-error"[^>]*title="[^"]*"/)?.[0] || 'No details');
      } else {
        console.log('❌ No KaTeX rendering found');
      }
      
    } catch (error) {
      console.error('Error:', error.message);
    }
  }
}

testSimpleMatrix(); 