const { unified } = require('unified');
const remarkParse = require('remark-parse');
const remarkMath = require('remark-math');
const remarkRehype = require('remark-rehype');
const rehypeKatex = require('rehype-katex');
const rehypeStringify = require('rehype-stringify');

async function testKatex() {
  const markdown = `
# Test Math

Inline math: $E = mc^2$

Display math:
$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$
`;

  try {
    const result = await unified()
      .use(remarkParse)
      .use(remarkMath)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeKatex)
      .use(rehypeStringify, { allowDangerousHtml: true })
      .process(markdown);
    
    console.log('=== KATEX PROCESSING RESULT ===');
    console.log(result.toString());
    console.log('=== END RESULT ===');
  } catch (error) {
    console.error('Error processing KaTeX:', error);
  }
}

testKatex(); 