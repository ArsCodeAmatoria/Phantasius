import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function debugMath() {
  const markdown1 = `Simple display math:

$$x = y + z$$

More text.`;

  const markdown2 = `Matrix math:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

More text.`;

  for (const [name, markdown] of [['Simple', markdown1], ['Matrix', markdown2]]) {
    console.log(`=== Testing ${name} ===`);
    console.log('Input:', JSON.stringify(markdown));

    try {
      const result = unified()
        .use(remarkParse)
        .use(remarkMath)
        .parse(markdown);
      
      console.log('Math node:', JSON.stringify(result.children.find(c => c.type === 'math'), null, 2));
      console.log('');
      
    } catch (error) {
      console.error('Error:', error);
    }
  }
}

debugMath(); 