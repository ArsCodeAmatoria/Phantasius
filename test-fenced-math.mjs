import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function testFencedMath() {
  // Try using fenced math blocks instead of $$
  const fencedContent = `Matrix examples:

\`\`\`math
\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}
\`\`\`

And another:

\`\`\`math
\\begin{pmatrix}
1 & 2 \\\\
3 & 4
\\end{pmatrix}
\`\`\``;

  console.log('=== Fenced math content ===');
  console.log(fencedContent);
  
  const tree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(fencedContent);
  
  function findMathNodes(node) {
    const mathNodes = [];
    const codeNodes = [];
    
    function traverse(n) {
      if (n.type === 'math') {
        mathNodes.push({
          type: 'math',
          value: n.value,
          meta: n.meta
        });
      }
      if (n.type === 'code' && n.lang === 'math') {
        codeNodes.push({
          type: 'code',
          value: n.value,
          lang: n.lang
        });
      }
      if (n.children) {
        n.children.forEach(traverse);
      }
    }
    
    traverse(node);
    return { mathNodes, codeNodes };
  }
  
  const { mathNodes, codeNodes } = findMathNodes(tree);
  
  console.log(`\\nFound ${mathNodes.length} math nodes and ${codeNodes.length} code nodes`);
  
  codeNodes.forEach((node, index) => {
    console.log(`\\nCode node ${index + 1}:`);
    console.log('Language:', node.lang);
    console.log('Value:', JSON.stringify(node.value));
  });
}

testFencedMath(); 