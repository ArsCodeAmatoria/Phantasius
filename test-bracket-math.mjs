import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function testBracketMath() {
  // Try using \[ \] delimiters instead of $$
  const bracketContent = `Matrix examples:

\\[\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}\\]

Second matrix:

\\[\\begin{pmatrix}
1 & 2 \\\\
3 & 4
\\end{pmatrix}\\]`;

  console.log('=== Bracket math content ===');
  console.log(bracketContent);
  
  const tree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(bracketContent);
  
  function findMathNodes(node) {
    const mathNodes = [];
    
    function traverse(n) {
      if (n.type === 'math') {
        mathNodes.push({
          value: n.value,
          meta: n.meta
        });
      }
      if (n.children) {
        n.children.forEach(traverse);
      }
    }
    
    traverse(node);
    return mathNodes;
  }
  
  const mathNodes = findMathNodes(tree);
  
  console.log(`\\nFound ${mathNodes.length} math nodes`);
  mathNodes.forEach((node, index) => {
    console.log(`\\nMath node ${index + 1}:`);
    console.log('Value (first 50 chars):', JSON.stringify(node.value.substring(0, 50)));
  });
}

testBracketMath(); 