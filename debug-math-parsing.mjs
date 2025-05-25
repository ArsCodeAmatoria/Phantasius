import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function debugMathParsing() {
  // Test case that demonstrates the problem
  const problematicContent = `First matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

Second matrix:

$$\\begin{pmatrix}
1 & 2 \\\\
3 & 4
\\end{pmatrix}$$`;

  console.log('=== Input content ===');
  console.log(problematicContent);
  
  // Parse with remark-math
  const tree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(problematicContent);
  
  // Find math nodes
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
  
  console.log(`\\n=== Found ${mathNodes.length} math nodes ===`);
  mathNodes.forEach((node, index) => {
    console.log(`\\nMath node ${index + 1}:`);
    console.log('Meta:', JSON.stringify(node.meta));
    console.log('Value (first 100 chars):', JSON.stringify(node.value.substring(0, 100)));
  });
  
  // Test with a different approach - what if we add more spacing?
  const spacedContent = `First matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

Text in between.

Second matrix:

$$\\begin{pmatrix}
1 & 2 \\\\
3 & 4
\\end{pmatrix}$$

End.`;

  console.log('\\n\\n=== Testing with more spacing ===');
  const spacedTree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(spacedContent);
    
  const spacedMathNodes = findMathNodes(spacedTree);
  console.log(`Found ${spacedMathNodes.length} math nodes with spacing`);
  spacedMathNodes.forEach((node, index) => {
    console.log(`\\nSpaced math node ${index + 1}:`);
    console.log('Value (first 100 chars):', JSON.stringify(node.value.substring(0, 100)));
  });
}

debugMathParsing(); 