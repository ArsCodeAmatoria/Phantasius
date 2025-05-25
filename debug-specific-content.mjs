import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function debugSpecificContent() {
  // Read the actual file
  const fullPath = path.join(process.cwd(), 'posts', 'test-katex.md');
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const matterResult = matter(fileContents);
  
  // Extract just the matrix section
  const content = matterResult.content;
  const matrixStart = content.indexOf('## Matrix and Vector Notation');
  const quantumStart = content.indexOf('## Quantum Mechanics Notation');
  
  if (matrixStart === -1 || quantumStart === -1) {
    console.log('Could not find matrix section boundaries');
    return;
  }
  
  const matrixSection = content.substring(matrixStart, quantumStart).trim();
  
  console.log('=== Isolated Matrix Section ===');
  console.log(matrixSection);
  console.log('\n=== Length:', matrixSection.length, 'chars ===');
  
  // Parse just this section
  const tree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(matrixSection);
  
  // Find all math nodes
  function findMathNodes(node, path = '') {
    const results = [];
    
    if (node.type === 'math') {
      results.push({
        path,
        meta: node.meta,
        value: node.value,
        position: node.position
      });
    }
    
    if (node.children) {
      node.children.forEach((child, index) => {
        results.push(...findMathNodes(child, `${path}[${index}]`));
      });
    }
    
    return results;
  }
  
  const mathNodes = findMathNodes(tree);
  
  console.log(`\n=== Found ${mathNodes.length} math nodes ===`);
  mathNodes.forEach((node, index) => {
    console.log(`\nMath node ${index + 1}:`);
    console.log('Path:', node.path);
    console.log('Meta:', JSON.stringify(node.meta));
    console.log('Value (first 100 chars):', JSON.stringify(node.value.substring(0, 100)));
    if (node.position) {
      console.log('Position:', `lines ${node.position.start.line}-${node.position.end.line}`);
    }
  });
}

debugSpecificContent(); 