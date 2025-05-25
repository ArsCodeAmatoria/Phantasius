import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

async function debugAST() {
  // Read the actual test-katex.md file
  const fullPath = path.join(process.cwd(), 'posts', 'test-katex.md');
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const matterResult = matter(fileContents);
  
  // Focus on just the matrix section
  const matrixSection = `## Matrix and Vector Notation

A simple 2×2 matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

The identity matrix:

$$I = \\begin{pmatrix}
1 & 0 & 0 \\\\
0 & 1 & 0 \\\\
0 & 0 & 1
\\end{pmatrix}$$`;

  console.log('=== Matrix section input ===');
  console.log(matrixSection);
  
  // Parse with remark to see the AST
  const tree = unified()
    .use(remarkParse)
    .use(remarkMath)
    .parse(matrixSection);
  
  console.log('\\n=== AST (Abstract Syntax Tree) ===');
  console.log(JSON.stringify(tree, null, 2));
  
  // Look specifically for math nodes
  function findMathNodes(node, path = '') {
    if (node.type === 'math') {
      console.log(`\\n=== Math node at ${path} ===`);
      console.log('Value:', JSON.stringify(node.value));
      console.log('Meta:', JSON.stringify(node.meta));
    }
    
    if (node.children) {
      node.children.forEach((child, index) => {
        findMathNodes(child, `${path}[${index}]`);
      });
    }
  }
  
  console.log('\\n=== Math nodes found ===');
  findMathNodes(tree);
}

debugAST(); 