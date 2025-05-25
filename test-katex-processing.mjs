import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

async function testKatex() {
  let markdown = `# Testing KaTeX Math Expressions

Let's test both inline and display math expressions.

## Display Math

The quadratic formula:

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

## Matrix and Vector Notation

A simple matrix:

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
`;

  // Preprocess: Temporarily replace & with a placeholder in math expressions
  console.log('=== Original markdown ===');
  console.log(markdown.substring(0, 500));
  
  // Replace & with placeholder in display math ($$...$$) - CORRECTED REGEX
  markdown = markdown.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
    return '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  });
  
  // Replace & with placeholder in inline math ($...$) - CORRECTED REGEX
  markdown = markdown.replace(/\$([^$\n]+?)\$/g, (match, mathContent) => {
    return '$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$';
  });

  console.log('\n=== Preprocessed markdown ===');
  console.log(markdown.substring(0, 500));

  try {
    const result = await unified()
      .use(remarkParse) // Parse markdown
      .use(remarkGfm) // GitHub Flavored Markdown for tables
      .use(remarkMath) // Parse math expressions
      .use(remarkRehype, { 
        allowDangerousHtml: true,
        entities: {
          useShortestReferences: true,
          useNamedReferences: true
        }
      }) // Convert to HTML with better entity handling
      .use(rehypeKatex, {
        strict: false,
        trust: true
      }) // Render math with KaTeX (less strict)
      .use(rehypeStringify, { 
        allowDangerousHtml: true,
        entities: {
          useShortestReferences: true,
          useNamedReferences: true
        }
      }) // Convert to string
      .process(markdown);
    
    let html = result.toString();
    
    // Post-process: Restore & characters in the final HTML
    html = html.replace(/XAMPERSANDX/g, '&');
    
    console.log('\n=== Final HTML (first 1000 chars) ===');
    console.log(html.substring(0, 1000));
    
    // Check if KaTeX classes are present
    if (html.includes('class="katex"')) {
      console.log('\n✅ KaTeX rendering detected!');
    } else {
      console.log('\n❌ No KaTeX rendering found');
    }
    
    // Check for errors
    if (html.includes('katex-error')) {
      console.log('\n❌ KaTeX errors detected!');
      const errorMatches = html.match(/class="katex-error"[^>]*title="([^"]*)"/g) || [];
      errorMatches.forEach((error, index) => {
        const titleMatch = error.match(/title="([^"]*)"/);
        if (titleMatch) {
          console.log(`   Error ${index + 1}: ${titleMatch[1].substring(0, 100)}...`);
        }
      });
    } else {
      console.log('\n✅ No KaTeX errors detected!');
    }
    
  } catch (error) {
    console.error('Error processing KaTeX:', error);
  }
}

testKatex(); 