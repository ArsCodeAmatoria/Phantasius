const testContent = `A simple matrix:

$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}$$

End of content.`;

console.log('Original content:');
console.log(testContent);

console.log('\\nTesting correct regex match:');
const regex = /\$\$([\s\S]*?)\$\$/g;
const matches = testContent.match(regex);
console.log('Matches found:', matches);

if (matches) {
  matches.forEach((match, index) => {
    console.log(`Match ${index}:`, JSON.stringify(match));
  });
}

console.log('\\nTesting replacement:');
const replaced = testContent.replace(regex, (match, mathContent) => {
  console.log('Replacing match:', JSON.stringify(match));
  console.log('Math content:', JSON.stringify(mathContent));
  const newContent = '$$' + mathContent.replace(/&/g, 'XAMPERSANDX') + '$$';
  console.log('New content:', JSON.stringify(newContent));
  return newContent;
});

console.log('\\nResult:');
console.log(replaced); 