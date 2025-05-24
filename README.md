# Phantasius - φαντασία

> A minimalist philosophical blog exploring ancient wisdom and modern consciousness through the lens of phantasia

## About

**Phantasius** is a sophisticated blog platform built with Next.js that explores the ancient Greek concept of **φαντασία** (phantasia) and its relevance to modern understanding of imagination, consciousness, and human experience. The name "Phantasius" invokes the Platonic and Stoic idea of internal images, appearances, and the theatre of the mind—the space where mental images arise and play out their eternal dance.

This blog serves as a digital space for philosophical contemplation, featuring essays on consciousness, ancient wisdom, and the intersection of classical thought with contemporary insights. The design reflects the contemplative nature of philosophical inquiry with careful attention to typography, readability, and aesthetic harmony.

## Features

### Content & Writing Experience
- **Markdown-Powered Posts** - Write and publish essays using Markdown with frontmatter metadata
- **Code Syntax Highlighting** - Beautiful Dracula-themed code blocks with VS Code-style windows
- **Table Support** - GitHub Flavored Markdown with table formatting
- **Tag-Based Organization** - Categorize posts with tags for easy discovery
- **Reading Progress Indicator** - Visual progress tracking for long-form content

### Design & Typography
- **Philosophical Typography** - Premium serif fonts (Crimson Text, Playfair Display) optimized for reading
- **Greek Text Support** - Proper rendering of Greek characters with Noto Serif
- **Dark Theme Design** - Sophisticated dark color palette for focused reading
- **Responsive Layout** - Beautiful presentation across all device sizes
- **Scroll-Like Spacing** - Contemplative design inspired by ancient manuscripts

### Technical Features
- **Static Site Generation** - Fast, SEO-optimized pages generated at build time
- **Modern Stack** - Built with Next.js 15, Tailwind CSS, and TypeScript
- **Component Library** - shadcn/ui components for consistent design
- **Performance Optimized** - Fast loading times and smooth interactions
- **Search & Navigation** - Easy content discovery and navigation

## Tech Stack

- **Framework**: Next.js 15 with App Router and TypeScript
- **Styling**: Tailwind CSS with custom design system
- **UI Components**: shadcn/ui component library
- **Content Management**: Markdown files with gray-matter frontmatter parsing
- **Syntax Highlighting**: react-syntax-highlighter with Dracula theme
- **Typography**: Google Fonts (Crimson Text, Playfair Display, Noto Serif, Inter)
- **Icons**: Lucide React icon library
- **Build & Deploy**: Static site generation, ready for any hosting platform

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm, yarn, or pnpm package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArsCodeAmatoria/Phantasius.git
cd phantasius
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) to view the site.

### Building for Production

```bash
npm run build
npm start
```

## Writing Posts

Create new blog posts by adding Markdown files to the `posts/` directory. Each post should include frontmatter metadata:

```markdown
---
title: "The Nature of Phantasia in Stoic Philosophy"
date: "2025-01-24"
excerpt: "Exploring how the Stoics understood the role of mental impressions in shaping our reality"
tags: ["stoicism", "consciousness", "ancient-philosophy"]
---

# The Nature of Phantasia in Stoic Philosophy

Your essay content begins here. You can use all standard Markdown features:

## Headings and Structure

Regular paragraphs with **bold** and *italic* text.

> Beautiful blockquotes for highlighting important passages

### Code Examples

```python
def contemplation():
    return "wisdom through reflection"
```

### Tables

| Philosopher | School | Key Concept |
|-------------|--------|-------------|
| Aristotle   | Peripatetic | Active Sensation |
| Zeno        | Stoic | Kataleptic Impressions |
```

### Frontmatter Fields

- **title**: The post title (required)
- **date**: Publication date in YYYY-MM-DD format (required)
- **excerpt**: Brief description for post previews (required)
- **tags**: Array of tags for categorization (optional)

## Project Structure

```
phantasius/
├── posts/                     # Markdown blog posts
├── src/
│   ├── app/                  # Next.js app router
│   │   ├── blog/[slug]/      # Dynamic blog post pages
│   │   ├── about/            # About page
│   │   ├── globals.css       # Global styles and theme
│   │   └── page.tsx          # Homepage
│   ├── components/           # React components
│   │   ├── ui/              # shadcn/ui components
│   │   ├── CodeBlock.tsx    # Syntax highlighted code blocks
│   │   ├── PostContent.tsx  # Blog post content renderer
│   │   ├── Header.tsx       # Site navigation
│   │   └── Footer.tsx       # Site footer
│   └── lib/
│       └── posts.ts         # Content parsing utilities
├── public/                  # Static assets
└── tailwind.config.ts      # Tailwind configuration
```

## Customization

### Theme Colors

The design uses a custom dark theme with philosophical color palette:

- **Sage Green** (--sage): Wisdom and contemplative elements
- **Gold** (--gold): Enlightenment and accent highlights  
- **Deep Slate** (--background): Primary background
- **Light Gray** (--foreground): Main text color
- **Muted Tones**: Secondary text and UI elements

### Typography Hierarchy

- **Display Headings**: Playfair Display for impact
- **Body Text**: Crimson Text for extended reading
- **Greek Text**: Noto Serif for proper Greek character rendering
- **Code**: JetBrains Mono and Consolas for code blocks
- **UI Elements**: Inter for navigation and interface

### Code Blocks

Code blocks feature a VS Code-inspired design with:
- Dark gray background for better contrast
- Dracula syntax highlighting theme
- Line numbers for reference
- Copy-to-clipboard functionality
- Proper file extension mapping
- Window chrome with traffic light controls

## Deployment

### Vercel (Recommended)

1. Push your repository to GitHub
2. Connect to Vercel and deploy automatically
3. Enjoy automatic deployments on every push

### Other Platforms

The blog generates static HTML and can be deployed anywhere:

- **Netlify**: Connect GitHub repo for automatic deployments
- **GitHub Pages**: Use GitHub Actions for static site hosting
- **Cloudflare Pages**: Fast global distribution
- **Traditional Hosting**: Upload the `out/` directory after `npm run build`

### Environment Setup

No environment variables required for basic functionality. The blog works out of the box with static generation.

## Philosophy & Design

This blog embodies the ancient Greek understanding of phantasia—the faculty by which appearances manifest in consciousness. The design philosophy emphasizes:

**Contemplative Experience**: Clean, distraction-free reading environment that encourages deep engagement with ideas.

**Typographic Excellence**: Premium fonts and careful spacing create a reading experience worthy of philosophical content.

**Classical Inspiration**: Visual elements inspired by ancient manuscripts and scrolls, including subtle background textures and section dividers.

**Modern Accessibility**: While honoring classical aesthetics, the site maintains modern usability standards and responsive design.

**Intellectual Rigor**: Code highlighting and table support enable technical discussions alongside philosophical essays.

## Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or suggesting new features, please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the existing style and includes appropriate documentation.

## License

MIT License - You're free to use this project as a foundation for your own philosophical blog or educational website.

---

*"ἡ φαντασία ἐστι κίνησις ὑπὸ τῆς κατὰ ἐνέργειαν αἰσθήσεως γιγνομένη"*

*"Phantasia is a movement produced by an active sensation" — Aristotle, De Anima*
