# Phantasius - φαντασία

> *A minimalist Markdown blog exploring ancient wisdom and modern consciousness through the lens of phantasia*

## About

**Phantasius** is a beautiful, minimalist blog built with Next.js that explores the ancient Greek concept of **φαντασία** (phantasia) and its relevance to modern understanding of imagination, consciousness, and human experience.

The name "Phantasius" invokes the Platonic and Stoic idea of internal images, appearances, and the theatre of the mind—the space where mental images arise and play out their eternal dance.

## Features

- **Beautiful Typography** - Serif fonts (Crimson Text, Playfair Display) with Greek-inspired styling
- **Dark Mode Design** - Sophisticated dark theme optimized for reading
- **Markdown-Powered** - Write posts in Markdown with frontmatter
- **Static Generation** - Fast, SEO-friendly static site generation
- **Responsive Design** - Beautiful on all devices
- **Philosophical Theme** - Greek typography and scroll-like spacing
- **Modern Stack** - Next.js 15, Tailwind CSS, shadcn/ui
- **Reading Progress** - Visual progress indicator for long-form content
- **Interactive Elements** - Smooth animations and hover effects

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS + shadcn/ui components
- **Content**: Markdown with gray-matter frontmatter parsing
- **Typography**: Google Fonts (Crimson Text, Playfair Display, Noto Serif, Inter)
- **Icons**: Lucide React
- **Theme**: Custom dark mode design
- **Deployment**: Ready for Vercel, Netlify, or any static host

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd phantasius
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Writing Posts

Create new blog posts by adding Markdown files to the `posts/` directory:

```markdown
---
title: "Your Post Title"
date: "2024-01-15"
excerpt: "A brief description of your post"
tags: ["philosophy", "consciousness"]
---

# Your Post Content

Write your post content here in Markdown...
```

### Frontmatter Fields

- `title`: The post title
- `date`: Publication date (YYYY-MM-DD format)
- `excerpt`: Brief description for the post card
- `tags`: Array of tags for categorization

## Project Structure

```
phantasius/
├── posts/                  # Markdown blog posts
├── src/
│   ├── app/               # Next.js app router pages
│   │   ├── blog/[slug]/   # Dynamic blog post pages
│   │   ├── about/         # About page
│   │   └── page.tsx       # Home page
│   ├── components/        # React components
│   │   ├── ui/           # shadcn/ui components
│   │   ├── Header.tsx    # Site header
│   │   ├── Footer.tsx    # Site footer
│   │   └── PostCard.tsx  # Blog post preview card
│   └── lib/
│       └── posts.ts      # Post parsing utilities
├── public/               # Static assets
└── tailwind.config.ts   # Tailwind configuration
```

## Customization

### Typography

The blog uses beautiful serif typography optimized for reading:

- **Primary Font**: Crimson Text (serif)
- **Display Font**: Playfair Display (serif)
- **Greek Text**: Noto Serif 
- **UI Elements**: Inter (sans-serif)

### Colors & Theme

Built on shadcn/ui's design system with custom CSS variables for philosophical theming. The dark color palette includes:

- **Sage green** for wisdom and emphasis
- **Gold** for enlightenment and accents
- **Deep slate** for backgrounds
- **Warm grays** for text and subtle elements

### Adding Components

Add new shadcn/ui components:

```bash
npx shadcn@latest add [component-name]
```

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy automatically

### Other Platforms

The blog generates static files and can be deployed to:
- Netlify
- GitHub Pages
- Any static hosting service

Build the static site:
```bash
npm run build
```

## Philosophy

This blog embodies the ancient Greek understanding of phantasia—the space where appearances manifest in the mind. The design reflects this with:

- Scroll-like spacing and typography
- Contemplative reading experience  
- Greek typographic elements
- Minimalist, distraction-free interface
- Dark theme for focused contemplation

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - feel free to use this project as a foundation for your own philosophical blog.

---

*"ἡ φαντασία ἐστι κίνησις ὑπὸ τῆς κατὰ ἐνέργειαν αἰσθήσεως γιγνομένη"*

*"Phantasia is a movement produced by an active sensation" — Aristotle*
