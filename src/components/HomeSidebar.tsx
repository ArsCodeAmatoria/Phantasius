import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatDate } from "@/lib/utils";
import { type PostMeta } from "@/lib/posts";

interface HomeSidebarProps {
  allPosts: PostMeta[];
  recentPosts: PostMeta[];
}

export function HomeSidebar({ allPosts, recentPosts }: HomeSidebarProps) {
  // Get popular tags (tags that appear most frequently)
  const tagCounts = allPosts.reduce((acc, post) => {
    post.tags?.forEach(tag => {
      acc[tag] = (acc[tag] || 0) + 1;
    });
    return acc;
  }, {} as Record<string, number>);

  const popularTags = Object.entries(tagCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6)
    .map(([tag]) => tag);

  // Categories with counts and visual elements
  const categories = [
    { 
      name: 'AGDEF Theory', 
      slug: 'agdef', 
      symbol: '∞',
      description: 'Higher-dimensional cosmology',
      count: allPosts.filter(p => p.tags?.includes('agdef')).length 
    },
    { 
      name: 'Ancient Philosophy', 
      slug: 'ancient-philosophy', 
      symbol: 'Φ',
      description: 'Classical wisdom traditions',
      count: allPosts.filter(p => p.tags?.includes('ancient-philosophy')).length 
    },
    { 
      name: 'Consciousness', 
      slug: 'consciousness', 
      symbol: 'Ψ',
      description: 'Mind and perception studies',
      count: allPosts.filter(p => p.tags?.includes('consciousness')).length 
    },
    { 
      name: 'Physics', 
      slug: 'physics', 
      symbol: 'Ω',
      description: 'Theoretical frameworks',
      count: allPosts.filter(p => p.tags?.includes('physics')).length 
    },
    { 
      name: 'Meditation', 
      slug: 'meditation', 
      symbol: '◉',
      description: 'Contemplative practices',
      count: allPosts.filter(p => p.tags?.includes('meditation')).length 
    },
  ].filter(cat => cat.count > 0);

  // Featured posts (highest tag count variety)
  const featuredPosts = allPosts
    .sort((a, b) => (b.tags?.length || 0) - (a.tags?.length || 0))
    .slice(0, 3);

  return (
    <div className="space-y-6">
      {/* Welcome Card */}
      <Card className="enhanced-card bg-gradient-to-br from-sage/12 to-gold/8 border-sage/30 overflow-hidden relative">
        <div className="absolute inset-0 opacity-30 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12"></div>
        </div>
        <CardContent className="pt-6 pb-6 text-center relative z-10">
          <div className="mb-4">
            <div className="w-12 h-12 mx-auto bg-gradient-to-br from-sage to-gold rounded-full flex items-center justify-center">
              <span className="text-2xl font-serif text-white">φ</span>
            </div>
          </div>
          <h3 className="font-serif font-semibold text-lg mb-2 text-sage">Phantasius</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The space where imagination meets philosophy
          </p>
        </CardContent>
      </Card>

      {/* Recent Essays */}
      <Card className="enhanced-card bg-gradient-to-br from-sage/8 to-gold/5 border-sage/25 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
        </div>
        <CardHeader className="relative z-10 pb-3">
          <CardTitle className="text-lg font-serif philosophical-heading flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-sage to-gold rounded flex items-center justify-center">
              <div className="w-3 h-3 bg-white rounded-sm"></div>
            </div>
            Recent Essays
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 relative z-10 pt-0">
          {recentPosts.slice(0, 4).map((post) => (
            <div key={post.slug} className="group/item border-b border-sage/15 last:border-0 pb-3 last:pb-0 hover:bg-sage/8 rounded-lg p-3 -m-3 transition-all duration-300">
              <Link href={`/blog/${post.slug}`}>
                <h4 className="font-medium text-sm leading-tight hover:text-gold transition-colors duration-300 line-clamp-2 mb-2">
                  {post.title}
                </h4>
              </Link>
              <div className="flex items-center justify-between">
                <time className="text-xs text-muted-foreground font-medium">
                  {formatDate(post.date)}
                </time>
                {post.tags && (
                  <span className="text-xs text-sage/80 opacity-60 group-hover/item:opacity-100 transition-opacity bg-sage/10 px-2 py-1 rounded">
                    {post.tags[0]}
                  </span>
                )}
              </div>
            </div>
          ))}
          <div className="pt-3">
            <Link href="/essays">
              <Button variant="outline" size="sm" className="w-full text-xs hover:bg-sage/10 hover:border-sage/40 transition-all duration-300">
                View All Essays ({allPosts.length})
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* Featured Essays */}
      <Card className="enhanced-card bg-gradient-to-br from-gold/8 to-sage/5 border-gold/25 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
        </div>
        <CardHeader className="relative z-10 pb-3">
          <CardTitle className="text-lg font-serif philosophical-heading flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-gold to-sage rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            Diverse Essays
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 relative z-10 pt-0">
          {featuredPosts.map((post) => (
            <Link key={post.slug} href={`/blog/${post.slug}`}>
              <div className="group/featured p-3 rounded-lg hover:bg-gradient-to-r hover:from-gold/10 hover:to-sage/10 transition-all duration-300 border border-transparent hover:border-gold/25">
                <h4 className="font-medium text-sm leading-tight group-hover/featured:text-gold transition-colors duration-300 line-clamp-2 mb-2">
                  {post.title}
                </h4>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    {post.tags?.length || 0} topics
                  </span>
                  <div className="flex gap-1">
                    {post.tags?.slice(0, 3).map((tag, i) => (
                      <div key={tag} className="w-2 h-2 rounded-full bg-gradient-to-r from-sage to-gold opacity-60 group-hover/featured:opacity-100 transition-opacity" 
                           style={{ animationDelay: `${i * 100}ms` }}></div>
                    ))}
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </CardContent>
      </Card>

      {/* Categories */}
      <Card className="enhanced-card bg-gradient-to-br from-sage/10 to-gold/6 border-sage/25 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
        </div>
        <CardHeader className="relative z-10 pb-3">
          <CardTitle className="text-lg font-serif philosophical-heading flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-sage to-gold rounded flex items-center justify-center">
              <div className="w-3 h-1 bg-white rounded-full"></div>
            </div>
            Categories
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 relative z-10 pt-0">
          {categories.map((category) => (
            <Link key={category.slug} href={`/essays?tag=${category.slug}`}>
              <div className="group/category flex items-center justify-between p-3 rounded-lg hover:bg-gradient-to-r hover:from-sage/12 hover:to-gold/12 transition-all duration-300 border border-transparent hover:border-sage/30">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-sage/20 to-gold/20 rounded-full flex items-center justify-center border border-sage/30 group-hover/category:scale-110 transition-transform duration-300">
                    <span className="text-sm font-serif text-sage group-hover/category:text-gold transition-colors duration-300">{category.symbol}</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium group-hover/category:text-sage transition-colors duration-300">{category.name}</div>
                    <div className="text-xs text-muted-foreground">{category.description}</div>
                  </div>
                </div>
                <span className="text-xs text-muted-foreground bg-sage/15 group-hover/category:bg-sage/25 px-2 py-1 rounded-full transition-all duration-300 min-w-[24px] text-center">
                  {category.count}
                </span>
              </div>
            </Link>
          ))}
        </CardContent>
      </Card>

      {/* Popular Tags */}
      <Card className="enhanced-card bg-gradient-to-br from-gold/10 to-sage/6 border-gold/25 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
        </div>
        <CardHeader className="relative z-10 pb-3">
          <CardTitle className="text-lg font-serif philosophical-heading flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-gold to-sage rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded"></div>
            </div>
            Popular Topics
          </CardTitle>
        </CardHeader>
        <CardContent className="relative z-10 pt-0">
          <div className="flex flex-wrap gap-2">
            {popularTags.map((tag, index) => (
              <Link key={tag} href={`/essays?tag=${tag}`}>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="text-xs border-sage/25 hover:bg-gradient-to-r hover:from-sage/15 hover:to-gold/15 hover:border-gold/40 transition-all duration-300 capitalize group"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <span className="group-hover:scale-105 transition-transform duration-300">
                    {tag.replace('-', ' ')}
                  </span>
                </Button>
              </Link>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Philosophy Quote */}
      <Card className="enhanced-card bg-gradient-to-br from-sage/12 to-gold/8 border-sage/30 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1200 ease-out"></div>
        </div>
        <CardContent className="pt-6 relative z-10">
          <div className="text-center">
            <div className="mb-4">
              <div className="w-10 h-10 mx-auto bg-gradient-to-br from-gold to-sage rounded-full flex items-center justify-center">
                <span className="text-lg text-white">§</span>
              </div>
            </div>
            <blockquote className="space-y-3">
              <p className="greek-text text-lg group-hover:text-gold transition-colors duration-300">
                &ldquo;φιλοσοφία βίου κυβερνήτης&rdquo;
              </p>
              <footer className="text-xs text-muted-foreground italic">
                Philosophy is the pilot of life — Cicero
              </footer>
            </blockquote>
          </div>
        </CardContent>
      </Card>

      {/* Stats Card */}
      <Card className="enhanced-card bg-gradient-to-br from-sage/6 to-gold/4 border-sage/20 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
        </div>
        <CardHeader className="relative z-10 pb-3">
          <CardTitle className="text-lg font-serif philosophical-heading flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-sage to-gold rounded flex items-center justify-center">
              <div className="w-2 h-3 bg-white rounded-sm"></div>
            </div>
            Library Stats
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0 relative z-10">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-sage/10 rounded-lg border border-sage/20">
              <div className="text-2xl font-bold text-sage">{allPosts.length}</div>
              <div className="text-xs text-muted-foreground">Essays</div>
            </div>
            <div className="text-center p-3 bg-gold/10 rounded-lg border border-gold/20">
              <div className="text-2xl font-bold text-gold">{Object.keys(tagCounts).length}</div>
              <div className="text-xs text-muted-foreground">Topics</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Connect Card */}
      <Card className="enhanced-card bg-gradient-to-br from-gold/12 to-sage/8 border-gold/30 overflow-hidden relative group">
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/8 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1200 ease-out"></div>
        </div>
        <CardContent className="pt-6 text-center relative z-10">
          <div className="mb-4">
            <div className="w-10 h-10 mx-auto bg-gradient-to-br from-gold to-sage rounded-full flex items-center justify-center">
              <span className="text-lg text-white">∴</span>
            </div>
          </div>
          <h4 className="font-serif font-semibold mb-2 text-gold">Stay Connected</h4>
          <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
            Join the philosophical journey through essays on consciousness, physics, and ancient wisdom.
          </p>
          <Link href="/about">
            <Button variant="outline" size="sm" className="w-full hover:bg-gradient-to-r hover:from-gold/15 hover:to-sage/15 hover:border-gold/40 transition-all duration-300 group">
              <span className="mr-2 group-hover:scale-110 transition-transform duration-300">→</span>
              Learn More
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
} 