import { getSortedPostsData } from "@/lib/posts";
import { PostCard } from "@/components/PostCard";
import { CompactPostCard } from "@/components/CompactPostCard";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  const allPostsData = getSortedPostsData();
  
  // Categorize posts by content themes - with null-safe checks
  const agdefPosts = allPostsData.filter(post => 
    post.tags?.includes('agdef') || 
    post.title?.toLowerCase()?.includes('agdef') ||
    post.title?.toLowerCase()?.includes('anti-gravity')
  );
  
  const ancientPhysicsPosts = allPostsData.filter(post => 
    (post.tags?.includes('physics') && post.tags?.includes('ancient-philosophy')) ||
    post.tags?.includes('quantum-philosophy') ||
    post.tags?.includes('stoic-physics') ||
    post.title?.toLowerCase()?.includes('platonic') ||
    (post.title?.toLowerCase()?.includes('stoic') && post.tags?.includes('physics'))
  );
  
  const philosophyOfMindPosts = allPostsData.filter(post => 
    post.tags?.includes('consciousness') ||
    post.tags?.includes('dreams') ||
    post.tags?.includes('attention') ||
    post.tags?.includes('memory') ||
    post.tags?.includes('phantasia') ||
    post.title?.toLowerCase()?.includes('consciousness') ||
    post.title?.toLowerCase()?.includes('mind') ||
    post.title?.toLowerCase()?.includes('dreams')
  );
  
  const practicalPhilosophyPosts = allPostsData.filter(post => 
    post.tags?.includes('contemplative-practice') ||
    post.tags?.includes('digital-minimalism') ||
    post.tags?.includes('meditation') ||
    post.tags?.includes('practical-wisdom') ||
    post.title?.toLowerCase()?.includes('technology') ||
    post.title?.toLowerCase()?.includes('meditation')
  );

  // Recent posts for hero section (top 3 most recent)
  const recentPosts = allPostsData.slice(0, 3);
  
  // Featured posts for the grid - limited to 2 rows (6 posts maximum)
  const featuredPosts = [
    ...recentPosts.slice(0, 2),
    ...agdefPosts.slice(0, 2),
    ...philosophyOfMindPosts.slice(0, 1),
    ...ancientPhysicsPosts.slice(0, 1),
  ].filter((post, index, self) => 
    index === self.findIndex(p => p.slug === post.slug)
  ).slice(0, 6); // Limit to 6 posts for 2 rows

  return (
    <div className="scroll-container py-16">
      <div className="max-w-6xl mx-auto px-4 lg:px-6">
        {/* Hero Section - Full Width */}
        <div className="text-center mb-24 scroll-spacing">
          <div className="space-y-8">
            <h1 className="text-6xl lg:text-7xl font-serif font-bold mb-8 tracking-tight philosophical-heading">
              Phantasius
            </h1>
            <div className="space-y-6">
              <p className="text-xl lg:text-2xl text-muted-foreground leading-relaxed">
                <span className="greek-text text-3xl lg:text-4xl font-medium hover:scale-110 inline-block transition-transform duration-300">φαντασία</span>
                <span className="mx-4 text-sage">—</span>
                <span className="italic">the space of imagination</span>
              </p>
              <div className="max-w-4xl mx-auto">
                <p className="text-lg lg:text-xl text-muted-foreground leading-relaxed">
                  Exploring theoretical physics, ancient wisdom, consciousness, and the theatre of the mind 
                  through philosophical reflection and practical insight.
                </p>
              </div>
            </div>
            
            {/* Decorative element */}
            <div className="flex justify-center pt-8">
              <div className="w-24 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
              <div className="mx-4 text-sage text-lg">⚬</div>
              <div className="w-24 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
            </div>
          </div>
        </div>

        {/* Main Content - Full Width */}
        <div className="space-y-20">
          {/* Featured Essays Grid - 2 Rows Maximum */}
          <section className="scroll-spacing">
            <div className="text-center mb-16">
              <h2 className="text-3xl lg:text-4xl font-serif font-semibold philosophical-heading mb-6">
                Featured Essays
              </h2>
              <p className="text-muted-foreground italic text-lg max-w-2xl mx-auto">
                A curated selection of recent essays exploring consciousness, physics, and philosophical reflection
              </p>
            </div>
            
            {/* Grid of posts - 2 rows maximum, 3 columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 max-w-5xl mx-auto">
              {featuredPosts.map((post, index) => (
                <div 
                  key={post.slug}
                  className="animate-fade-in-up"
                  style={{ 
                    animationDelay: `${index * 150}ms`,
                    opacity: 0,
                    animation: `fadeInUp 0.6s ease-out ${index * 150}ms forwards`
                  }}
                >
                  <CompactPostCard post={post} />
                </div>
              ))}
            </div>
            
            {/* View More Featured */}
            <div className="text-center mt-12">
              <Link href="/essays">
                <Button 
                  variant="outline" 
                  size="default"
                  className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300 px-8"
                >
                  Explore All Essays
                </Button>
              </Link>
            </div>
          </section>

          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-sage/20"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="bg-background px-6 text-muted-foreground">Recent Publications</span>
            </div>
          </div>

          {/* Latest Full Posts */}
          <section className="scroll-spacing">
            <div className="text-center mb-16">
              <h3 className="text-2xl lg:text-3xl font-serif font-semibold philosophical-heading mb-4">
                Latest Essays
              </h3>
              <p className="text-muted-foreground italic max-w-xl mx-auto">
                In-depth explorations of philosophical questions and theoretical frameworks
              </p>
            </div>
            
            <div className="space-y-12 max-w-4xl mx-auto">
              {recentPosts.map((post, index) => (
                <article 
                  key={post.slug}
                  className="animate-fade-in-up"
                  style={{ 
                    animationDelay: `${index * 200}ms`,
                    opacity: 0,
                    animation: `fadeInUp 0.8s ease-out ${index * 200}ms forwards`
                  }}
                >
                  <PostCard post={post} />
                </article>
              ))}
            </div>
          </section>

          {/* Categories Preview */}
          <section className="scroll-spacing">
            <div className="text-center mb-16">
              <h3 className="text-2xl lg:text-3xl font-serif font-semibold philosophical-heading mb-4">
                Explore by Theme
              </h3>
              <p className="text-muted-foreground italic max-w-xl mx-auto">
                Navigate through interconnected topics of philosophical inquiry
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-4xl mx-auto">
              {[
                { name: 'AGDEF Theory', slug: 'agdef', count: agdefPosts.length, symbol: '∞' },
                { name: 'Consciousness', slug: 'consciousness', count: philosophyOfMindPosts.length, symbol: 'Ψ' },
                { name: 'Ancient Philosophy', slug: 'ancient-philosophy', count: ancientPhysicsPosts.length, symbol: 'Φ' },
                { name: 'Physics', slug: 'physics', count: allPostsData.filter(p => p.tags?.includes('physics')).length, symbol: 'Ω' }
              ].filter(cat => cat.count > 0).map((category, index) => (
                <Link key={category.slug} href={`/essays?tag=${category.slug}`}>
                  <div 
                    className="enhanced-card bg-gradient-to-br from-sage/8 to-gold/5 border-sage/20 p-6 text-center group cursor-pointer h-full"
                    style={{ 
                      animationDelay: `${index * 100}ms`,
                      opacity: 0,
                      animation: `fadeInUp 0.6s ease-out ${index * 100}ms forwards`
                    }}
                  >
                    <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-br from-sage/20 to-gold/20 rounded-full flex items-center justify-center border border-sage/30 group-hover:scale-110 transition-transform duration-300">
                      <span className="text-xl font-serif text-sage group-hover:text-gold transition-colors duration-300">{category.symbol}</span>
                    </div>
                    <h4 className="font-serif font-semibold mb-2 group-hover:text-sage transition-colors duration-300">{category.name}</h4>
                    <p className="text-sm text-muted-foreground">{category.count} essays</p>
                  </div>
                </Link>
              ))}
            </div>
          </section>

          {/* Browse All Button */}
          <div className="text-center pt-8">
            <Link href="/essays">
              <Button 
                variant="outline" 
                size="lg"
                className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300 text-lg px-12 py-4"
              >
                Browse Complete Library ({allPostsData.length} Essays)
              </Button>
            </Link>
          </div>
        </div>

        {/* About Section - Full Width */}
        <section className="mt-32 pt-20 border-t border-sage/20">
          <div className="text-center max-w-4xl mx-auto scroll-spacing">
            <h3 className="text-2xl lg:text-3xl font-serif font-semibold philosophical-heading mb-8">
              About This Space
            </h3>
            <div className="philosophical-callout">
              <p className="text-lg leading-relaxed mb-6">
                In the tradition of ancient philosophy, this digital space explores theoretical physics, 
                consciousness, and the continuing relevance of classical insights into human experience.
                From cutting-edge cosmological theories to contemplative practices, each essay bridges 
                the abstract and the practical.
              </p>
              <p className="text-muted-foreground italic">
                Each essay is an invitation to philosophical reflection and deeper understanding.
              </p>
            </div>
            
            {/* Explore Topics */}
            <div className="mt-16">
              <h4 className="text-xl font-serif font-medium mb-8 text-sage">Popular Topics</h4>
              <div className="flex flex-wrap justify-center gap-3">
                {[
                  'agdef', 'consciousness', 'ancient-philosophy', 'physics',
                  'attention', 'beauty', 'dreams', 'meditation'
                ].map((tag) => (
                  <Link key={tag} href={`/essays?tag=${tag}`}>
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-sage/20 hover:bg-sage/10 hover:border-sage/40 transition-all duration-300 capitalize"
                    >
                      {tag.replace('-', ' ')}
                    </Button>
                  </Link>
                ))}
              </div>
            </div>
            
            {/* Ancient Greek quote */}
            <div className="mt-16 pt-12 border-t border-gold/20">
              <blockquote className="text-center">
                <p className="greek-text text-xl mb-4">
                  &ldquo;γνῶθι σεαυτόν&rdquo;
                </p>
                <footer className="text-sm text-muted-foreground italic">
                  Know thyself — Inscribed at the Temple of Apollo at Delphi
                </footer>
              </blockquote>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
