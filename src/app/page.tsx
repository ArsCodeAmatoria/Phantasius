import { getSortedPostsData } from "@/lib/posts";
import { PostCard } from "@/components/PostCard";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  const allPostsData = getSortedPostsData();
  
  // Categorize posts by content themes
  const agdefPosts = allPostsData.filter(post => 
    post.tags.includes('agdef') || 
    post.title.toLowerCase().includes('agdef') ||
    post.title.toLowerCase().includes('anti-gravity')
  );
  
  const ancientPhysicsPosts = allPostsData.filter(post => 
    (post.tags.includes('physics') && post.tags.includes('ancient-philosophy')) ||
    post.tags.includes('quantum-philosophy') ||
    post.tags.includes('stoic-physics') ||
    post.title.toLowerCase().includes('platonic') ||
    post.title.toLowerCase().includes('stoic') && post.tags.includes('physics')
  );
  
  const philosophyOfMindPosts = allPostsData.filter(post => 
    post.tags.includes('consciousness') ||
    post.tags.includes('dreams') ||
    post.tags.includes('attention') ||
    post.tags.includes('memory') ||
    post.tags.includes('phantasia') ||
    post.title.toLowerCase().includes('consciousness') ||
    post.title.toLowerCase().includes('mind') ||
    post.title.toLowerCase().includes('dreams')
  );
  
  const practicalPhilosophyPosts = allPostsData.filter(post => 
    post.tags.includes('contemplative-practice') ||
    post.tags.includes('digital-minimalism') ||
    post.tags.includes('meditation') ||
    post.tags.includes('practical-wisdom') ||
    post.title.toLowerCase().includes('technology') ||
    post.title.toLowerCase().includes('meditation')
  );
  
  // Get remaining posts that don't fit the above categories
  const categorizedSlugs = [
    ...agdefPosts.map(p => p.slug),
    ...ancientPhysicsPosts.map(p => p.slug),
    ...philosophyOfMindPosts.map(p => p.slug),
    ...practicalPhilosophyPosts.map(p => p.slug)
  ];
  
  const otherPosts = allPostsData.filter(post => 
    !categorizedSlugs.includes(post.slug)
  );

  // Recent posts for hero section (top 3 most recent)
  const recentPosts = allPostsData.slice(0, 3);

  const PostSection = ({ title, description, posts, maxPosts = 3, sectionId }: {
    title: string;
    description: string;
    posts: typeof allPostsData;
    maxPosts?: number;
    sectionId: string;
  }) => {
    if (posts.length === 0) return null;
    
    const displayPosts = posts.slice(0, maxPosts);
    
    return (
      <div className="scroll-spacing mb-16">
        <div className="text-center mb-8">
          <h3 className="text-3xl font-serif font-semibold philosophical-heading mb-3">
            {title}
          </h3>
          <p className="text-muted-foreground italic text-lg">
            {description}
          </p>
        </div>
        
        <div className="grid gap-8 max-w-4xl mx-auto">
          {displayPosts.map((post, index) => (
            <div 
              key={post.slug}
              className="animate-fade-in-up"
              style={{ 
                animationDelay: `${index * 100}ms`,
                opacity: 0,
                animation: `fadeInUp 0.6s ease-out ${index * 100}ms forwards`
              }}
            >
              <PostCard post={post} />
            </div>
          ))}
        </div>
        
        {posts.length > maxPosts && (
          <div className="text-center mt-8">
            <Link href={`/essays#${sectionId}`}>
              <Button 
                variant="outline" 
                size="sm"
                className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300"
              >
                View All {title} ({posts.length})
              </Button>
            </Link>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="scroll-container py-16">
      <div className="max-w-5xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-20 scroll-spacing">
          <div className="space-y-8">
            <h1 className="text-7xl font-serif font-bold mb-8 tracking-tight philosophical-heading">
              Phantasius
            </h1>
            <div className="space-y-4">
              <p className="text-2xl text-muted-foreground leading-relaxed">
                <span className="greek-text text-4xl font-medium hover:scale-110 inline-block transition-transform duration-300">φαντασία</span>
                <span className="mx-4 text-sage">—</span>
                <span className="italic">the space of imagination</span>
              </p>
              <div className="max-w-3xl mx-auto">
                <p className="text-xl text-muted-foreground leading-relaxed">
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

        {/* Latest Essays */}
        <div className="scroll-spacing mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-semibold philosophical-heading mb-4">
              Latest Essays
            </h2>
            <p className="text-muted-foreground italic">
              Recent invitations to philosophical reflection
            </p>
          </div>
          
          <div className="grid gap-10 max-w-4xl mx-auto">
            {recentPosts.map((post, index) => (
              <div 
                key={post.slug}
                className="animate-fade-in-up"
                style={{ 
                  animationDelay: `${index * 150}ms`,
                  opacity: 0,
                  animation: `fadeInUp 0.8s ease-out ${index * 150}ms forwards`
                }}
              >
                <PostCard post={post} />
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-sage/20 mb-16"></div>

        {/* Essay Collections */}
        <div className="space-y-20">
          <PostSection
            title="AGDEF Theory"
            description="Anti-Gravity Dark Energy Field theory and higher-dimensional cosmology"
            posts={agdefPosts}
            maxPosts={3}
            sectionId="agdef"
          />
          
          <PostSection
            title="Ancient Philosophy & Physics"
            description="Classical wisdom meets modern theoretical physics"
            posts={ancientPhysicsPosts}
            maxPosts={3}
            sectionId="ancient-physics"
          />
          
          <PostSection
            title="Philosophy of Mind"
            description="Consciousness, perception, and the nature of experience"
            posts={philosophyOfMindPosts}
            maxPosts={3}
            sectionId="mind"
          />
          
          <PostSection
            title="Practical Philosophy"
            description="Contemplative practices and wisdom for modern life"
            posts={practicalPhilosophyPosts}
            maxPosts={3}
            sectionId="practical"
          />
          
          {otherPosts.length > 0 && (
            <PostSection
              title="Additional Essays"
              description="Further explorations in philosophy and reflection"
              posts={otherPosts}
              maxPosts={3}
              sectionId="other"
            />
          )}
        </div>

        {/* View All Essays Button */}
        <div className="text-center mt-16">
          <Link href="/essays">
            <Button 
              variant="outline" 
              size="lg"
              className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300 text-lg px-8 py-3"
            >
              Browse All Essays ({allPostsData.length})
            </Button>
          </Link>
        </div>

        {/* About Section */}
        <div className="mt-24 pt-16 border-t border-sage/20">
          <div className="text-center max-w-3xl mx-auto scroll-spacing">
            <h3 className="text-3xl font-serif font-semibold philosophical-heading mb-8">
              About This Space
            </h3>
            <div className="philosophical-callout">
              <p className="text-lg leading-relaxed mb-6">
                In the tradition of ancient philosophy, this blog explores theoretical physics, 
                consciousness, and the continuing relevance of classical insights into human experience.
                From cutting-edge cosmological theories to contemplative practices, each essay bridges 
                the abstract and the practical.
              </p>
              <p className="text-muted-foreground italic">
                Each essay is an invitation to philosophical reflection and deeper understanding.
              </p>
            </div>
            
            {/* Explore Topics */}
            <div className="mt-12">
              <h4 className="text-xl font-serif font-medium mb-6 text-sage">Explore by Topic</h4>
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
            <div className="mt-12 pt-8 border-t border-gold/20">
              <blockquote className="text-center">
                <p className="greek-text text-xl mb-3">
                  &ldquo;γνῶθι σεαυτόν&rdquo;
                </p>
                <footer className="text-sm text-muted-foreground italic">
                  Know thyself — Inscribed at the Temple of Apollo at Delphi
                </footer>
              </blockquote>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
