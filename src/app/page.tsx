import { getSortedPostsData } from "@/lib/posts";
import { PostCard } from "@/components/PostCard";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  const allPostsData = getSortedPostsData();
  const recentPosts = allPostsData.slice(0, 4); // Show only 4 most recent posts

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
                  Exploring the ancient wisdom of mental images, consciousness, and the theatre of the mind 
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

        {/* Recent Essays */}
        <div className="scroll-spacing">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-semibold philosophical-heading mb-4">
              Recent Essays
            </h2>
            <p className="text-muted-foreground italic">
              Latest invitations to philosophical reflection
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

          {/* View All Essays Button */}
          <div className="text-center mt-12">
            <Link href="/essays">
              <Button 
                variant="outline" 
                size="lg"
                className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300 text-lg px-8 py-3"
              >
                View All Essays ({allPostsData.length})
              </Button>
            </Link>
          </div>
        </div>

        {/* About Section */}
        <div className="mt-24 pt-16 border-t border-sage/20">
          <div className="text-center max-w-3xl mx-auto scroll-spacing">
            <h3 className="text-3xl font-serif font-semibold philosophical-heading mb-8">
              About This Space
            </h3>
            <div className="philosophical-callout">
              <p className="text-lg leading-relaxed mb-6">
                In the tradition of ancient philosophy, this blog explores the continuing relevance 
                of classical insights into consciousness, imagination, and human experience.
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
                  'consciousness', 'ancient-wisdom', 'philosophy', 
                  'attention', 'beauty', 'dreams'
                ].map((tag) => (
                  <Link key={tag} href={`/essays?tag=${tag}`}>
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-sage/20 hover:bg-sage/10 hover:border-sage/40 transition-all duration-300 capitalize"
                    >
                      {tag}
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
