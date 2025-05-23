import { getSortedPostsData } from "@/lib/posts";
import { PostCard } from "@/components/PostCard";

export default function Home() {
  const allPostsData = getSortedPostsData();

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-serif font-bold mb-6 tracking-tight">
            Phantasius
          </h1>
          <p className="text-xl text-muted-foreground mb-4 leading-relaxed">
            <span className="greek-text text-2xl">φαντασία</span> — the space of imagination
          </p>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Exploring the ancient wisdom of mental images, consciousness, and the theatre of the mind 
            through philosophical reflection and practical insight.
          </p>
        </div>

        {/* Blog Posts */}
        <div className="space-y-8">
          <h2 className="text-3xl font-serif font-semibold mb-8">Recent Essays</h2>
          <div className="grid gap-8">
            {allPostsData.map((post) => (
              <PostCard key={post.slug} post={post} />
            ))}
          </div>
        </div>

        {/* About Section */}
        <div className="mt-20 pt-12 border-t border-border">
          <div className="text-center">
            <h3 className="text-2xl font-serif font-semibold mb-4">About This Space</h3>
            <p className="text-muted-foreground leading-relaxed max-w-2xl mx-auto">
              In the tradition of ancient philosophy, this blog explores the continuing relevance 
              of classical insights into consciousness, imagination, and human experience. 
              Each essay is an invitation to philosophical reflection.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
