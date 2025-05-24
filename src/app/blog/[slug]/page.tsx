import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Clock } from "lucide-react";
import { getPostData, getAllPostSlugs } from "@/lib/posts";
import { formatDate } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ReadingProgress } from "@/components/ReadingProgress";
import { PostContent } from "@/components/PostContent";

interface BlogPostProps {
  params: Promise<{
    slug: string;
  }>;
}

export async function generateStaticParams() {
  const slugs = getAllPostSlugs();
  return slugs.map((slug) => ({
    slug,
  }));
}

export async function generateMetadata({ params }: BlogPostProps) {
  try {
    const { slug } = await params;
    const post = await getPostData(slug);
    return {
      title: `${post.title} | Phantasius`,
      description: post.excerpt,
    };
  } catch {
    return {
      title: "Post Not Found | Phantasius",
    };
  }
}

// Estimate reading time (average 200 words per minute)
function estimateReadingTime(content: string): number {
  const wordsPerMinute = 200;
  const words = content.replace(/<[^>]*>/g, '').split(/\s+/).length;
  return Math.ceil(words / wordsPerMinute);
}

export default async function BlogPost({ params }: BlogPostProps) {
  try {
    const { slug } = await params;
    const post = await getPostData(slug);
    const readingTime = estimateReadingTime(post.content);

    return (
      <>
        {/* Reading Progress Bar */}
        <ReadingProgress />
        
        <div className="scroll-container py-12">
          <div className="max-w-4xl mx-auto">
            {/* Back Button */}
            <div className="mb-12">
              <Link href="/">
                <Button variant="ghost" className="pl-0 hover:bg-sage/10 transition-colors duration-300">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Essays
                </Button>
              </Link>
            </div>

            {/* Article Header */}
            <header className="mb-16 text-center">
              <div className="space-y-6">
                <h1 className="text-5xl md:text-6xl font-serif font-bold leading-tight philosophical-heading">
                  {post.title}
                </h1>
                
                <div className="flex items-center justify-center space-x-6 text-muted-foreground">
                  <time dateTime={post.date} className="flex items-center space-x-2">
                    <span className="text-sage">📅</span>
                    <span className="font-medium">{formatDate(post.date)}</span>
                  </time>
                  
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-sage" />
                    <span>{readingTime} min read</span>
                  </div>
                </div>

                {post.tags && post.tags.length > 0 && (
                  <div className="flex justify-center">
                    <div className="flex flex-wrap gap-3">
                      {post.tags.map((tag) => (
                        <span 
                          key={tag} 
                          className="px-4 py-2 bg-gradient-to-r from-sage/10 to-gold/10 border border-sage/20 rounded-full text-sm font-medium text-sage hover:bg-gradient-to-r hover:from-sage/20 hover:to-gold/20 transition-all duration-300"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="max-w-3xl mx-auto">
                  <p className="text-xl text-muted-foreground leading-relaxed italic">
                    {post.excerpt}
                  </p>
                </div>
                
                {/* Decorative separator */}
                <div className="flex justify-center pt-6">
                  <div className="w-32 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
                  <div className="mx-4 text-sage text-xl">⚬</div>
                  <div className="w-32 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
                </div>
              </div>
            </header>

            {/* Article Content */}
            <article className="max-w-3xl mx-auto">
              <PostContent content={post.content} />
            </article>

            {/* Navigation Footer */}
            <footer className="mt-20 pt-12 border-t border-sage/20">
              <div className="flex flex-col sm:flex-row items-center justify-between space-y-4 sm:space-y-0">
                <Link href="/">
                  <Button 
                    variant="outline" 
                    className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 transition-all duration-300"
                  >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Return to Essays
                  </Button>
                </Link>
                
                <div className="text-center">
                  <p className="text-sm text-muted-foreground italic">
                    Continue your philosophical journey
                  </p>
                  <p className="greek-text text-lg mt-1">
                    φαντασία
                  </p>
                </div>
              </div>
            </footer>
          </div>
        </div>
      </>
    );
  } catch {
    notFound();
  }
} 