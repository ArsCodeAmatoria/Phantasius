import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { getPostData, getAllPostSlugs, formatDate } from "@/lib/posts";
import { Button } from "@/components/ui/button";

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

export default async function BlogPost({ params }: BlogPostProps) {
  try {
    const { slug } = await params;
    const post = await getPostData(slug);

    return (
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto">
          {/* Back Button */}
          <div className="mb-8">
            <Link href="/">
              <Button variant="ghost" className="pl-0">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Essays
              </Button>
            </Link>
          </div>

          {/* Article Header */}
          <header className="mb-12">
            <h1 className="text-4xl font-serif font-bold leading-tight mb-6">
              {post.title}
            </h1>
            
            <div className="flex items-center space-x-4 text-muted-foreground mb-6">
              <time dateTime={post.date} className="text-sm">
                {formatDate(post.date)}
              </time>
              {post.tags && post.tags.length > 0 && (
                <>
                  <span>•</span>
                  <div className="flex space-x-2">
                    {post.tags.map((tag) => (
                      <span 
                        key={tag} 
                        className="px-2 py-1 bg-muted rounded-md text-xs font-medium"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>

            <p className="text-xl text-muted-foreground leading-relaxed">
              {post.excerpt}
            </p>
          </header>

          {/* Article Content */}
          <article 
            className="prose prose-lg max-w-none"
            dangerouslySetInnerHTML={{ __html: post.content }}
          />

          {/* Navigation */}
          <div className="mt-16 pt-8 border-t border-border">
            <Link href="/">
              <Button variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Return to Essays
              </Button>
            </Link>
          </div>
        </div>
      </div>
    );
  } catch {
    notFound();
  }
} 