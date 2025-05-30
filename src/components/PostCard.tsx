import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatDate } from "@/lib/utils";
import { type PostMeta } from "@/lib/posts";

interface PostCardProps {
  post: PostMeta;
}

export function PostCard({ post }: PostCardProps) {
  return (
    <Card className="enhanced-card smooth-transition group bg-gradient-to-br from-sage/6 to-gold/4 border-sage/20 overflow-hidden relative">
      {/* Shine effect overlay */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/8 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1200 ease-out"></div>
      </div>
      
      <CardHeader className="pb-4 relative z-10">
        <div className="space-y-3">
          <Link href={`/blog/${post.slug}`}>
            <h2 className="philosophical-heading text-2xl font-serif font-semibold leading-tight group-hover:text-white transition-colors duration-300">
              {post.title}
            </h2>
          </Link>
          <div className="flex items-center space-x-3 text-sm text-muted-foreground">
            <time 
              dateTime={post.date}
              className="font-medium"
            >
              {formatDate(post.date)}
            </time>
            {post.tags && post.tags.length > 0 && (
              <>
                <span className="text-sage">•</span>
                <div className="flex flex-wrap gap-2">
                  {post.tags.map((tag) => (
                    <span 
                      key={tag} 
                      className="px-3 py-1 bg-gradient-to-r from-sage/15 to-gold/12 border border-sage/25 rounded-full text-xs font-medium text-sage hover:from-sage/25 hover:to-gold/20 hover:bg-gradient-to-r hover:text-white transition-all duration-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 relative z-10">
        <p className="text-muted-foreground leading-relaxed text-justify group-hover:text-white/90 transition-colors duration-300">
          {post.excerpt}
        </p>
        <div className="pt-2">
          <Link 
            href={`/blog/${post.slug}`}
            className="inline-flex items-center text-sm font-medium text-sage hover:text-white transition-colors duration-300 group"
          >
            <span>Continue reading</span>
            <svg 
              className="ml-2 h-4 w-4 transform group-hover:translate-x-1 transition-transform duration-300" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
} 