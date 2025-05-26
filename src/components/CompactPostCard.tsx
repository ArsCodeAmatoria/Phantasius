import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatDate } from "@/lib/utils";
import { type PostMeta } from "@/lib/posts";

interface CompactPostCardProps {
  post: PostMeta;
}

export function CompactPostCard({ post }: CompactPostCardProps) {
  return (
    <Card className="enhanced-card smooth-transition group h-full flex flex-col bg-gradient-to-br from-sage/8 to-gold/6 border-sage/20 overflow-hidden relative">
      {/* Shine effect overlay */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
      </div>
      
      <CardHeader className="pb-3 flex-shrink-0 relative z-10">
        <div className="space-y-2">
          <Link href={`/blog/${post.slug}`}>
            <h3 className="philosophical-heading text-lg font-serif font-semibold leading-tight group-hover:text-white transition-colors duration-300 line-clamp-2">
              {post.title}
            </h3>
          </Link>
          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
            <time 
              dateTime={post.date}
              className="font-medium"
            >
              {formatDate(post.date)}
            </time>
            {post.tags && post.tags.length > 0 && (
              <>
                <span className="text-sage">•</span>
                <div className="flex flex-wrap gap-1">
                  {post.tags.slice(0, 2).map((tag) => (
                    <span 
                      key={tag} 
                      className="px-2 py-0.5 bg-gradient-to-r from-sage/20 to-gold/15 border border-sage/30 rounded-full text-xs font-medium text-sage group-hover:from-sage/30 group-hover:to-gold/25 group-hover:text-white transition-all duration-300"
                    >
                      {tag}
                    </span>
                  ))}
                  {post.tags.length > 2 && (
                    <span className="text-xs text-muted-foreground">
                      +{post.tags.length - 2}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col justify-between space-y-3 relative z-10">
        <p className="text-sm text-muted-foreground leading-relaxed line-clamp-3 group-hover:text-white/90 transition-colors duration-300">
          {post.excerpt}
        </p>
        <div className="pt-1">
          <Link 
            href={`/blog/${post.slug}`}
            className="inline-flex items-center text-xs font-medium text-sage hover:text-white transition-colors duration-300 group"
          >
            <span>Read more</span>
            <svg 
              className="ml-1 h-3 w-3 transform group-hover:translate-x-1 transition-transform duration-300" 
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