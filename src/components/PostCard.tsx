import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatDate, type PostMeta } from "@/lib/posts";

interface PostCardProps {
  post: PostMeta;
}

export function PostCard({ post }: PostCardProps) {
  return (
    <Card className="group hover:shadow-md transition-shadow duration-200">
      <CardHeader className="pb-4">
        <div className="space-y-2">
          <Link href={`/blog/${post.slug}`}>
            <h2 className="text-xl font-serif font-semibold leading-tight group-hover:text-primary transition-colors">
              {post.title}
            </h2>
          </Link>
          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <time dateTime={post.date}>{formatDate(post.date)}</time>
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
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground leading-relaxed">
          {post.excerpt}
        </p>
        <Link 
          href={`/blog/${post.slug}`}
          className="inline-block mt-4 text-sm font-medium text-primary hover:underline"
        >
          Read more →
        </Link>
      </CardContent>
    </Card>
  );
} 