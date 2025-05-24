"use client";

import { useState, useMemo, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { getSortedPostsData } from "@/lib/posts";
import { PostCard } from "@/components/PostCard";
import { Button } from "@/components/ui/button";

const POSTS_PER_PAGE = 6;

export default function EssaysPage() {
  const allPostsData = getSortedPostsData();
  const searchParams = useSearchParams();
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  // Initialize tag from URL params
  useEffect(() => {
    const tagParam = searchParams.get('tag');
    if (tagParam && tagParam !== selectedTag) {
      setSelectedTag(tagParam);
      setCurrentPage(1);
    }
  }, [searchParams, selectedTag]);

  // Get all unique tags
  const allTags = useMemo(() => {
    const tags = new Set<string>();
    allPostsData.forEach(post => {
      post.tags?.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }, [allPostsData]);

  // Filter posts by selected tag
  const filteredPosts = useMemo(() => {
    if (!selectedTag) return allPostsData;
    return allPostsData.filter(post => 
      post.tags?.includes(selectedTag)
    );
  }, [allPostsData, selectedTag]);

  // Calculate pagination
  const totalPages = Math.ceil(filteredPosts.length / POSTS_PER_PAGE);
  const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
  const endIndex = startIndex + POSTS_PER_PAGE;
  const currentPosts = filteredPosts.slice(startIndex, endIndex);

  // Reset page when tag changes
  const handleTagChange = (tag: string | null) => {
    setSelectedTag(tag);
    setCurrentPage(1);
    
    // Update URL without page refresh
    const url = new URL(window.location.href);
    if (tag) {
      url.searchParams.set('tag', tag);
    } else {
      url.searchParams.delete('tag');
    }
    window.history.replaceState({}, '', url);
  };

  return (
    <div className="scroll-container py-16">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-serif font-bold philosophical-heading mb-6">
            Philosophical Essays
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed max-w-3xl mx-auto">
            Explorations of ancient wisdom, consciousness, and the enduring questions 
            that shape human understanding. A collection of reflections on <span className="greek-text">φαντασία</span> and 
            the philosophical life.
          </p>
          
          {/* Decorative element */}
          <div className="flex justify-center pt-6">
            <div className="w-24 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
            <div className="mx-4 text-sage text-lg">⚬</div>
            <div className="w-24 h-px bg-gradient-to-r from-transparent via-sage to-transparent"></div>
          </div>
        </div>

        {/* Tag Filter */}
        <div className="mb-12">
          <h3 className="text-lg font-serif font-medium mb-4 text-center">Filter by Topic</h3>
          <div className="flex flex-wrap justify-center gap-3">
            <Button
              variant={selectedTag === null ? "default" : "outline"}
              onClick={() => handleTagChange(null)}
              className={`${
                selectedTag === null 
                  ? "bg-sage hover:bg-sage/90 text-white" 
                  : "border-sage/30 hover:bg-sage/10 hover:border-sage/50"
              } transition-all duration-300`}
            >
              All Essays ({allPostsData.length})
            </Button>
            {allTags.map((tag) => (
              <Button
                key={tag}
                variant={selectedTag === tag ? "default" : "outline"}
                onClick={() => handleTagChange(tag)}
                className={`${
                  selectedTag === tag 
                    ? "bg-sage hover:bg-sage/90 text-white" 
                    : "border-sage/30 hover:bg-sage/10 hover:border-sage/50"
                } transition-all duration-300 capitalize`}
              >
                {tag} ({allPostsData.filter(post => post.tags?.includes(tag)).length})
              </Button>
            ))}
          </div>
        </div>

        {/* Posts Grid */}
        <div className="scroll-spacing">
          {filteredPosts.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground text-lg">
                No essays found for the selected topic.
              </p>
            </div>
          ) : (
            <>
              <div className="grid gap-8 max-w-4xl mx-auto">
                {currentPosts.map((post, index) => (
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

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-16 flex justify-center items-center space-x-4">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                    className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 disabled:opacity-50 disabled:hover:bg-transparent"
                  >
                    Previous
                  </Button>
                  
                  <div className="flex items-center space-x-2">
                    {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                      <Button
                        key={page}
                        variant={currentPage === page ? "default" : "outline"}
                        onClick={() => setCurrentPage(page)}
                        className={`w-10 h-10 p-0 ${
                          currentPage === page
                            ? "bg-sage hover:bg-sage/90 text-white"
                            : "border-sage/30 hover:bg-sage/10 hover:border-sage/50"
                        }`}
                      >
                        {page}
                      </Button>
                    ))}
                  </div>
                  
                  <Button
                    variant="outline"
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                    className="border-sage/30 hover:bg-sage/10 hover:border-sage/50 disabled:opacity-50 disabled:hover:bg-transparent"
                  >
                    Next
                  </Button>
                </div>
              )}

              {/* Results Summary */}
              <div className="mt-8 text-center text-sm text-muted-foreground">
                Showing {startIndex + 1}-{Math.min(endIndex, filteredPosts.length)} of {filteredPosts.length} essays
                {selectedTag && (
                  <span className="ml-2">
                    in <span className="text-sage font-medium capitalize">{selectedTag}</span>
                  </span>
                )}
              </div>
            </>
          )}
        </div>

        {/* Call to Action */}
        <div className="mt-20 pt-12 border-t border-sage/20 text-center">
          <div className="max-w-2xl mx-auto">
            <h3 className="text-2xl font-serif font-semibold philosophical-heading mb-4">
              Continue Your Philosophical Journey
            </h3>
            <p className="text-muted-foreground mb-6">
              Each essay invites deeper reflection on the timeless questions of human experience. 
              Take time to contemplate, question, and integrate these insights into your own philosophical practice.
            </p>
            <div className="greek-text text-2xl">
              φαντασία
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 