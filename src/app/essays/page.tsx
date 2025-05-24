import { Suspense } from "react";
import { getSortedPostsData } from "@/lib/posts";
import { EssaysClient } from "@/components/EssaysClient";

export default function EssaysPage() {
  const allPostsData = getSortedPostsData();

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

        {/* Client-side interactive components */}
        <Suspense fallback={<div className="text-center py-12">Loading essays...</div>}>
          <EssaysClient posts={allPostsData} />
        </Suspense>

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