import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

export const metadata = {
  title: "About | Phantasius",
  description: "Learn about the philosophy and purpose behind Phantasius, a space for exploring ancient wisdom and modern consciousness.",
};

export default function About() {
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

        {/* Page Content */}
        <div className="prose prose-lg max-w-none">
          <h1 className="text-4xl font-serif font-bold leading-tight mb-8">
            About Phantasius
          </h1>

          <div className="text-center mb-12">
            <p className="text-2xl greek-text mb-2">φαντασία</p>
            <p className="text-muted-foreground italic">
              &ldquo;the space where appearances manifest in the mind&rdquo;
            </p>
          </div>

          <div className="space-y-6 leading-relaxed">
            <p>
              <strong>Phantasius</strong> is a space dedicated to exploring the ancient Greek concept of 
              <em> phantasia</em> and its profound relevance to our modern understanding of consciousness, 
              imagination, and human experience.
            </p>

            <p>
              In the philosophical traditions of antiquity, phantasia represented neither mere sensation 
              nor pure intellection, but rather the mysterious space where mental images, memories, and 
              imaginative constructs arise and play out their eternal dance. It is the internal theatre 
              of consciousness—the realm where appearances manifest and meaning emerges.
            </p>

            <h2>The Ancient Wisdom</h2>

            <p>
              The Stoics understood that our phantasiai—our mental impressions and the images that arise 
              in consciousness—form the very fabric of our subjective reality. As Epictetus taught, 
              it is not what happens to us, but our phantasiai about what happens, that truly affect us.
            </p>

            <p>
              Plato&rsquo;s insights into phantasia as the realm of appearances remind us that what we often 
              take to be reality is frequently the play of images in consciousness. Yet these very images 
              are the raw material from which wisdom and understanding emerge.
            </p>

            <h2>Modern Relevance</h2>

            <p>
              Today, as we navigate an increasingly complex world of digital images, virtual realities, 
              and information overload, the ancient understanding of phantasia becomes more relevant than ever. 
              Our capacity to work skillfully with mental images—to observe them, understand them, and 
              choose our relationship to them—may be one of the most important skills we can develop.
            </p>

            <p>
              This blog explores these themes through philosophical reflection, practical wisdom, and 
              the continuing dialogue between ancient insights and contemporary experience.
            </p>

            <h2>The Space of Imagination</h2>

            <p>
              Phantasia is where creativity blooms, where memory reconstructs the past, and where 
              imagination projects possible futures. It is, perhaps, the most distinctly human of 
              all our capacities—the space where we are neither purely rational nor merely sensual, 
              but something more mysterious and wonderful.
            </p>

            <p>
              Welcome to this exploration of the theatre of mental images. May these reflections 
              serve as invitations to deeper understanding and more skillful living.
            </p>
          </div>

          <div className="mt-12 pt-8 border-t border-border text-center">
            <p className="text-muted-foreground italic">
              &ldquo;The unexamined life is not worth living.&rdquo; — Socrates
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 