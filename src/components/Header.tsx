import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <Link href="/" className="group flex items-center space-x-3">
              <div className="relative">
                <h1 className="text-3xl font-serif font-bold tracking-tight philosophical-heading">
                  Phantasius
                </h1>
                <div className="absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-sage to-gold scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
              </div>
            </Link>
            <div className="hidden sm:block">
              <span className="text-sm text-muted-foreground">
                <span className="greek-text text-lg font-medium">φαντασία</span>
                <span className="ml-2 opacity-60">—</span>
                <span className="ml-2 italic">the space of imagination</span>
              </span>
            </div>
          </div>
          
          <nav className="flex items-center space-x-8">
            <Link 
              href="/" 
              className="text-sm font-medium transition-colors duration-300 hover:text-sage relative group"
            >
              <span>Essays</span>
              <div className="absolute -bottom-1 left-0 w-full h-0.5 bg-sage scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></div>
            </Link>
            <Link 
              href="/about" 
              className="text-sm font-medium transition-colors duration-300 hover:text-sage relative group"
            >
              <span>About</span>
              <div className="absolute -bottom-1 left-0 w-full h-0.5 bg-sage scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></div>
            </Link>
            <div className="border-l border-border pl-6">
              <ThemeToggle />
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
} 