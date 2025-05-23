import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  return (
    <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link href="/" className="flex items-center space-x-2">
            <h1 className="text-2xl font-serif font-semibold tracking-tight">
              Phantasius
            </h1>
          </Link>
          <span className="text-sm text-muted-foreground hidden sm:inline">
            <span className="greek-text">φαντασία</span> – the space of imagination
          </span>
        </div>
        
        <nav className="flex items-center space-x-6">
          <Link 
            href="/" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Essays
          </Link>
          <Link 
            href="/about" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            About
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
} 