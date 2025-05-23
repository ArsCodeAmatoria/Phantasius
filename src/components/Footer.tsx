export function Footer() {
  return (
    <footer className="border-t border-border bg-background/95">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="text-sm text-muted-foreground">
            <p className="greek-text mb-2">
              &ldquo;ἡ φαντασία ἐστι κίνησις ὑπὸ τῆς κατὰ ἐνέργειαν αἰσθήσεως γιγνομένη&rdquo;
            </p>
            <p className="text-xs italic">
              &ldquo;Phantasia is a movement produced by an active sensation&rdquo; — Aristotle
            </p>
          </div>
          
          <div className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} Phantasius. A space for philosophical reflection.
          </div>
        </div>
      </div>
    </footer>
  );
} 