"use client";

import { useEffect, useState } from "react";

export function ReadingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    function updateReadingProgress() {
      const article = document.querySelector('article');
      if (!article) return;
      
      const scrollTop = window.pageYOffset;
      const articleTop = article.offsetTop;
      const articleHeight = article.offsetHeight;
      const windowHeight = window.innerHeight;
      
      const progressValue = Math.max(0, Math.min(1, 
        (scrollTop - articleTop + windowHeight * 0.1) / articleHeight
      ));
      
      setProgress(progressValue);
    }
    
    window.addEventListener('scroll', updateReadingProgress);
    window.addEventListener('resize', updateReadingProgress);
    updateReadingProgress();
    
    return () => {
      window.removeEventListener('scroll', updateReadingProgress);
      window.removeEventListener('resize', updateReadingProgress);
    };
  }, []);

  return (
    <div 
      className="reading-progress"
      style={{ transform: `scaleX(${progress})` }}
    />
  );
} 