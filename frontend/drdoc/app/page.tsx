"use client";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function LandingPage() {
  const [mounted, setMounted] = useState(false);

  // Ensure hydration matches for the animated background elements
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative min-h-screen bg-slate-50 flex flex-col justify-between items-center p-6 font-sans overflow-hidden select-none">

      {/* --- Ambient Floating Bubbles Background --- */}
      {mounted && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden z-0">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="absolute rounded-full bg-blue-400/10 mix-blend-multiply filter blur-sm animate-float"
              style={{
                width: `${Math.random() * 60 + 20}px`,
                height: `${Math.random() * 60 + 20}px`,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${Math.random() * 10 + 10}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* --- Top Navbar --- */}
      <header className="w-full max-w-6xl flex justify-between items-center py-4 z-10">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🚀</span>
          <span className="text-xl font-bold text-slate-800 tracking-tight">Dr. Doc</span>
        </div>
        <div className="text-sm font-medium text-slate-500 bg-white px-3 py-1 rounded-full shadow-sm border border-slate-100">
          Hackathon Edition v1.0
        </div>
      </header>

      {/* --- Main Hero Section --- */}
      <main className="flex-1 flex flex-col justify-center items-center text-center max-w-3xl w-full z-10 mt-8 mb-12">

        {/* The Video Container */}
        {/* Adds a beautiful white border, soft shadow, and perfectly clips the video */}
        <div className="relative w-full max-w-xl aspect-video bg-white rounded-3xl overflow-hidden shadow-2xl shadow-blue-900/10 mb-10 border-8 border-white group">
          <video
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-cover pointer-events-none transform group-hover:scale-105 transition-transform duration-700 ease-in-out"
          >
            {/* Make sure your video is named hero-doctor.mp4 in the /public folder! */}
            <source src="/hero-doctor.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>

          {/* Subtle inner shadow overlay to blend the video edges */}
          <div className="absolute inset-0 shadow-inner rounded-2xl pointer-events-none"></div>
        </div>

        {/* Headline & Value Prop */}
        <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          Your Intelligent <span className="text-blue-600">Continuous-Learning</span> Scribe
        </h1>
        <p className="text-lg text-slate-600 mb-10 leading-relaxed max-w-2xl mx-auto">
          The only clinical note generator that studies your edits, updates your local database, and never makes the same mistake twice. Focus on your patients, let the AI handle the paperwork.
        </p>

        {/* Call to Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
          <Link
            href="/login?role=doctor"
            className="bg-blue-600 text-white font-semibold py-3.5 px-8 rounded-xl hover:bg-blue-700 shadow-lg shadow-blue-500/30 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
          >
            Sign In as Doctor
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </Link>
          <Link
            href="/login?role=patient"
            className="bg-white text-slate-700 font-semibold py-3.5 px-8 rounded-xl border-2 border-slate-200 hover:border-slate-300 hover:bg-slate-50 active:scale-95 transition-all duration-200"
          >
            Patient Portal
          </Link>
        </div>
      </main>

      {/* --- Footer --- */}
      <footer className="text-sm font-medium text-slate-400 py-6 z-10">
        Secure, HIPAA-compliant pipeline with real-time continuous learning loops.
      </footer>

      {/* --- Custom Float Animation (Inline so you don't need to touch Tailwind configs!) --- */}
      <style jsx global>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-20px) scale(1.05); }
        }
        .animate-float {
          animation-name: float;
          animation-timing-function: ease-in-out;
          animation-iteration-count: infinite;
        }
      `}</style>
    </div>
  );
}