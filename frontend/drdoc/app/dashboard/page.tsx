"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";
import ClinicalProfileModal from "../components/ClinicalProfileModal";
import ReactMarkdown from "react-markdown";
import { useReactToPrint } from "react-to-print";

export default function DashboardPage() {
  const router = useRouter();
  const { user, isLoading, logout } = useAuth();
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [generatedNote, setGeneratedNote] = useState("");
  const [recentNotes, setRecentNotes] = useState<any[]>([]);

  const [threadId, setThreadId] = useState("");
  const [noteId, setNoteId] = useState("");
  const [feedbackText, setFeedbackText] = useState("");
  const [isCorrecting, setIsCorrecting] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const handlePrint = useReactToPrint({ contentRef });

  const fetchNotes = async () => {
    if (!user?.email) return;
    try {
      const res = await fetch(`/api/notes?email=${encodeURIComponent(user.email)}`);
      const data = await res.json();
      if (data.notes) setRecentNotes(data.notes);
    } catch (e) {
      console.error("Failed to fetch notes:", e);
    }
  };

  useEffect(() => {
    fetchNotes();
  }, [user?.email]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: "audio/webm" });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob);
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsGenerating(true);
    setGeneratedNote("");
    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");
      // Pass the user's email or ID so the backend knows which profile to fetch
      formData.append("email", user?.email || "");

      const res = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to generate note");

      setGeneratedNote(data.note);
      setThreadId(data.threadId);
      setNoteId(data.noteId);
      fetchNotes(); // Refresh recent notes list
    } catch (err) {
      console.error(err);
      alert("Error generating note. Check console for details.");
    } finally {
      setIsGenerating(false);
    }
  };

  const submitCorrection = async () => {
    if (!feedbackText || !threadId || !noteId) return;
    setIsCorrecting(true);
    try {
      const res = await fetch("/api/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threadId, noteId, feedback: feedbackText }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to correct note");
      
      setGeneratedNote(data.note);
      setFeedbackText("");
      fetchNotes();
      alert("Note corrected successfully!");
    } catch (err) {
      console.error(err);
      alert("Error correcting note.");
    } finally {
      setIsCorrecting(false);
    }
  };

  const deleteNote = async (id: string) => {
    if (!confirm("Are you sure you want to delete this note?")) return;
    try {
      const res = await fetch(`/api/notes?id=${id}`, { method: "DELETE" });
      if (res.ok) {
        if (noteId === id) {
          setGeneratedNote("");
          setNoteId("");
          setThreadId("");
        }
        fetchNotes();
      } else {
        alert("Failed to delete note.");
      }
    } catch (err) {
      console.error(err);
      alert("Error deleting note.");
    }
  };

  const viewNote = (note: any) => {
    setGeneratedNote(note.noteContent);
    setNoteId(note.id);
    setThreadId(""); // Clear thread ID because past notes cannot be corrected in the same continuous thread
    setFeedbackText("");
  };

  const hasProfile = !!user?.clinicalProfile;

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/login");
    }
  }, [isLoading, user, router]);

  if (isLoading || !user) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-slate-500">
          <span className="w-6 h-6 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin" />
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-slate-50 font-sans">
      {/* ========================================== */}
      {/* TOP NAVBAR */}
      {/* ========================================== */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-slate-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <span className="text-2xl">🩺</span>
            <span className="text-lg font-bold text-slate-800 tracking-tight">
              Dr. Doc
            </span>
            <span className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full ml-1">
              Dashboard
            </span>
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-3">
            {/* Update Clinical Profile Button */}
            <button
              onClick={() => setProfileModalOpen(true)}
              className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all active:scale-95 ${
                hasProfile
                  ? "bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200"
                  : "bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/25 animate-pulse-gentle"
              }`}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
              </svg>
              {hasProfile ? "Update Clinical Profile" : "Set Up Clinical Profile"}
            </button>

            {/* User Avatar / Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-slate-100 transition-all"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white text-sm font-bold shadow-inner">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <span className="text-sm font-medium text-slate-700 hidden sm:block">
                  {user.name}
                </span>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 text-slate-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>

              {/* Dropdown */}
              {showUserMenu && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setShowUserMenu(false)}
                  />
                  <div className="absolute right-0 mt-2 w-56 bg-white rounded-2xl shadow-xl border border-slate-100 py-2 z-50 animate-fadeIn">
                    <div className="px-4 py-3 border-b border-slate-100">
                      <p className="text-sm font-semibold text-slate-800">
                        {user.name}
                      </p>
                      <p className="text-xs text-slate-500">{user.email}</p>
                      <span className="inline-flex items-center gap-1 mt-1.5 text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                        Doctor
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        setShowUserMenu(false);
                        setProfileModalOpen(true);
                      }}
                      className="w-full text-left px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-slate-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                      </svg>
                      Clinical Profile
                    </button>
                    <div className="border-t border-slate-100 mt-1 pt-1">
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          logout();
                          router.push("/");
                        }}
                        className="w-full text-left px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors flex items-center gap-2"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zm10.293 9.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L14.586 9H7a1 1 0 100 2h7.586l-1.293 1.293z" clipRule="evenodd" />
                        </svg>
                        Sign Out
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ========================================== */}
      {/* MAIN CONTENT */}
      {/* ========================================== */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Profile Setup Banner (shown when no clinical profile) */}
        {!hasProfile && (
          <div className="mb-8 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-2xl p-6 flex items-start gap-4 animate-fadeIn">
            <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center justify-center text-2xl shrink-0">
              ⚠️
            </div>
            <div className="flex-1">
              <h3 className="text-base font-bold text-amber-900 mb-1">
                Complete Your Clinical Profile
              </h3>
              <p className="text-sm text-amber-700 leading-relaxed mb-3">
                Before you can generate clinical notes, you need to set up your
                clinical profile. Tell us about your specialties, commonly
                prescribed medicines, and note formatting preferences so the AI
                can tailor its output to your practice.
              </p>
              <button
                onClick={() => setProfileModalOpen(true)}
                className="inline-flex items-center gap-2 bg-amber-600 text-white text-sm font-semibold px-5 py-2.5 rounded-xl hover:bg-amber-700 shadow-lg shadow-amber-500/20 active:scale-95 transition-all"
              >
                Set Up Profile Now
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Profile Summary (shown when profile exists) */}
        {hasProfile && (
          <div className="mb-8 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-6 animate-fadeIn">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-emerald-100 flex items-center justify-center text-lg">
                  ✅
                </div>
                <div>
                  <h3 className="text-base font-bold text-emerald-900">
                    Clinical Profile Active
                  </h3>
                  <p className="text-xs text-emerald-600">
                    Your AI scribe is configured and ready
                  </p>
                </div>
              </div>
              <button
                onClick={() => setProfileModalOpen(true)}
                className="text-sm font-medium text-emerald-700 hover:text-emerald-800 underline underline-offset-2 transition-colors"
              >
                Edit
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Specialties */}
              <div className="bg-white/60 rounded-xl p-4 border border-emerald-100">
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-2">
                  Specialties
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {user.clinicalProfile!.specialties.map((s, i) => (
                    <span
                      key={i}
                      className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-md font-medium"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>

              {/* Medicines */}
              <div className="bg-white/60 rounded-xl p-4 border border-emerald-100">
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-2">
                  Common Medicines
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {user.clinicalProfile!.commonMedicines.slice(0, 5).map((m, i) => (
                    <span
                      key={i}
                      className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-md font-medium"
                    >
                      {m}
                    </span>
                  ))}
                  {user.clinicalProfile!.commonMedicines.length > 5 && (
                    <span className="text-xs text-emerald-500 font-medium px-1">
                      +{user.clinicalProfile!.commonMedicines.length - 5} more
                    </span>
                  )}
                </div>
              </div>

              {/* Preferences */}
              <div className="bg-white/60 rounded-xl p-4 border border-emerald-100">
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-2">
                  Note Preferences
                </p>
                <p className="text-xs text-slate-600 line-clamp-3 leading-relaxed">
                  {user.clinicalProfile!.notePreferences}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ========================================== */}
        {/* NOTE GENERATION AREA */}
        {/* ========================================== */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Card — Generate Note */}
          <div className="lg:col-span-2">
            <div
              className={`bg-white rounded-2xl border shadow-sm transition-all overflow-hidden ${
                hasProfile
                  ? "border-slate-200"
                  : "border-slate-200 opacity-50 pointer-events-none select-none"
              }`}
            >
              <div className="p-6 border-b border-slate-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-lg">
                      🎙️
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-900">
                        Record Consultation
                      </h2>
                      <p className="text-xs text-slate-500">
                        Click start to begin capturing the conversation
                      </p>
                    </div>
                  </div>
                  {!hasProfile && (
                    <span className="text-xs font-medium text-slate-400 bg-slate-100 px-3 py-1 rounded-full">
                      🔒 Profile required
                    </span>
                  )}
                </div>
              </div>

              <div className="p-6 flex flex-col items-center justify-center min-h-[300px] bg-slate-50">
                {isGenerating ? (
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                    <p className="text-slate-600 font-medium">Processing audio & generating note...</p>
                  </div>
                ) : generatedNote ? (
                  <div className="w-full h-full text-left flex flex-col">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="font-bold text-slate-800">Generated Clinical Note</h3>
                      <div className="flex gap-3">
                        <button onClick={() => handlePrint()} className="text-xs text-slate-700 font-medium hover:bg-slate-100 border border-slate-200 px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M5 4v3H4a2 2 0 00-2 2v3a2 2 0 002 2h1v2a2 2 0 002 2h6a2 2 0 002-2v-2h1a2 2 0 002-2V9a2 2 0 00-2-2h-1V4a2 2 0 00-2-2H7a2 2 0 00-2 2zm8 0H7v3h6V4zm0 8H7v4h6v-4z" clipRule="evenodd" />
                          </svg>
                          Print / PDF
                        </button>
                        <button onClick={() => setGeneratedNote("")} className="text-xs text-blue-600 font-medium hover:underline flex items-center">Start New</button>
                      </div>
                    </div>
                    <div ref={contentRef} className="bg-white p-4 rounded-xl border border-slate-200 text-sm text-slate-700 h-[220px] overflow-y-auto mb-4 print:h-auto print:overflow-visible print:border-none print:p-8 print:m-0 print:text-base">
                      <div className="hidden print:block mb-6 border-b border-slate-200 pb-4">
                        <h1 className="text-2xl font-bold text-slate-900">Clinical Consultation Note</h1>
                        <p className="text-sm text-slate-500 mt-1">Generated by Dr. Doc AI Scribe</p>
                      </div>
                      <ReactMarkdown
                        components={{
                          h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-4 mb-2 text-slate-900" {...props} />,
                          h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-4 mb-2 text-slate-800" {...props} />,
                          h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-3 mb-2 text-slate-800" {...props} />,
                          p: ({node, ...props}) => <p className="mb-3 leading-relaxed" {...props} />,
                          ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
                          ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 space-y-1" {...props} />,
                          li: ({node, ...props}) => <li className="mb-1" {...props} />,
                          strong: ({node, ...props}) => <strong className="font-semibold text-slate-900" {...props} />,
                          em: ({node, ...props}) => <em className="italic" {...props} />,
                          blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-slate-300 pl-4 italic text-slate-600 mb-3" {...props} />
                        }}
                      >
                        {generatedNote}
                      </ReactMarkdown>
                    </div>
                    <div className="mt-auto border-t border-slate-100 pt-4">
                      <p className="text-xs font-semibold text-slate-600 mb-2">Human Review & Correction</p>
                      <div className="flex gap-2">
                        <input 
                          type="text" 
                          value={feedbackText}
                          onChange={(e) => setFeedbackText(e.target.value)}
                          placeholder="Type any corrections here (e.g. 'Change dosage to 250mg')" 
                          className="flex-1 text-sm px-3 py-2 rounded-lg border border-slate-200 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                          disabled={isCorrecting}
                        />
                        <button 
                          onClick={submitCorrection}
                          disabled={isCorrecting || !feedbackText}
                          className="px-4 py-2 bg-slate-800 text-white text-sm font-medium rounded-lg hover:bg-slate-900 disabled:opacity-50 transition-colors"
                        >
                          {isCorrecting ? "Fixing..." : "Correct Note"}
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <button
                      disabled={!hasProfile}
                      onClick={isRecording ? stopRecording : startRecording}
                      className={`relative flex items-center justify-center w-24 h-24 rounded-full shadow-lg transition-all active:scale-95 ${
                        isRecording 
                          ? "bg-red-50 hover:bg-red-100 border-4 border-red-500 shadow-red-500/20" 
                          : "bg-blue-600 hover:bg-blue-700 border-4 border-transparent shadow-blue-500/30 disabled:opacity-50 disabled:bg-slate-300"
                      }`}
                    >
                      {isRecording && (
                        <span className="absolute inset-0 rounded-full animate-ping bg-red-400 opacity-20"></span>
                      )}
                      {isRecording ? (
                        <div className="w-8 h-8 bg-red-600 rounded-sm"></div>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                    <p className={`mt-6 font-semibold ${isRecording ? "text-red-600 animate-pulse" : "text-slate-600"}`}>
                      {isRecording ? "Recording in progress..." : "Click to Start Recording"}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar — Recent Notes */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm h-full flex flex-col">
              <div className="p-5 border-b border-slate-100">
                <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-slate-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  Recent Notes
                </h3>
              </div>
              <div className="p-0 overflow-y-auto max-h-[350px]">
                {recentNotes.length > 0 ? (
                  recentNotes.map((note) => {
                    const patientName = note.patientInfo?.name && note.patientInfo.name !== "Unknown" ? note.patientInfo.name : "Anonymous Patient";
                    const date = new Date(note.createdAt).toLocaleDateString();
                    return (
                      <div 
                        key={note.id} 
                        onClick={() => viewNote(note)}
                        className="p-4 border-b border-slate-50 hover:bg-slate-50 cursor-pointer transition-colors group relative"
                      >
                        <div className="flex items-start justify-between pr-6">
                          <div>
                            <p className="text-sm font-semibold text-slate-800">{patientName}</p>
                            <p className="text-xs text-slate-500">{date}</p>
                          </div>
                          <span className="text-xs font-medium bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded-md">Saved</span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteNote(note.id);
                          }}
                          className="absolute right-4 top-4 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-500"
                          title="Delete Note"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </div>
                    )
                  })
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center text-2xl mb-3">
                      📋
                    </div>
                    <p className="text-sm font-medium text-slate-500 mb-1">
                      No notes yet
                    </p>
                    <p className="text-xs text-slate-400">
                      Generated notes will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
                <p className="text-2xl font-bold text-slate-800">{recentNotes.length}</p>
                <p className="text-xs text-slate-500 mt-1">Notes Generated</p>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
                <p className="text-2xl font-bold text-slate-800">
                  {hasProfile ? "✓" : "—"}
                </p>
                <p className="text-xs text-slate-500 mt-1">Profile Status</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Clinical Profile Modal */}
      <ClinicalProfileModal
        isOpen={profileModalOpen}
        onClose={() => setProfileModalOpen(false)}
      />
    </div>
  );
}
