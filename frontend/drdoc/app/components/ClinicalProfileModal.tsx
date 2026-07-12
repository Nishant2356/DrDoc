"use client";
import { useState, useEffect, useRef } from "react";
import { useAuth, ClinicalProfile } from "../context/AuthContext";

// ==========================================
// Tag Input Component (reusable for specialties & medicines)
// ==========================================

function TagInput({
  label,
  placeholder,
  tags,
  onTagsChange,
  suggestions,
}: {
  label: string;
  placeholder: string;
  tags: string[];
  onTagsChange: (tags: string[]) => void;
  suggestions: string[];
}) {
  const [input, setInput] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = suggestions.filter(
    (s) =>
      s.toLowerCase().includes(input.toLowerCase()) &&
      !tags.includes(s)
  );

  const addTag = (tag: string) => {
    const trimmed = tag.trim();
    if (trimmed && !tags.includes(trimmed)) {
      onTagsChange([...tags, trimmed]);
    }
    setInput("");
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const removeTag = (idx: number) => {
    onTagsChange(tags.filter((_, i) => i !== idx));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      if (input.trim()) addTag(input);
    } else if (e.key === "Backspace" && !input && tags.length) {
      removeTag(tags.length - 1);
    }
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-slate-700">
        {label}
      </label>

      {/* Tags display */}
      <div className="flex flex-wrap gap-2 mb-2">
        {tags.map((tag, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 bg-blue-50 text-blue-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-blue-200 transition-all hover:bg-blue-100"
          >
            {tag}
            <button
              type="button"
              onClick={() => removeTag(i)}
              className="ml-0.5 text-blue-400 hover:text-red-500 transition-colors font-bold text-xs"
            >
              ✕
            </button>
          </span>
        ))}
      </div>

      {/* Input */}
      <div className="relative">
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            setShowSuggestions(true);
          }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 transition-all"
        />

        {/* Suggestions dropdown */}
        {showSuggestions && input && filtered.length > 0 && (
          <div className="absolute z-50 w-full mt-1 bg-white border border-slate-200 rounded-xl shadow-xl max-h-40 overflow-y-auto">
            {filtered.map((s, i) => (
              <button
                key={i}
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => addTag(s)}
                className="w-full text-left px-4 py-2.5 text-sm text-slate-700 hover:bg-blue-50 hover:text-blue-700 transition-colors first:rounded-t-xl last:rounded-b-xl"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-400">
        Type and press Enter or comma to add. You can also pick from suggestions.
      </p>
    </div>
  );
}

// ==========================================
// Clinical Profile Modal
// ==========================================

const SPECIALTY_SUGGESTIONS = [
  "Allopathic (Western Medicine)",
  "Homeopathic",
  "Ayurvedic",
  "General Practice",
  "Internal Medicine",
  "Pediatrics",
  "Cardiology",
  "Dermatology",
  "Orthopedics",
  "Psychiatry",
  "Neurology",
  "Oncology",
  "Pulmonology",
  "Gastroenterology",
  "Endocrinology",
  "Rheumatology",
  "Nephrology",
  "Ophthalmology",
  "ENT (Otolaryngology)",
  "Urology",
  "Obstetrics & Gynecology",
  "Emergency Medicine",
  "Family Medicine",
  "Naturopathy",
  "Unani Medicine",
  "Chiropractic",
  "Osteopathic Medicine",
];

const MEDICINE_SUGGESTIONS = [
  "Nux Vomica",
  "Arsenic Album",
  "Bryonia",
  "Pulsatilla",
  "Sulphur",
  "Belladonna",
  "Lycopodium",
  "Amoxicillin",
  "Azithromycin",
  "Ciprofloxacin",
  "Metformin",
  "Lisinopril",
  "Atorvastatin",
  "Omeprazole",
  "Acetaminophen",
  "Ibuprofen",
  "Sumatriptan",
  "Aspirin",
  "Prednisone",
  "Fluticasone",
  "Montelukast",
  "Cetirizine",
  "Alprazolam",
  "Amlodipine",
  "Losartan",
  "Clopidogrel",
  "Pantoprazole",
  "Diclofenac",
  "Paracetamol",
  "Levothyroxine",
  "Metoprolol",
];

export default function ClinicalProfileModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const { user, updateClinicalProfile } = useAuth();

  const [specialties, setSpecialties] = useState<string[]>([]);
  const [medicines, setMedicines] = useState<string[]>([]);
  const [notePreferences, setNotePreferences] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  // Pre-fill from existing profile
  useEffect(() => {
    if (isOpen && user?.clinicalProfile) {
      setSpecialties(user.clinicalProfile.specialties);
      setMedicines(user.clinicalProfile.commonMedicines);
      setNotePreferences(user.clinicalProfile.notePreferences);
    } else if (isOpen) {
      setSpecialties([]);
      setMedicines([]);
      setNotePreferences("");
    }
    setError("");
  }, [isOpen, user?.clinicalProfile]);

  const handleSave = async () => {
    setError("");

    if (specialties.length === 0) {
      setError("Please add at least one medical specialty.");
      return;
    }
    if (medicines.length === 0) {
      setError("Please add at least one commonly prescribed medicine.");
      return;
    }
    if (!notePreferences.trim()) {
      setError("Please describe your note generation preferences.");
      return;
    }

    setSaving(true);

    try {
      const profile: ClinicalProfile = {
        specialties,
        commonMedicines: medicines,
        notePreferences: notePreferences.trim(),
      };
      await updateClinicalProfile(profile);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save profile.");
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-3xl shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto animate-modalSlideUp">
        {/* Header */}
        <div className="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-slate-100 px-8 py-5 rounded-t-3xl z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                <span className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center text-lg">
                  🩺
                </span>
                Clinical Profile
              </h2>
              <p className="text-sm text-slate-500 mt-1">
                Configure your AI scribe to match your practice
              </p>
            </div>
            <button
              onClick={onClose}
              className="w-9 h-9 rounded-xl bg-slate-100 hover:bg-slate-200 flex items-center justify-center text-slate-500 hover:text-slate-700 transition-all"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="px-8 py-6 space-y-8">
          {/* Section 1: Specialties */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-6 h-6 rounded-md bg-emerald-100 flex items-center justify-center text-sm">
                1
              </span>
              <span className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
                Field of Medicine
              </span>
            </div>
            <TagInput
              label="What type of clinical doctor are you?"
              placeholder="e.g. Allopathic, Homeopathic, Ayurvedic..."
              tags={specialties}
              onTagsChange={setSpecialties}
              suggestions={SPECIALTY_SUGGESTIONS}
            />
          </div>

          {/* Divider */}
          <div className="border-t border-slate-100" />

          {/* Section 2: Common Medicines */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-6 h-6 rounded-md bg-amber-100 flex items-center justify-center text-sm">
                2
              </span>
              <span className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
                Common Medicines
              </span>
            </div>
            <TagInput
              label="What medicines do you generally prescribe?"
              placeholder="e.g. Nux Vomica, Arsenic Album, Amoxicillin..."
              tags={medicines}
              onTagsChange={setMedicines}
              suggestions={MEDICINE_SUGGESTIONS}
            />
          </div>

          {/* Divider */}
          <div className="border-t border-slate-100" />

          {/* Section 3: Note Preferences */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-6 h-6 rounded-md bg-violet-100 flex items-center justify-center text-sm">
                3
              </span>
              <span className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
                Note Generation Preferences
              </span>
            </div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              How should your clinical notes be generated?
            </label>
            <textarea
              value={notePreferences}
              onChange={(e) => setNotePreferences(e.target.value)}
              rows={6}
              placeholder={`Describe your preferred note format and clinical protocols. For example:\n\n• Use SOAP format (Subjective, Objective, Assessment, Plan)\n• Always include follow-up timeframe in the Plan section\n• Default antibiotic for sinusitis: Amoxicillin 500mg TID for 7 days\n• If patient has penicillin allergy, switch to Azithromycin Z-Pak\n• All prescriptions must state dosage, route, and duration`}
              className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 transition-all resize-none leading-relaxed"
            />
            <p className="text-xs text-slate-400">
              Be as specific as possible — this text is injected directly into the AI&apos;s prompt to guide note generation.
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm font-medium animate-shake">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-white/95 backdrop-blur-sm border-t border-slate-100 px-8 py-5 rounded-b-3xl">
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-400">
              {specialties.length} specialties · {medicines.length} medicines
            </p>
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="px-5 py-2.5 text-sm font-semibold text-slate-600 bg-slate-100 rounded-xl hover:bg-slate-200 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-6 py-2.5 text-sm font-semibold text-white bg-blue-600 rounded-xl hover:bg-blue-700 shadow-lg shadow-blue-500/25 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {saving ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    Save Profile
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
