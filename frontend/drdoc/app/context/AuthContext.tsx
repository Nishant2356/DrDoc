"use client";
import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

// ==========================================
// Types
// ==========================================

export interface ClinicalProfile {
  specialties: string[];
  commonMedicines: string[];
  notePreferences: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: "DOCTOR";
  clinicalProfile: ClinicalProfile | null;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  updateClinicalProfile: (profile: ClinicalProfile) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ==========================================
// Provider
// ==========================================

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Hydrate from localStorage on mount (session persistence)
  useEffect(() => {
    const stored = localStorage.getItem("drdoc_user");
    if (stored) {
      try {
        setUser(JSON.parse(stored));
      } catch {
        localStorage.removeItem("drdoc_user");
      }
    }
    setIsLoading(false);
  }, []);

  // Persist user to localStorage whenever it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem("drdoc_user", JSON.stringify(user));
    }
  }, [user]);

  const login = async (email: string, password: string) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Login failed.");
    }

    setUser(data.user);
  };

  const signup = async (name: string, email: string, password: string) => {
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Signup failed.");
    }

    setUser(data.user);
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("drdoc_user");
  };

  const updateClinicalProfile = async (profile: ClinicalProfile) => {
    if (!user) return;

    const res = await fetch("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId: user.id,
        specialties: profile.specialties,
        commonMedicines: profile.commonMedicines,
        notePreferences: profile.notePreferences,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Profile update failed.");
    }

    // Update local user state with the saved profile
    const updatedUser = { ...user, clinicalProfile: data.clinicalProfile };
    setUser(updatedUser);
  };

  return (
    <AuthContext.Provider
      value={{ user, isLoading, login, signup, logout, updateClinicalProfile }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// ==========================================
// Hook
// ==========================================

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (ctx === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
