import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type TextScale = "sm" | "md" | "lg";
export type PortalTheme = "light" | "dark";
export type PortalLanguage = "en" | "hi";

type PortalPrefs = {
  textScale: TextScale;
  highContrast: boolean;
  theme: PortalTheme;
  language: PortalLanguage;
  setTextScale: (scale: TextScale) => void;
  setTheme: (theme: PortalTheme) => void;
  setLanguage: (language: PortalLanguage) => void;
  toggleHighContrast: () => void;
  toggleTheme: () => void;
};

const STORAGE_KEY = "poshanscan_portal_prefs";
const PrefsContext = createContext<PortalPrefs | null>(null);

function readStored(): {
  textScale: TextScale;
  highContrast: boolean;
  theme: PortalTheme;
  language: PortalLanguage;
} {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { textScale: "md", highContrast: false, theme: "light", language: "en" };
    const parsed = JSON.parse(raw) as {
      textScale?: TextScale;
      highContrast?: boolean;
      theme?: PortalTheme;
      language?: PortalLanguage;
    };
    const textScale =
      parsed.textScale === "sm" || parsed.textScale === "lg" ? parsed.textScale : "md";
    const theme = parsed.theme === "dark" ? "dark" : "light";
    const language = parsed.language === "hi" ? "hi" : "en";
    return { textScale, highContrast: Boolean(parsed.highContrast), theme, language };
  } catch {
    return { textScale: "md", highContrast: false, theme: "light", language: "en" };
  }
}

export function PortalPrefsProvider({ children }: { children: ReactNode }) {
  const [initialPrefs] = useState(readStored);
  const [textScale, setTextScaleState] = useState<TextScale>(initialPrefs.textScale);
  const [highContrast, setHighContrast] = useState(initialPrefs.highContrast);
  const [theme, setTheme] = useState<PortalTheme>(initialPrefs.theme);
  const [language, setLanguageState] = useState<PortalLanguage>(initialPrefs.language);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset["textScale"] = textScale;
    root.classList.toggle("high-contrast", highContrast);
    root.classList.toggle("dark", theme === "dark");
    root.style.colorScheme = theme;
    root.lang = language === "hi" ? "hi-IN" : "en-IN";
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ textScale, highContrast, theme, language }));
  }, [textScale, highContrast, theme, language]);

  function setLanguage(next: PortalLanguage) {
    setLanguageState(next);
    try {
      const current = readStored();
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ ...current, textScale, highContrast, theme, language: next }),
      );
    } catch {
      // The in-memory preference still applies when browser storage is unavailable.
    }
  }

  return (
    <PrefsContext.Provider
      value={{
        textScale,
        highContrast,
        theme,
        language,
        setTextScale: setTextScaleState,
        setTheme,
        setLanguage,
        toggleHighContrast: () => setHighContrast((value) => !value),
        toggleTheme: () => setTheme((value) => (value === "light" ? "dark" : "light")),
      }}
    >
      {children}
    </PrefsContext.Provider>
  );
}

export function usePortalPrefs() {
  const context = useContext(PrefsContext);
  if (!context) throw new Error("PortalPrefsProvider missing");
  return context;
}
