import { Moon, Sun } from "lucide-react";
import { usePortalPrefs } from "@/lib/portal-prefs";
import { t } from "@/lib/i18n";

export function ThemeToggle() {
  const { theme, toggleTheme } = usePortalPrefs();
  const dark = theme === "dark";
  const label = t(dark ? "Switch to light mode" : "Switch to dark mode");

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={label}
      aria-pressed={dark}
      title={label}
      className="inline-flex size-9 items-center justify-center rounded-md border border-border bg-background text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {dark ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}
