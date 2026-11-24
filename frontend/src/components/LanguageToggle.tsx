import { T } from "@/lib/i18n";
import { Languages } from "lucide-react";
import { t } from "@/lib/i18n";
import { usePortalPrefs, type PortalLanguage } from "@/lib/portal-prefs";

export function LanguageToggle({ compact = false }: { compact?: boolean }) {
  const { language, setLanguage } = usePortalPrefs();

  return (
    <label className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 text-foreground">
      <Languages size={15} aria-hidden="true" />
      <span className="sr-only">{t("Language")}</span>
      <select
        aria-label={t("Language")}
        title={t("Language")}
        value={language}
        onChange={(event) => setLanguage(event.target.value as PortalLanguage)}
        className={`min-h-9 cursor-pointer bg-transparent text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring ${compact ? "max-w-20" : "max-w-28"}`}
      >
        <option value="en">
          <T source="English" />
        </option>
        <option value="hi">हिन्दी</option>
      </select>
    </label>
  );
}
