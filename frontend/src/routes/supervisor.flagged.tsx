import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { FlaggedPage } from "@/supervisor/pages/FlaggedPage";

export const Route = createFileRoute("/supervisor/flagged")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Flagged Scans — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Flagged Scans")}>
        <FlaggedPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
