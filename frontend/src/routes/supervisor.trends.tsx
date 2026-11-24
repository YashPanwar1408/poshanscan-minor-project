import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { TrendsPage } from "@/supervisor/pages/TrendsPage";

export const Route = createFileRoute("/supervisor/trends")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Trends — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Trends")}>
        <TrendsPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
