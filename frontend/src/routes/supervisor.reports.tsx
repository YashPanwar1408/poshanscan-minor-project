import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { ReportsPage } from "@/supervisor/pages/ReportsPage";

export const Route = createFileRoute("/supervisor/reports")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Reports — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Reports")}>
        <ReportsPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
