import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { WorkersPage } from "@/supervisor/pages/WorkersPage";

export const Route = createFileRoute("/supervisor/workers")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Workers — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Workers")}>
        <WorkersPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
