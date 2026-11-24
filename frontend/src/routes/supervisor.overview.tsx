import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { OverviewPage } from "@/supervisor/pages/OverviewPage";

export const Route = createFileRoute("/supervisor/overview")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Overview — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Overview")}>
        <OverviewPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
