import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell, ProtectedRoute } from "@/supervisor/components/AppShell";
import { LocationsPage } from "@/supervisor/pages/LocationsPage";

export const Route = createFileRoute("/supervisor/locations")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Locations — PoshanScan Supervisor");
        },
      },
    ],
  }),
  component: () => (
    <ProtectedRoute>
      <AppShell title={t("Locations")}>
        <LocationsPage />
      </AppShell>
    </ProtectedRoute>
  ),
});
