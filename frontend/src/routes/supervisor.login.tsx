import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { SupervisorLoginPage } from "@/supervisor/pages/LoginPage";

export const Route = createFileRoute("/supervisor/login")({
  head: () => ({
    meta: [
      {
        get title() {
          return t("Supervisor login — PoshanScan");
        },
      },
      {
        name: "description",
        get content() {
          return t("Sign in to the PoshanScan supervisor workspace.");
        },
      },
    ],
  }),
  component: SupervisorLoginPage,
});
