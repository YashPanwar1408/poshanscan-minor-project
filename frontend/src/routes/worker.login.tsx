import { t } from "@/lib/i18n";
import { createFileRoute } from "@tanstack/react-router";
import { RoleLogin } from "@/components/RoleLogin";

export const Route = createFileRoute("/worker/login")({
  head: () => ({
    meta: [
      { get title() { return t("Worker login — PoshanScan"); } },
      { name: "description", get content() { return t("Sign in to the PoshanScan field worker workspace."); } },
    ],
  }),
  component: () => <RoleLogin role="worker" />,
});
