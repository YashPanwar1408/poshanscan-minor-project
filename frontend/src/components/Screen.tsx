import { T, t } from "@/lib/i18n";
import { Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, LogOut, WifiOff, UserRound } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Brand } from "./Brand";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { listQueued, deleteQueued } from "@/lib/offline";
import { api } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";
import { LanguageToggle } from "@/components/LanguageToggle";

type OfflineSyncItemResult = {
  client_scan_id?: string | null;
  scan_id?: string | null;
  error?: string | null;
  message?: string | null;
};

export function RequireRole({ allowed, children }: { allowed: string[]; children: ReactNode }) {
  const { user, ready } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (ready && (!user || !allowed.includes(user.role))) navigate({ to: "/login", replace: true });
  }, [ready, user, allowed, navigate]);
  if (!ready)
    return (
      <div className="min-h-screen grid place-items-center text-muted-foreground">
        <T source="Loading…" />
      </div>
    );
  if (!user || !allowed.includes(user.role)) return null;
  return <>{children}</>;
}
export function WorkerShell({
  children,
  title,
  backToScreening = false,
  backAgeCategory,
}: {
  children: ReactNode;
  title: string;
  backToScreening?: boolean;
  backAgeCategory?: "child_6_59m" | "adult" | undefined;
}) {
  const { logout, user } = useAuth();
  const [profileOpen, setProfileOpen] = useState(false);
  const [pending, setPending] = useState(0);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncRun, setSyncRun] = useState(0);
  const rejectedSyncIds = useRef(new Set<string>());
  useEffect(() => {
    let active = true;
    async function sync() {
      try {
        const queuedScans = await listQueued();
        if (!active) return;
        setPending(queuedScans.length);
        if (!navigator.onLine || !queuedScans.length) return;
        const scans = queuedScans.filter(
          (scan) => !rejectedSyncIds.current.has(scan.client_scan_id),
        );
        if (!scans.length) return;
        const body = new FormData();
        body.append("metadata", JSON.stringify(scans.map(({ image, ...metadata }) => metadata)));
        scans.forEach((scan) =>
          body.append(
            "images",
            new File([scan.image], `${scan.client_scan_id}.jpg`, {
              type: scan.image.type || "image/jpeg",
            }),
          ),
        );
        const response = await api.post<OfflineSyncItemResult[]>("/sync", body);
        const resultByClientId = new Map(
          (Array.isArray(response.data) ? response.data : []).map((item) => [
            item.client_scan_id,
            item,
          ]),
        );
        let firstFailure: string | null = null;
        for (const scan of scans) {
          const item = resultByClientId.get(scan.client_scan_id);
          if (!item || item.error || !item.scan_id) {
            rejectedSyncIds.current.add(scan.client_scan_id);
            firstFailure ??= item?.message || "The server did not acknowledge this scan.";
            continue;
          }
          await deleteQueued(scan.client_scan_id);
        }
        const remaining = await listQueued();
        if (!active) return;
        setPending(remaining.length);
        if (firstFailure) setSyncError(firstFailure);
        else if (!remaining.length) setSyncError(null);
      } catch {
        /* keep queued scans for the next retry */
      }
    }
    sync();
    window.addEventListener("online", sync);
    const interval = window.setInterval(sync, 30000);
    return () => {
      active = false;
      window.removeEventListener("online", sync);
      window.clearInterval(interval);
    };
  }, [syncRun]);

  function retryRejectedScans() {
    rejectedSyncIds.current.clear();
    setSyncRun((current) => current + 1);
  }
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
          <Brand />
          <div className="flex items-center gap-3">
            <LanguageToggle compact />
            <ThemeToggle />
            <div className="relative">
              <Button
                variant="outline"
                onClick={() => setProfileOpen((value) => !value)}
                aria-expanded={profileOpen}
                title={t("Worker profile")}
              >
                <UserRound size={16} />
                <span className="hidden sm:inline">{user?.name}</span>
              </Button>
              {profileOpen && (
                <div className="absolute right-0 top-full z-20 mt-2 w-56 rounded-md border border-border bg-card p-4 text-sm shadow-lg">
                  <p className="font-semibold">{user?.name}</p>
                  <p className="mt-1 text-muted-foreground">{user?.phone}</p>
                  {user?.email && <p className="mt-1 text-muted-foreground">{user.email}</p>}
                  {user?.village && <p className="mt-1 text-muted-foreground">{user.village}</p>}
                  <p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">
                    {user?.role}
                  </p>
                </div>
              )}
            </div>
            {pending > 0 && (
              <span className="flex items-center gap-1 text-xs text-warning">
                <WifiOff size={14} />
                {pending} <T source="pending sync" />{" "}
              </span>
            )}
            <Button
              variant="ghost"
              size="icon"
              title={t("Log out")}
              aria-label={t("Log out")}
              onClick={logout}
            >
              <LogOut />
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-5 py-10">
        {syncError && (
          <div
            role="alert"
            className="mb-6 rounded-md border border-warning bg-warning-soft p-4 text-sm"
          >
            <p className="font-semibold">
              <T source="An offline scan was not accepted." />
            </p>
            <p className="mt-1">
              <T>{syncError}</T>
            </p>
            <p className="mt-1">
              <T source="The scan remains stored on this device and stays pending until the server accepts it." />{" "}
            </p>
            <Button className="mt-3" variant="outline" size="sm" onClick={retryRejectedScans}>
              <T source="Retry rejected scans" />{" "}
            </Button>
          </div>
        )}
        <Link
          to={backToScreening ? "/app/scan" : "/app/children"}
          search={backToScreening ? {} : { register: false, age_category: backAgeCategory }}
          className="mb-8 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary"
        >
          <ArrowLeft size={16} />
          {t(" ")}
          {backToScreening
            ? t("Back to screening")
            : backAgeCategory === "adult"
              ? t("All adults")
              : t("All children")}
        </Link>
        <h1 className="mb-8 font-display text-3xl font-semibold">{title}</h1>
        {children}
      </main>
    </div>
  );
}
