import { Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, LogOut, WifiOff, UserRound } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Brand } from "./Brand";
import { useEffect, useState, type ReactNode } from "react";
import { listQueued, deleteQueued } from "@/lib/offline";
import { api } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";
export function RequireRole({ allowed, children }: { allowed: string[]; children: ReactNode }) {
  const { user, ready } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (ready && (!user || !allowed.includes(user.role))) navigate({ to: "/login", replace: true });
  }, [ready, user, allowed, navigate]);
  if (!ready)
    return (
      <div className="min-h-screen grid place-items-center text-muted-foreground">Loading…</div>
    );
  if (!user || !allowed.includes(user.role)) return null;
  return <>{children}</>;
}
export function WorkerShell({
  children,
  title,
  backToScreening = false,
}: {
  children: ReactNode;
  title: string;
  backToScreening?: boolean;
}) {
  const { logout, user } = useAuth();
  const [profileOpen, setProfileOpen] = useState(false);
  const [pending, setPending] = useState(0);
  useEffect(() => {
    let active = true;
    async function sync() {
      try {
        const scans = await listQueued();
        if (!active) return;
        setPending(scans.length);
        if (!navigator.onLine || !scans.length) return;
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
        await api.post("/sync", body);
        for (const scan of scans) await deleteQueued(scan.client_scan_id);
        if (active) setPending(0);
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
  }, []);
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
          <Brand />
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <div className="relative">
              <Button
                variant="outline"
                onClick={() => setProfileOpen((value) => !value)}
                aria-expanded={profileOpen}
                title="Worker profile"
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
                {pending} pending sync
              </span>
            )}
            <Button
              variant="ghost"
              size="icon"
              title="Log out"
              aria-label="Log out"
              onClick={logout}
            >
              <LogOut />
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-5 py-10">
        <Link
          to={backToScreening ? "/app/scan" : "/app/children"}
          search={backToScreening ? {} : { register: false }}
          className="mb-8 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary"
        >
          <ArrowLeft size={16} /> {backToScreening ? "Back to screening" : "All children"}
        </Link>
        <h1 className="mb-8 font-display text-3xl font-semibold">{title}</h1>
        {children}
      </main>
    </div>
  );
}
