import { T, t } from "@/lib/i18n";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight, ClipboardList, LockKeyhole, Phone, Users } from "lucide-react";
import { Brand } from "@/components/Brand";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { errorMessage, loginSchema, signupSchema } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";
import { LanguageToggle } from "@/components/LanguageToggle";

type LoginRole = "worker" | "supervisor";

const roleCopy = {
  worker: {
    get eyebrow() { return t("FIELD WORKER ACCESS"); },
    get title() { return t("Log in as a worker"); },
    get description() { return t("Start guided screenings, capture photos, and follow up with children."); },
    get submit() { return t("Open worker workspace"); },
    get help() { return t("Worker access is provided by your program administrator."); },
    get panelTitle() { return t("Capture clearly. Screen consistently. Support better care."); },
    get panelDescription() { return t("Use guided photo capture to record reliable screening information in the field."); },
    get nextStep() { return t("Start a screening"); },
    get alternate() { return t("Supervisor login"); },
    alternateTo: "/supervisor/login" as const,
    icon: Phone,
  },
  supervisor: {
    get eyebrow() { return t("PROGRAM SUPERVISOR ACCESS"); },
    get title() { return t("Log in as a supervisor"); },
    get description() { return t("Review program progress, spot follow-up needs, and support your field team."); },
    get submit() { return t("Open supervisor dashboard"); },
    get help() { return t("Supervisor access is managed by your program administrator."); },
    get panelTitle() { return t("See program progress. Coordinate timely follow-up."); },
    get panelDescription() { return t("Review screening trends and help field teams connect children with the right care."); },
    get nextStep() { return t("Review program activity"); },
    get alternate() { return t("Worker login"); },
    alternateTo: "/worker/login" as const,
    icon: ClipboardList,
  },
} satisfies Record<LoginRole, object>;

export function RoleLogin({ role }: { role: LoginRole }) {
  const copy = roleCopy[role];
  const { login, registerWorker } = useAuth();
  const navigate = useNavigate();
  const [signup, setSignup] = useState(false);
  const [phone, setPhone] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [village, setVillage] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    const checked = signup
      ? signupSchema.safeParse({ name, phone, email, village, password, confirmPassword })
      : loginSchema.safeParse({ phone, password });
    if (!checked.success) {
      setError(checked.error.issues[0]?.message || t("Please check your details."));
      return;
    }

    setBusy(true);
    try {
      const user = signup
        ? await registerWorker(
            checked.data.phone,
            checked.data.password,
            String("name" in checked.data ? checked.data.name : name),
            email,
            village,
          )
        : await login(checked.data.phone, checked.data.password);
      if (user.role !== role) {
        setError(
          t("This account is not a {role} account. Please use the {userRole} login.", {
            role: t(role),
            userRole: t(user.role),
          }),
        );
        return;
      }
      navigate({ to: role === "worker" ? "/app/scan" : "/supervisor/login", replace: true });
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const RoleIcon = role === "worker" ? Phone : Users;

  return (
    <div className={`role-login role-login-${role} grid min-h-[100svh] lg:grid-cols-2`}>
      <div className="flex max-h-[100svh] min-h-0 flex-col overflow-y-auto px-6 py-5 sm:px-12 lg:px-20 lg:py-7">
        <div className="flex items-center justify-between gap-4">
          <Brand />
          <div className="flex items-center gap-2">
            <LanguageToggle compact />
            <ThemeToggle />
          </div>
        </div>
        <div className={`mx-auto flex w-full max-w-md flex-1 flex-col py-7 lg:py-10 ${signup ? "justify-start" : "justify-center"}`}>
          <Link to="/" className="mb-6 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary">
            <ArrowLeft size={16} /> {" "}<T source="Back to home" />{" "}</Link>
          <p className="eyebrow">{copy.eyebrow}</p>
          <h1 className="mt-3 font-display text-4xl font-semibold">
            {signup ? t("Create a worker account") : copy.title}
          </h1>
          <p className="mt-3 text-muted-foreground">
            {signup ? t("Create your account to start field screenings.") : copy.description}
          </p>

          <form onSubmit={submit} className={`mt-7 ${signup ? "space-y-3.5" : "space-y-5"}`}>
            <label className="block text-sm font-semibold">
              {signup && "Full name"}
              {signup && (
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  autoComplete="name"
                  placeholder={t("Enter your full name")}
                  className="field mt-2"
                />
              )}
            </label>
            {signup && (
              <>
                <label className="block text-sm font-semibold">
                  <T source="Email address" />{" "}<input
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    autoComplete="email"
                    placeholder={t("Enter your email (optional)")}
                    className="field mt-2"
                    type="email"
                  />
                </label>
                <label className="block text-sm font-semibold">
                  <T source="Village or field area" />{" "}<input
                    value={village}
                    onChange={(event) => setVillage(event.target.value)}
                    placeholder={t("Enter your working area")}
                    className="field mt-2"
                  />
                </label>
              </>
            )}
            <label className="block text-sm font-semibold">
              <T source="Phone number" />{" "}<div className="relative mt-2">
                <RoleIcon size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={phone}
                  onChange={(event) => setPhone(event.target.value)}
                  autoComplete="tel"
                  inputMode="tel"
                  placeholder={t("Enter your phone number")}
                  className="field pl-12"
                />
              </div>
            </label>
            <label className="block text-sm font-semibold">
              <T source="Password" />{" "}<div className="relative mt-2">
                <LockKeyhole size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  placeholder={t("Enter your password")}
                  className="field pl-12"
                />
              </div>
            </label>
            {signup && (
              <label className="block text-sm font-semibold">
                <T source="Confirm password" />{" "}<div className="relative mt-2">
                  <LockKeyhole size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    autoComplete="new-password"
                    placeholder={t("Re-enter your password")}
                    className="field pl-12"
                  />
                </div>
              </label>
            )}
            {error && (
              <p role="alert" className="rounded-md bg-danger-soft p-3 text-sm text-danger">
                <T>{error}</T>
              </p>
            )}
            <Button type="submit" size="lg" className="w-full" disabled={busy}>
              {busy ? (signup ? t("Creating account…") : t("Signing in…")) : signup ? t("Create worker account") : copy.submit}{t(" ")}
              <ArrowRight />
            </Button>
          </form>

          {role === "worker" && (
            <p className="mt-5 text-center text-sm text-muted-foreground">
              {signup ? t("Already have an account?") : t("New worker?")}{t(" ")}
              <button
                type="button"
                className="font-medium text-primary hover:underline"
                onClick={() => {
                  setSignup((value) => !value);
                  setError("");
                }}
              >
                {signup ? t("Log in") : t("Sign up")}
              </button>
            </p>
          )}

          <div className="mt-7 rounded-md border border-border bg-card p-4 text-sm">
            <p className="font-semibold"><T source="Looking for the other workspace?" /></p>
            <Link to={copy.alternateTo} className="mt-2 inline-flex items-center gap-2 font-medium text-primary hover:underline">
              {copy.alternate} <ArrowRight size={15} />
            </Link>
          </div>
          <p className="mt-6 text-center text-xs text-muted-foreground">{copy.help}</p>
        </div>
      </div>

      <aside className="relative hidden overflow-hidden bg-primary lg:block">
        <div className="absolute inset-0 flex flex-col justify-end p-16 text-primary-foreground">
          <div className="mb-7 flex size-16 items-center justify-center rounded-md bg-primary-foreground/10">
            {role === "worker" ? <LockKeyhole size={30} /> : <Users size={30} />}
          </div>
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-primary-foreground/70">{copy.nextStep}</p>
          <h2 className="max-w-lg font-display text-4xl font-semibold leading-tight">{copy.panelTitle}</h2>
          <p className="mt-5 max-w-md leading-relaxed text-primary-foreground/75">{copy.panelDescription}</p>
        </div>
      </aside>
    </div>
  );
}
