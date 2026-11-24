import { formatNumber, T, t } from "@/lib/i18n";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState, type FormEvent } from "react";
import { ArrowRight, Plus, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RequireRole, WorkerShell } from "@/components/Screen";
import { api, childSchema, errorMessage, type Child } from "@/lib/api";
export const Route = createFileRoute("/app/children")({
  validateSearch: (search: Record<string, unknown>) => ({
    register: search["register"] === true,
    age_category:
      search["age_category"] === "adult"
        ? ("adult" as const)
        : search["age_category"] === "child_6_59m"
          ? ("child_6_59m" as const)
          : undefined,
  }),
  head: () => ({
    meta: [
      { get title() { return t("Children — PoshanScan"); } },
      { name: "description", get content() { return t("Find or register a child for nutrition screening."); } },
      { property: "og:title", get content() { return t("Children — PoshanScan"); } },
      { property: "og:description", get content() { return t("Find or register a child for nutrition screening."); } },
      { property: "og:type", get content() { return t("website"); } },
      { name: "twitter:card", get content() { return t("summary"); } },
    ],
  }),
  component: () => (
    <RequireRole allowed={["worker"]}>
      <Children />
    </RequireRole>
  ),
});
function Children() {
  const { register, age_category } = Route.useSearch();
  const isAdult = age_category === "adult";
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<Child[]>([]);
  const [open, setOpen] = useState(register);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    name: "",
    dob: "",
    gender: "F" as "M" | "F" | "O",
    guardian_name: "",
    village: "",
  });
  useEffect(() => {
    let active = true;
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const { data } = await api.get("/children/search", {
          params: {
            q: query.trim().slice(0, 100),
            age_category,
          },
        });
        if (active) {
          setItems(Array.isArray(data) ? data : data.children || []);
          setError("");
        }
      } catch (e) {
        if (active) setError(errorMessage(e));
      } finally {
        if (active) setLoading(false);
      }
    }, 300);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query, age_category]);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    const valid = childSchema.safeParse(form);
    if (!valid.success) {
      setError(valid.error.issues[0]?.message || "Please check the details.");
      return;
    }
    setSaving(true);
    try {
      const { data } = await api.post("/children", {
        ...valid.data,
        age_category,
      });
      setItems((prev) => [data, ...prev]);
      setOpen(false);
      setForm({ name: "", dob: "", gender: "F", guardian_name: "", village: "" });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  }
  return (
    <WorkerShell title={t(isAdult ? "Adults" : "Children")} backToScreening>
      <div className="max-w-4xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-muted-foreground">
            {isAdult ? t("Find an adult to start a new screening.") : t("Find a child to start a new screening.")}
          </p>
          <Button onClick={() => setOpen((v) => !v)}>
            <Plus /> {" "}<T source="Register new" />{" "}{isAdult ? t("adult") : t("child")}
          </Button>
        </div>
        {open && (
          <form onSubmit={submit} className="panel mt-7 p-6">
            <h2 className="text-xl font-semibold"><T source="Register new" />{" "}{isAdult ? t("adult") : t("child")}</h2>
            <div className="mt-6 grid gap-5 sm:grid-cols-2">
              {(
                [
                  ["name", isAdult ? "Adult name" : "Child name", "text"],
                  ["dob", "Date of birth", "date"],
                  ["guardian_name", "Guardian name", "text"],
                  ["village", "Village", "text"],
                ] as const
              ).map(([key, label, type]) => (
                <label key={key} className="text-sm font-medium">
                  {label}
                  <input
                    className="field mt-2"
                    type={type}
                    value={form[key]}
                    onChange={(e) => setForm((v) => ({ ...v, [key]: e.target.value }))}
                    required
                    maxLength={100}
                  />
                </label>
              ))}
              <label className="text-sm font-medium">
                <T source="Gender" />{" "}<select
                  className="field mt-2"
                  value={form.gender}
                  onChange={(e) =>
                    setForm((v) => ({ ...v, gender: e.target.value as "M" | "F" | "O" }))
                  }
                >
                  <option value="F"><T source="Female" /></option>
                  <option value="M"><T source="Male" /></option>
                  <option value="O"><T source="Other" /></option>
                </select>
              </label>
            </div>
            <div className="mt-6 flex gap-2">
              <Button type="submit" disabled={saving}>
                {saving ? t("Saving…") : `Save ${isAdult ? "adult" : "child"}`}
              </Button>
              <Button type="button" variant="outline" onClick={() => setOpen(false)}>
                <T source="Cancel" />{" "}</Button>
            </div>
          </form>
        )}
        <div className="relative mt-8">
          <Search
            size={19}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <input
            className="field pl-12"
            aria-label={t("Search children")}
            placeholder={t(isAdult ? "Search adults by name…" : "Search children by name…")}
            value={query}
            maxLength={100}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        {error && (
          <p role="alert" className="mt-5 rounded-md bg-danger-soft p-4 text-sm text-danger">
            <T>{error}</T>
          </p>
        )}
        <div className="mt-6 overflow-hidden rounded-md border border-border bg-card">
          {loading ? (
            <p className="p-6 text-muted-foreground"><T source="Searching…" /></p>
          ) : items.length ? (
            items.map((item) => {
              const scanId = item.latest_scan_id;
              return (
                <div
                  key={item.id}
                  className="flex items-center justify-between gap-4 border-b border-border p-5 last:border-0 hover:bg-secondary/50"
                >
                  <Link
                    to={scanId ? "/app/result/$scanId" : "/app/capture/$childId"}
                    params={scanId ? { scanId } : { childId: item.id }}
                    search={scanId ? {} : { age_category, rescan: false }}
                    className="min-w-0 flex-1"
                  >
                    <p className="font-semibold">{item.name}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {item.village} · {item.guardian_name}
                    </p>
                    {scanId && (
                      <p className="mt-2 text-xs font-medium text-primary">
                        <T source="Already screened ·" />{" "}
                        {item.risk_band
                          ? t(item.risk_band.toUpperCase() === "NORMAL" ? "Normal" : item.risk_band)
                          : t("View report")}{" "}·{t(" ")}
                        {item.muac_estimate_mm
                          ? `${formatNumber(item.muac_estimate_mm, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} mm`
                          : t("Open report")}
                      </p>
                    )}
                  </Link>
                  <div className="flex shrink-0 items-center gap-2">
                    {scanId && (
                      <Link
                        to="/app/capture/$childId"
                        params={{ childId: item.id }}
                        search={{ age_category, rescan: true }}
                        className="rounded-md border border-border px-3 py-2 text-xs font-semibold hover:bg-secondary"
                      >
                        <T source="Screen again" />{" "}</Link>
                    )}
                    <ArrowRight size={18} className="text-primary" />
                  </div>
                </div>
              );
            })
          ) : (
            <p className="p-8 text-center text-muted-foreground">
              <T source="No" />{" "}{isAdult ? t("adults") : t("children")} {" "}<T source="found. Register an" />{" "}{isAdult ? t("adult") : t("child")} {" "}<T source="to get started." />{" "}</p>
          )}
        </div>
      </div>
    </WorkerShell>
  );
}
