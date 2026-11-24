import { formatDate, formatNumber, T, t } from "@/lib/i18n";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";
import { ArrowRight, AlertTriangle, CheckCircle, UserRound, UsersRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RequireRole, WorkerShell } from "@/components/Screen";
import { api, errorMessage, type ScanResult } from "@/lib/api";
import { getMuacBands, type MuacAgeCategory } from "@/lib/muac";

export const Route = createFileRoute("/app/result/$scanId")({
  head: () => ({
    meta: [
      { get title() { return t("Screening report — PoshanScan"); } },
      { name: "description", get content() { return t("Detailed MUAC screening report and recommended next steps."); } },
    ],
  }),
  component: () => (
    <RequireRole allowed={["worker"]}>
      <Result />
    </RequireRole>
  ),
});

const guidance = {
  child_6_59m: {
    NORMAL:
      "This screening does not indicate elevated risk on the child MUAC scale. Continue routine growth and nutrition monitoring under your local protocol. Encourage age-appropriate nutritious feeding and seek assessment if the child becomes unwell or concerns arise. Repeat screening at the interval recommended by your programme.",
    MAM:
      "This result indicates moderate screening risk and needs follow-up. Arrange assessment by a trained health worker to review the child's nutrition, health, and feeding. Provide counselling and any support recommended by your local programme. Schedule follow-up and repeat measurement according to the care protocol.",
    SAM:
      "This result indicates high screening risk and requires prompt clinical assessment. Refer the child urgently to an appropriate health facility or nutrition service under your local protocol. A trained clinician should assess for complications and determine the right treatment setting. If the child is very unwell or has danger signs, seek emergency care immediately.",
  },
  adult: {
    NORMAL:
      "This screening does not indicate elevated risk on the adult MUAC scale. Continue routine nutrition and health monitoring under your local protocol. Encourage a varied, adequate diet where possible. Repeat screening at the interval recommended by your programme or sooner if concerns arise.",
    MAM:
      "This result indicates moderate screening risk and should be followed up. Arrange assessment by a trained health worker to review nutrition status and possible contributing health or dietary factors. Follow local guidance for counselling, support, and any further assessment. Agree on a follow-up plan and repeat measurement as recommended.",
    SAM:
      "This result indicates high screening risk and needs prompt clinical assessment. Refer the adult to an appropriate health facility or nutrition service under your local protocol. A trained clinician should assess overall health and determine the appropriate care. If the person is acutely unwell, seek emergency care immediately.",
  },
} as const;

function Result() {
  const { scanId } = Route.useParams();
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const cached = sessionStorage.getItem(`poshanscan_result_${scanId}`);
    if (cached) {
      try {
        setResult(JSON.parse(cached));
      } catch {
        sessionStorage.removeItem(`poshanscan_result_${scanId}`);
      }
    }
    api
      .get<ScanResult>(`/scans/${encodeURIComponent(scanId)}`)
      .then(({ data }) => setResult(data))
      .catch((cause) => {
        if (!sessionStorage.getItem(`poshanscan_result_${scanId}`)) {
          setError(errorMessage(cause));
        }
      });
  }, [scanId]);

  return (
    <WorkerShell title={t("Screening report")} backAgeCategory={result?.age_category}>
      {result ? (
        <Report result={result} />
      ) : (
        <p className="text-muted-foreground"><T>{error || "Loading report…"}</T></p>
      )}
    </WorkerShell>
  );
}

function Report({ result }: { result: ScanResult }) {
  const category: MuacAgeCategory =
    result.age_category === "adult" ? "adult" : "child_6_59m";
  const bands = getMuacBands(category);
  const value = Number(result.muac_estimate_mm);
  const scaleMin = Math.min(bands.min, value);
  const scaleMax = Math.max(bands.max, value);
  const position = ((value - scaleMin) / (scaleMax - scaleMin)) * 100;
  const severeWidth = ((bands.severeBelow - scaleMin) / (scaleMax - scaleMin)) * 100;
  const moderateWidth = ((bands.moderateBelow - bands.severeBelow) / (scaleMax - scaleMin)) * 100;
  const normalWidth = 100 - severeWidth - moderateWidth;
  const subject = result.child ?? {};
  const subjectName = subject.name ?? result.child_name ?? result.name ?? result.child_id;
  const workerName = result.worker?.name ?? result.worker_name ?? "Signed-in field worker";
  const workerPhone = result.worker?.phone ?? result.worker_phone;
  const workerEmail = result.worker?.email;
  const workerVillage = result.worker?.village;
  const recordedAt = result.captured_at ?? result.created_at;
  const bandLabel =
    result.risk_band === "NORMAL"
      ? t("Normal")
      : result.risk_band === "MAM"
        ? t("Moderate risk (MAM)")
        : t("High risk (SAM)");

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="text-center">
        <p className="eyebrow"><T source="POSHANSCAN SCREENING REPORT" /></p>
        <h2 className="mt-2 font-display text-3xl font-semibold"><T source="MUAC assessment summary" /></h2>
        <p className="mt-2 text-sm text-muted-foreground">
          <T source="Review the measured value, applicable scale, and recommended follow-up." />{" "}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <InfoCard icon={<UserRound size={18} />} title={t(category === "adult" ? "Adult details" : "Child details")}>
          <Detail label={t("Name")} value={subjectName} />
          <Detail label={t("ID")} value={result.child_id} />
          <Detail label={t("Date of birth")} value={subject.dob ?? result.dob} />
          <Detail label={t("Gender")} value={subject.gender ?? result.gender} />
          <Detail label={t("Village")} value={subject.village ?? result.village} />
          <Detail label={t("Guardian")} value={subject.guardian_name ?? result.guardian_name} />
        </InfoCard>
        <InfoCard icon={<UsersRound size={18} />} title={t("Screening details")}>
          <Detail label={t("Field worker")} value={workerName} />
          <Detail label={t("Worker ID")} value={result.worker_id} />
          <Detail label={t("Worker phone")} value={workerPhone} />
          <Detail label={t("Worker email")} value={workerEmail} />
          <Detail label={t("Worker area")} value={workerVillage} />
          <Detail label={t("Scan ID")} value={result.scan_id} />
          <Detail label={t("Captured")} value={recordedAt ? formatDate(recordedAt, { dateStyle: "medium", timeStyle: "short" }) : undefined} />
          <Detail label={t("Reference marker")} value={`${formatNumber(result.reference_size_mm ?? 50)} mm ArUco`} />
          <Detail label={t("Cohort")} value={bands.label} />
        </InfoCard>
      </div>

      {result.image_url && (
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow"><T source="CAPTURED PHOTO" /></p>
          <img
            src={result.image_url}
            alt={t("Screening photo for {name}", { name: subjectName })}
            className="mt-4 max-h-[28rem] w-full rounded-md border border-border object-contain"
          />
        </section>
      )}

      <section className="panel p-6 sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="eyebrow"><T source="MEASUREMENT SCALE" /></p>
            <h3 className="mt-2 text-xl font-semibold">
              {category === "adult" ? t("Adult MUAC scale") : t("Child MUAC scale (6–59 months)")}
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">
              <T source="Thresholds: SAM &lt;" />{" "}{formatNumber(bands.severeBelow)} {" "}<T source="mm · MAM" />{" "}{formatNumber(bands.severeBelow)}<T source="–&lt;" />{" "}{formatNumber(bands.moderateBelow)} {" "}<T source="mm · Normal ≥" />{" "}{formatNumber(bands.moderateBelow)} {" "}<T source="mm" />{" "}</p>
          </div>
          <div className="text-right">
            <p className="text-xs uppercase tracking-wider text-muted-foreground"><T source="Received MUAC" /></p>
            <p className="font-display text-4xl font-semibold">{formatNumber(value, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} {" "}<T source="mm" /></p>
          </div>
        </div>
        <div className="relative mt-8 pt-7">
          <div className="absolute top-0 -translate-x-1/2 text-center" style={{ left: `${position}%` }}>
            <span className="rounded bg-foreground px-2 py-1 font-mono text-xs font-bold text-background">
              {formatNumber(value, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} {" "}<T source="mm" />{" "}</span>
            <div className="mx-auto h-2 w-0 border-x-4 border-t-4 border-x-transparent border-t-foreground" />
          </div>
          <div className="flex h-8 overflow-hidden rounded-md text-center text-xs font-bold text-white shadow-inner">
            <div style={{ width: `${severeWidth}%` }} className="flex items-center justify-center bg-rose-600">
              <T source="SAM (&lt;" />{formatNumber(bands.severeBelow)})
            </div>
            <div style={{ width: `${moderateWidth}%` }} className="flex items-center justify-center bg-amber-500 text-zinc-900">
              <T source="MAM" />{" "}</div>
            <div style={{ width: `${normalWidth}%` }} className="flex items-center justify-center bg-teal-700">
              <T source="Normal (≥" />{formatNumber(bands.moderateBelow)})
            </div>
          </div>
          <div className="mt-2 flex justify-between font-mono text-xs text-muted-foreground">
            <span>{formatNumber(scaleMin)} {" "}<T source="mm" /></span>
            <span>{formatNumber(bands.severeBelow)} {" "}<T source="mm" /></span>
            <span>{formatNumber(bands.moderateBelow)} {" "}<T source="mm" /></span>
            <span>{formatNumber(scaleMax)} {" "}<T source="mm" /></span>
          </div>
        </div>
      </section>

      <section className="panel p-6 sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-6">
          <div>
            <p className="eyebrow"><T source="SCREENING RESULT" /></p>
            <p className="mt-2 text-sm text-muted-foreground"><T source="Backend classification for the selected cohort" /></p>
          </div>
          <span className={`rounded-md px-4 py-2 text-sm font-bold ${result.risk_band === "NORMAL" ? "bg-success-soft text-success" : result.risk_band === "MAM" ? "bg-warning-soft text-warning" : "bg-danger-soft text-danger"}`}>
            {bandLabel}
          </span>
        </div>
        <div className="mt-6 grid gap-6 sm:grid-cols-3">
          <Metric label={t("Estimated MUAC")} value={`${formatNumber(value, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} mm`} />
          <Metric label={t("Model confidence")} value={`${formatNumber(Math.round(result.confidence_score * 100))}%`} />
          <Metric label={t("Applied category")} value={t(category === "adult" ? "Adult" : "Child")} />
        </div>
        <div className="mt-7 rounded-md border border-border bg-secondary/40 p-5">
          <h3 className="font-semibold"><T source="Recommended next step" /></h3>
          <p className="mt-2 leading-relaxed text-muted-foreground">
            <T>{guidance[category][result.risk_band] || t("Confirm this result with a trained health worker.")}</T>
          </p>
        </div>
        {result.confidence_score < 0.6 && (
          <p className="mt-5 flex items-center gap-2 rounded-md bg-warning-soft p-3 text-sm font-medium text-warning">
            <AlertTriangle size={18} /> {" "}<T source="Low confidence — repeat the scan and confirm with a standard MUAC tape." />{" "}</p>
        )}
        {result.risk_band === "NORMAL" && <p className="mt-5 flex items-center gap-2 text-sm text-success"><CheckCircle size={18} /> {" "}<T source="No elevated screening risk indicated by this scale." /></p>}
      </section>

      <p className="text-center text-sm text-muted-foreground">
        <T source="PoshanScan is a screening aid, not a medical diagnosis. Always confirm abnormal or low-confidence results with a trained health worker." />{" "}</p>
      <div className="flex justify-center">
        <div className="flex flex-wrap justify-center gap-3">
          <Button asChild size="lg">
            <Link to="/app/capture/$childId" params={{ childId: result.child_id }} search={{ age_category: category, rescan: true }}>
              <T source="Screen again" />{" "}<ArrowRight />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link to="/app/scan"><T source="Choose another person" /></Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

function InfoCard({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="panel p-6">
      <h3 className="flex items-center gap-2 font-semibold">{icon}{title}</h3>
      <div className="mt-5 divide-y divide-border">{children}</div>
    </section>
  );
}

function Detail({ label, value }: { label: string; value?: string | number | undefined }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value || t("Not provided")}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="mt-2 font-display text-2xl font-semibold">{value}</p>
    </div>
  );
}
