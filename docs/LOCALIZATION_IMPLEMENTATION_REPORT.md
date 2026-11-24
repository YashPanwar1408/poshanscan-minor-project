# PoshanScan localization implementation report

**Completed:** 24 November 2026

**Languages:** English (`en`) and Hindi (`hi`)

## Implementation

- Added a centralized English/Hindi catalog and reusable `translate`, `t`,
  `plural`, `formatNumber`, and `formatDate` helpers in
  `frontend/src/lib/i18n.tsx`. Unknown strings safely fall back to their source
  English; missing Hindi keys log a development warning.
- Added an accessible language selector next to the theme control. It is used
  on the landing page, worker and supervisor sign-in screens, worker shell, and
  supervisor shell. Changing language updates the mounted page without a
  reload.
- The preference provider starts from saved state and stores language and
  theme together in `localStorage` under `poshanscan_portal_prefs`. English is
  the default. The document `lang` value changes between `en-IN` and `hi-IN`.
- Displayed dates and numbers use `en-IN` or `hi-IN` formatting. Internal
  values remain numeric and API payloads are not localized.
- Known backend error codes are mapped to localized user-facing copy. Unknown
  backend messages retain the safe English fallback; diagnostic API data and
  user-entered content are not translated.
- Localized authentication, registration, capture and scan results, child
  records, common UI primitives, notifications and errors, landing-page copy,
  supervisor navigation, dashboards, flagged review, reports, trends, workers,
  locations, and API-inspector labels. Technical names such as MUAC and ArUco
  remain recognizable.
- No API contract, authentication rule, MUAC calculation, marker calibration,
  clinical threshold, or ML behavior was changed for localization.

## Coverage and verification

- `npm test -- --reporter=dot`: **44 passed** across 3 test files. The suite
  checks default language, live switching, saved preference, route navigation,
  independent theme behavior, locale formatting, and a static scan that every
  literal `t()`/`T` key has a Hindi entry.
- A separate TSX scan found no untranslated plain JSX text or literal
  accessibility/title/placeholder strings. Its only English result was the
  API-inspector sentence already wrapped in `<T>` for translation.
- `npx tsc --noEmit`: passed.
- `npm run build`: passed (Vite production build).
- `npm run lint`: passed with 0 errors and 17 Fast Refresh warnings. These
  warnings concern modules that export both React components and helpers; they
  are not localization errors.

## Files changed for localization

```text
frontend/src/components/Brand.tsx
frontend/src/components/LanguageToggle.tsx
frontend/src/components/RoleLogin.tsx
frontend/src/components/Screen.tsx
frontend/src/components/ThemeToggle.tsx
frontend/src/components/UtilityBar.tsx
frontend/src/components/ui/breadcrumb.tsx
frontend/src/components/ui/calendar.tsx
frontend/src/components/ui/carousel.tsx
frontend/src/components/ui/chart.tsx
frontend/src/components/ui/dialog.tsx
frontend/src/components/ui/pagination.tsx
frontend/src/components/ui/sheet.tsx
frontend/src/components/ui/sidebar.tsx
frontend/src/lib/api.ts
frontend/src/lib/i18n.tsx
frontend/src/lib/portal-prefs.tsx
frontend/src/main.tsx
frontend/src/routes/__root.tsx
frontend/src/routes/app/capture.$childId.tsx
frontend/src/routes/app/children.tsx
frontend/src/routes/app/result.$scanId.tsx
frontend/src/routes/app/scan.tsx
frontend/src/routes/index.tsx
frontend/src/routes/supervisor.dev.api.tsx
frontend/src/routes/supervisor.flagged.tsx
frontend/src/routes/supervisor.locations.tsx
frontend/src/routes/supervisor.login.tsx
frontend/src/routes/supervisor.overview.tsx
frontend/src/routes/supervisor.reports.tsx
frontend/src/routes/supervisor.trends.tsx
frontend/src/routes/supervisor.workers.tsx
frontend/src/routes/worker.login.tsx
frontend/src/supervisor/api/adapters.ts
frontend/src/supervisor/api/client.ts
frontend/src/supervisor/components/AppShell.tsx
frontend/src/supervisor/components/BandBadge.tsx
frontend/src/supervisor/components/CommandPalette.tsx
frontend/src/supervisor/components/KpiCard.tsx
frontend/src/supervisor/components/Primitives.tsx
frontend/src/supervisor/pages/ApiInspectorPage.tsx
frontend/src/supervisor/pages/FlaggedPage.tsx
frontend/src/supervisor/pages/LocationsPage.tsx
frontend/src/supervisor/pages/LoginPage.tsx
frontend/src/supervisor/pages/OverviewPage.tsx
frontend/src/supervisor/pages/ReportsPage.tsx
frontend/src/supervisor/pages/TrendsPage.tsx
frontend/src/supervisor/pages/WorkersPage.tsx
frontend/src/test/app-routing.test.tsx
frontend/src/test/i18n.test.tsx
```

## Limits

The automated check covers source translation keys and common preference
behavior; it is not a human review of every Hindi sentence. No browser-based
desktop/mobile visual run was available. Unknown server text and arbitrary
user-provided names remain in their original language by design.
