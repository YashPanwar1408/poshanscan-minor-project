import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LanguageToggle } from "@/components/LanguageToggle";
import { ThemeToggle } from "@/components/ThemeToggle";
import type * as tsTypes from "typescript";
import { PortalPrefsProvider, usePortalPrefs } from "@/lib/portal-prefs";
import {
  formatDate,
  formatNumber,
  normalizeTranslationKey,
  t,
  translate,
  translationKeys,
} from "@/lib/i18n";

function LocalizedContent() {
  const { language } = usePortalPrefs();
  void language;
  return (
    <>
      <h1>{t("Who is being screened?")}</h1>
      <p>{t("Search {category} by name…", { category: "बच्चे" })}</p>
    </>
  );
}

function Harness() {
  return (
    <PortalPrefsProvider>
      <LanguageToggle />
      <ThemeToggle />
      <LocalizedContent />
    </PortalPrefsProvider>
  );
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  document.documentElement.lang = "";
  document.documentElement.classList.remove("dark", "high-contrast");
});

describe("English and Hindi preferences", () => {
  it("defaults to English and switches to Hindi immediately while persisting the choice", async () => {
    render(<Harness />);

    expect(screen.getByRole("heading", { name: "Who is being screened?" })).toBeInTheDocument();
    const language = screen.getByRole("combobox", { name: "Language" });
    fireEvent.change(language, { target: { value: "hi" } });

    expect(
      screen.getByRole("heading", { name: translate("Who is being screened?", "hi") }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(t("Search {category} by name…", { category: "बच्चे" })),
    ).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("poshanscan_portal_prefs") || "{}").language).toBe("hi");
    await waitFor(() => expect(document.documentElement.lang).toBe("hi-IN"));
  });

  it("restores the saved language after a fresh mount and keeps theme independent", async () => {
    localStorage.setItem(
      "poshanscan_portal_prefs",
      JSON.stringify({ language: "hi", theme: "light" }),
    );
    const view = render(<Harness />);

    expect(
      screen.getByRole("heading", { name: translate("Who is being screened?", "hi") }),
    ).toBeInTheDocument();
    const theme = screen.getByRole("button", { name: translate("Switch to dark mode", "hi") });
    fireEvent.click(theme);

    expect(theme).toHaveAttribute("aria-pressed", "true");
    expect(JSON.parse(localStorage.getItem("poshanscan_portal_prefs") || "{}").language).toBe("hi");
    await waitFor(() => expect(document.documentElement.classList.contains("dark")).toBe(true));

    view.unmount();
    render(<Harness />);
    expect(screen.getByRole("combobox", { name: translate("Language", "hi") })).toHaveValue("hi");
  });

  it("formats dates and numbers using the selected Indian locale", () => {
    localStorage.setItem("poshanscan_portal_prefs", JSON.stringify({ language: "hi" }));
    expect(formatNumber(123456.7)).toBe(new Intl.NumberFormat("hi-IN").format(123456.7));
    expect(formatDate("2026-11-24", { dateStyle: "medium", timeZone: "UTC" })).toBe(
      new Intl.DateTimeFormat("hi-IN", { dateStyle: "medium", timeZone: "UTC" }).format(
        new Date("2026-11-24"),
      ),
    );
  });
});

describe("static translation coverage", () => {
  it("provides Hindi entries for every literal translation key", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const ts = await import("typescript");
    const sourceRoot = path.resolve(process.cwd(), "src");
    const keys = new Set(translationKeys());
    const referenced = new Map<string, string>();

    function files(directory: string): string[] {
      return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const fullPath = path.join(directory, entry.name);
        if (entry.isDirectory()) return files(fullPath);
        if (
          !/\.(tsx|ts)$/.test(entry.name) ||
          entry.name.endsWith(".test.ts") ||
          entry.name.endsWith(".test.tsx")
        )
          return [];
        return [fullPath];
      });
    }

    function collectKey(node: tsTypes.Node, file: string, line: number) {
      if (ts.isConditionalExpression(node)) {
        collectKey(node.whenTrue, file, line);
        collectKey(node.whenFalse, file, line);
        return;
      }
      if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
        const key = normalizeTranslationKey(node.text);
        if (key) referenced.set(key, `${path.relative(sourceRoot, file)}:${line}`);
      }
    }

    for (const file of files(sourceRoot)) {
      if (file.endsWith(`${path.sep}lib${path.sep}i18n.tsx`)) continue;
      const source = ts.createSourceFile(
        file,
        fs.readFileSync(file, "utf8"),
        ts.ScriptTarget.Latest,
        true,
        file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
      );
      function visit(node: tsTypes.Node) {
        const line = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
        if (ts.isJsxAttribute(node) && node.name.getText(source) === "source" && node.initializer) {
          const value = ts.isJsxExpression(node.initializer)
            ? node.initializer.expression
            : node.initializer;
          if (value) collectKey(value, file, line);
        }
        if (
          ts.isCallExpression(node) &&
          ts.isIdentifier(node.expression) &&
          node.expression.text === "t" &&
          node.arguments[0]
        ) {
          collectKey(node.arguments[0], file, line);
        }
        ts.forEachChild(node, visit);
      }
      visit(source);
    }

    const missing = [...referenced].filter(([key]) => !keys.has(key));
    expect(missing.map(([key, where]) => `${where} ${key}`)).toEqual([]);
  });
});
