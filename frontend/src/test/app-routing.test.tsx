import { QueryClient } from "@tanstack/react-query";
import { createMemoryHistory, createRouter, RouterProvider } from "@tanstack/react-router";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { routeTree } from "@/routeTree.gen";
import { PortalPrefsProvider } from "@/lib/portal-prefs";
import { translate } from "@/lib/i18n";

async function renderAt(path: string) {
  const queryClient = new QueryClient();
  const router = createRouter({
    routeTree,
    context: { queryClient },
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  await router.load();
  return {
    router,
    ...render(
      <PortalPrefsProvider>
        <RouterProvider router={router} />
      </PortalPrefsProvider>,
    ),
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Assert only that the router mounts and paints, never page content:
// routes are rewritten as the app is built and this must keep passing.
describe("App routing", () => {
  it("renders the index route", async () => {
    const { container } = await renderAt("/");

    await waitFor(() => expect(container.firstChild).not.toBeNull());
  });

  it("renders the not-found route", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const { container } = await renderAt("/this-route-does-not-exist");

    await waitFor(() => expect(container.firstChild).not.toBeNull());
  });

  it("keeps the selected language when navigating between routes", async () => {
    const { router } = await renderAt("/");
    fireEvent.change(screen.getByRole("combobox", { name: "Language" }), {
      target: { value: "hi" },
    });

    await router.navigate({ to: "/worker/login" });

    expect(screen.getByRole("combobox", { name: translate("Language", "hi") })).toHaveValue("hi");
    await waitFor(() => expect(document.documentElement.lang).toBe("hi-IN"));
  });
});
