import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import { getRouter } from "./router";
import { PortalPrefsProvider } from "./lib/portal-prefs";
import "./styles.css";

const router = getRouter();

const rootElement = document.getElementById("root");
if (rootElement && !rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <PortalPrefsProvider>
        <RouterProvider router={router} />
      </PortalPrefsProvider>
    </React.StrictMode>,
  );
}
