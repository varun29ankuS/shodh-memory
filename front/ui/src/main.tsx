import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { App } from "./app/App";
import { applyStoredGround } from "./lib/ground";

// Before the first paint, and outside React: the ground is a document-level
// attribute, and a destination that mounts on paper and corrects itself to
// night one frame later is a flash the person did not ask for. index.html
// ships `class="dark"`; this is what reconciles it with the stored choice.
applyStoredGround();

const root = document.getElementById("root");
if (!root) throw new Error("index.html is missing #root");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
