import fs from "fs";
import path from "path";

type ReadTextFile = (filePath: string) => string;

function defaultReadTextFile(filePath: string): string {
  return fs.readFileSync(filePath, "utf-8");
}

export function resolvePackageVersion(
  baseDir: string,
  readTextFile: ReadTextFile = defaultReadTextFile,
): string {
  const candidates = [
    path.join(baseDir, "..", "package.json"),
    path.join(baseDir, "package.json"),
  ];

  for (const candidate of candidates) {
    try {
      const pkg = JSON.parse(readTextFile(candidate));
      if (typeof pkg.version === "string" && pkg.version) return pkg.version;
    } catch {
      // Try the next layout candidate.
    }
  }

  return "unknown";
}
