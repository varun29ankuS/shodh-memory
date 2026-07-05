import path from "path";
import { describe, expect, it } from "vitest";
import { resolvePackageVersion } from "../version";

function reader(files: Record<string, string>) {
  return (filePath: string): string => {
    const normalized = path.normalize(filePath);
    const content = files[normalized];
    if (content === undefined) {
      throw new Error(`missing test file: ${normalized}`);
    }
    return content;
  };
}

describe("resolvePackageVersion", () => {
  it("reads the package version from the published dist layout", () => {
    const root = path.join("pkg", "node_modules", "@shodh", "memory-mcp");
    const baseDir = path.join(root, "dist");

    expect(
      resolvePackageVersion(
        baseDir,
        reader({
          [path.normalize(path.join(root, "package.json"))]:
            '{"version":"0.2.0"}',
        }),
      ),
    ).toBe("0.2.0");
  });

  it("reads the package version from the development layout", () => {
    const root = path.join("repo", "mcp-server");

    expect(
      resolvePackageVersion(
        root,
        reader({
          [path.normalize(path.join(root, "package.json"))]:
            '{"version":"0.2.1"}',
        }),
      ),
    ).toBe("0.2.1");
  });

  it("falls back to unknown when no package version can be resolved", () => {
    expect(resolvePackageVersion("repo/mcp-server", reader({}))).toBe("unknown");
  });
});
