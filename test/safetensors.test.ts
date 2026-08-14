import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parseSafetensors, parseSafetensorsIndex, summarizeSafetensors } from "../src/safetensors.js";
import { buildShard, buildTinySafetensors } from "./fixtures.js";

let dir: string;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "gguf-mcp-st-test-"));
});

afterAll(() => rm(dir, { recursive: true, force: true }));

describe("parseSafetensors", () => {
  it("parses header: tensors, params, dtypes, metadata, data bytes", async () => {
    const p = join(dir, "tiny.safetensors");
    await writeFile(p, buildTinySafetensors());
    const m = await parseSafetensors(p);
    expect(m.paramCount).toBe(10 * 20 + 5);
    expect(m.weightsBytes).toBe(420);
    expect(m.dtypes).toEqual({ F16: 1, F32: 1 });
    expect(m.metadata).toEqual({ format: "pt" });
    expect(m.tensors.map((t) => t.name).sort()).toEqual(["emb", "w"]);
  });

  it("rejects files whose header is not JSON", async () => {
    const p = join(dir, "bad.safetensors");
    const b = Buffer.alloc(20);
    b.writeBigUInt64LE(10n, 0); // length says 10, bytes are zeros
    await writeFile(p, b);
    await expect(parseSafetensors(p)).rejects.toThrow(/not valid JSON/);
  });

  it("rejects absurd header lengths", async () => {
    const p = join(dir, "huge.safetensors");
    const b = Buffer.alloc(16);
    b.writeBigUInt64LE(BigInt(1 << 30), 0);
    await writeFile(p, b);
    await expect(parseSafetensors(p)).rejects.toThrow(/sanity cap/);
  });
});

describe("parseSafetensorsIndex", () => {
  it("aggregates across shards", async () => {
    await writeFile(join(dir, "model-00001-of-00002.safetensors"), buildShard("layer.0", [100, 10]));
    await writeFile(join(dir, "model-00002-of-00002.safetensors"), buildShard("layer.1", [50]));
    const indexPath = join(dir, "model.safetensors.index.json");
    await writeFile(
      indexPath,
      JSON.stringify({
        metadata: { total_size: 4200 },
        weight_map: {
          "layer.0": "model-00001-of-00002.safetensors",
          "layer.1": "model-00002-of-00002.safetensors",
        },
      }),
    );
    const m = await parseSafetensorsIndex(indexPath);
    expect(m.shards).toBe(2);
    expect(m.paramCount).toBe(100 * 10 + 50);
    expect(m.weightsBytes).toBe(4 * (1000 + 50));
    const s = summarizeSafetensors(m);
    expect(s.parameters).toBe("1.1K");
    expect(s.shards).toBe(2);
  });

  it("rejects an empty weight_map", async () => {
    const p = join(dir, "empty.safetensors.index.json");
    await writeFile(p, JSON.stringify({ weight_map: {} }));
    await expect(parseSafetensorsIndex(p)).rejects.toThrow(/empty weight_map/);
  });
});
