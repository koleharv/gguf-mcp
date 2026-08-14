import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { estimateVram, humanBytes, humanCount, parseGguf, summarizeGguf, type ArraySummary } from "../src/gguf.js";
import { buildTinyGguf, TINY_GGUF_WEIGHTS } from "./fixtures.js";

let dir: string;
let ggufPath: string;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "gguf-mcp-test-"));
  ggufPath = join(dir, "tiny.gguf");
  await writeFile(ggufPath, buildTinyGguf());
});

afterAll(() => rm(dir, { recursive: true, force: true }));

describe("parseGguf", () => {
  it("parses header, metadata, tensors, and params without touching tensor data", async () => {
    const m = await parseGguf(ggufPath);
    expect(m.version).toBe(3);
    expect(m.paramCount).toBe(64 * 64 + 64 * 100);
    expect(m.tensors).toEqual([
      { name: "blk.0.attn_q.weight", shape: [64, 64], type: "F32" },
      { name: "output.weight", shape: [64, 100], type: "F16" },
    ]);
    expect(m.metadata["general.architecture"]).toBe("llama");
    expect(m.metadata["llama.context_length"]).toBe(4096);
    expect(m.weightsBytes).toBe(TINY_GGUF_WEIGHTS);
  });

  it("summarizes large arrays instead of dumping them", async () => {
    const m = await parseGguf(ggufPath);
    const tokens = m.metadata["tokenizer.ggml.tokens"] as ArraySummary;
    expect(tokens.array).toBe(true);
    expect(tokens.count).toBe(7);
    expect(tokens.elementType).toBe("STRING");
    expect(tokens.sample).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("rejects non-GGUF files by magic", async () => {
    const bad = join(dir, "bad.gguf");
    await writeFile(bad, Buffer.from("PK\x03\x04 definitely a zip"));
    await expect(parseGguf(bad)).rejects.toThrow(/magic mismatch/);
  });

  it("rejects unsupported and byte-swapped versions", async () => {
    const v1 = buildTinyGguf();
    v1.writeUInt32LE(1, 4);
    const v1Path = join(dir, "v1.gguf");
    await writeFile(v1Path, v1);
    await expect(parseGguf(v1Path)).rejects.toThrow(/Unsupported GGUF version: 1/);

    const be = buildTinyGguf();
    be.writeUInt32BE(3, 4); // big-endian writer: version reads as 0x03000000
    const bePath = join(dir, "be.gguf");
    await writeFile(bePath, be);
    await expect(parseGguf(bePath)).rejects.toThrow(/Big-endian/);
  });

  it("rejects truncated files", async () => {
    const cut = buildTinyGguf().subarray(0, 40);
    const cutPath = join(dir, "cut.gguf");
    await writeFile(cutPath, cut);
    await expect(parseGguf(cutPath)).rejects.toThrow(/Unexpected end of file/);
  });
});

describe("summarizeGguf", () => {
  it("produces the human summary", async () => {
    const s = summarizeGguf(await parseGguf(ggufPath));
    expect(s).toMatchObject({
      format: "gguf",
      name: "TestLlama",
      architecture: "llama",
      parameters: "10.5K",
      quantization: "Q4_K_M",
      contextLength: 4096,
      tensorCount: 2,
      hasChatTemplate: false,
    });
  });
});

describe("estimateVram", () => {
  // layers=2, kv_heads=2, head_dim=64/4=16: KV = 2*2*ctx*2*16*2 bytes = 256*ctx
  it("computes weights + KV cache from metadata", async () => {
    const est = estimateVram(await parseGguf(ggufPath));
    expect(est.contextUsed).toBe(4096); // min(model ctx 4096, default cap 8192)
    expect(est.weights).toBe("128 B");
    expect(est.kvCache).toBe("1.0 MiB"); // 256 * 4096 = 1 MiB
  });

  it("respects an explicit context length", async () => {
    const est = estimateVram(await parseGguf(ggufPath), 1024);
    expect(est.contextUsed).toBe(1024);
    expect(est.kvCache).toBe("256.0 KiB");
  });

  it("degrades to weights-only when architecture metadata is missing", async () => {
    const m = await parseGguf(ggufPath);
    delete m.metadata["llama.block_count"];
    const est = estimateVram(m);
    expect(est.kvCache).toBeUndefined();
    expect(est.note).toMatch(/weights only/);
  });
});

describe("humanizers", () => {
  it("formats counts and bytes", () => {
    expect(humanCount(7_240_000_000)).toBe("7.24B");
    expect(humanCount(160_500_000)).toBe("160.5M");
    expect(humanBytes(4.2 * (1 << 30))).toBe("4.20 GiB");
    expect(humanBytes(512)).toBe("512 B");
  });
});
