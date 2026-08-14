#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { open, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import {
  estimateVram,
  humanBytes,
  parseGguf,
  summarizeGguf,
  type GgufModel,
  type MetadataValue,
} from "./gguf.js";
import {
  parseSafetensors,
  parseSafetensorsIndex,
  summarizeSafetensors,
  type SafetensorsModel,
} from "./safetensors.js";

const server = new McpServer({ name: "gguf", version: "0.1.0" });

type ToolResult = { content: { type: "text"; text: string }[]; isError?: boolean };

function jsonResult(data: unknown): ToolResult {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

function guarded<A extends unknown[]>(fn: (...args: A) => Promise<ToolResult>) {
  return async (...args: A): Promise<ToolResult> => {
    try {
      return await fn(...args);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { content: [{ type: "text", text: msg }], isError: true };
    }
  };
}

type Model = GgufModel | SafetensorsModel;

// Accepts a .gguf / .safetensors / .safetensors.index.json file or a model
// directory. Extension-less files (e.g. Ollama blobs) are sniffed by magic.
async function loadModel(path: string): Promise<Model> {
  const s = await stat(path).catch(() => {
    throw new Error(`Path not found: ${path}`);
  });

  if (s.isDirectory()) {
    const entries = await readdir(path);
    const index = entries.find((e) => e.endsWith(".safetensors.index.json"));
    if (index) return parseSafetensorsIndex(join(path, index));
    const st = entries.filter((e) => e.endsWith(".safetensors"));
    if (st.length === 1) return parseSafetensors(join(path, st[0]));
    const gg = entries.filter((e) => e.endsWith(".gguf"));
    if (gg.length === 1) return parseGguf(join(path, gg[0]));
    throw new Error(`Could not find exactly one model (or an index.json) in directory: ${path}`);
  }

  if (path.endsWith(".gguf")) return parseGguf(path);
  if (path.endsWith(".safetensors")) return parseSafetensors(path);
  if (path.endsWith(".index.json")) return parseSafetensorsIndex(path);

  // No extension — sniff: GGUF magic, else safetensors' u64-length + '{'.
  const fh = await open(path, "r");
  const head = Buffer.alloc(9);
  await fh.read(head, 0, 9, 0);
  await fh.close();
  if (head.subarray(0, 4).toString("latin1") === "GGUF") return parseGguf(path);
  if (head[8] === 0x7b) return parseSafetensors(path);
  throw new Error(`Unrecognized model file (expected GGUF or safetensors): ${path}`);
}

const pathArg = z.string().describe("Path to a .gguf or .safetensors file, a *.safetensors.index.json, or a model directory");

server.registerTool(
  "inspect_model",
  {
    title: "Inspect a local model file",
    description:
      "Summarize a local GGUF or safetensors model: architecture, parameter count, quantization, context length, file size, tensor count. Reads only headers — inspecting a 70 GB model is instant. Works on extension-less GGUF blobs (e.g. Ollama's).",
    inputSchema: { path: pathArg },
  },
  guarded(async ({ path }) => {
    const model = await loadModel(path);
    return jsonResult(model.format === "gguf" ? summarizeGguf(model) : summarizeSafetensors(model));
  }),
);

server.registerTool(
  "list_tensors",
  {
    title: "List tensors",
    description: "Tensor names, shapes, and storage types inside a model file. Filter by substring (e.g. 'attn' or 'blk.0').",
    inputSchema: {
      path: pathArg,
      filter: z.string().optional().describe("Case-insensitive substring to filter tensor names"),
      limit: z.number().int().min(1).max(500).default(40).describe("Max tensors to return"),
    },
  },
  guarded(async ({ path, filter, limit }) => {
    const model = await loadModel(path);
    const needle = filter?.toLowerCase();
    const matching = needle ? model.tensors.filter((t) => t.name.toLowerCase().includes(needle)) : model.tensors;
    return jsonResult({
      totalTensors: model.tensors.length,
      matching: matching.length,
      showing: Math.min(matching.length, limit),
      tensors: matching.slice(0, limit),
    });
  }),
);

server.registerTool(
  "estimate_vram",
  {
    title: "Estimate VRAM needed",
    description:
      "Will this model fit? Estimates memory as exact weights size plus a modeled fp16 KV cache (GGUF; context defaults to min(model context, 8192)). Safetensors models get a weights-only figure.",
    inputSchema: {
      path: pathArg,
      context_length: z.number().int().min(1).optional().describe("Context window to budget the KV cache for"),
    },
  },
  guarded(async ({ path, context_length }) => {
    const model = await loadModel(path);
    if (model.format === "gguf") return jsonResult(estimateVram(model, context_length));
    return jsonResult({
      weights: humanBytes(model.weightsBytes),
      total: humanBytes(model.weightsBytes),
      note: "safetensors carries no architecture metadata, so this covers weights only (no KV cache).",
    });
  }),
);

server.registerTool(
  "get_metadata",
  {
    title: "Get model metadata",
    description:
      "The model's metadata key-value store (GGUF) or __metadata__ block (safetensors). Large arrays arrive as {count, sample} summaries and long strings are truncated, so tokenizer vocabularies can't flood the context. Filter keys by substring (e.g. 'tokenizer' or 'rope').",
    inputSchema: {
      path: pathArg,
      filter: z.string().optional().describe("Case-insensitive substring to filter metadata keys"),
    },
  },
  guarded(async ({ path, filter }) => {
    const model = await loadModel(path);
    const source: Record<string, MetadataValue | string> = model.metadata;
    const needle = filter?.toLowerCase();
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(source)) {
      if (needle && !key.toLowerCase().includes(needle)) continue;
      out[key] = typeof value === "string" && value.length > 500 ? `${value.slice(0, 500)}… [truncated, ${value.length} chars]` : value;
    }
    return jsonResult(out);
  }),
);

await server.connect(new StdioServerTransport());
