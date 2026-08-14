// Self-contained smoke test: generates tiny-but-valid model files in a temp
// dir, then drives the built server over stdio. No network, no downloads.
// Usage: npm run build && node scripts/smoke.mjs
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// --- minimal GGUF/safetensors writers (mirror of test/fixtures.ts) ---------
const u32 = (n) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const u64 = (n) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
const gstr = (s) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s, "utf8")]);
const kvU32 = (k, v) => Buffer.concat([gstr(k), u32(4), u32(v)]);
const kvStr = (k, v) => Buffer.concat([gstr(k), u32(8), gstr(v)]);
const kvArr = (k, vals) => Buffer.concat([gstr(k), u32(9), u32(8), u64(vals.length), ...vals.map(gstr)]);
const tensor = (name, shape, type) => Buffer.concat([gstr(name), u32(shape.length), ...shape.map(u64), u32(type), u64(0)]);

function tinyGguf() {
  const kvs = [
    kvStr("general.architecture", "llama"),
    kvStr("general.name", "SmokeLlama"),
    kvU32("general.file_type", 15),
    kvU32("llama.context_length", 4096),
    kvU32("llama.block_count", 2),
    kvU32("llama.attention.head_count", 4),
    kvU32("llama.attention.head_count_kv", 2),
    kvU32("llama.embedding_length", 64),
    kvArr("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world", "!", "?", "…"]),
  ];
  const tensors = [tensor("blk.0.attn_q.weight", [64, 64], 0), tensor("output.weight", [64, 100], 1)];
  const header = Buffer.concat([Buffer.from("GGUF", "latin1"), u32(3), u64(2), u64(kvs.length), ...kvs, ...tensors]);
  const pad = Buffer.alloc((32 - (header.length % 32)) % 32);
  return Buffer.concat([header, pad, Buffer.alloc(128, 7)]);
}

function tinySafetensors() {
  const json = JSON.stringify({
    __metadata__: { format: "pt" },
    emb: { dtype: "F16", shape: [10, 20], data_offsets: [0, 400] },
    w: { dtype: "F32", shape: [5], data_offsets: [400, 420] },
  });
  return Buffer.concat([u64(Buffer.byteLength(json)), Buffer.from(json), Buffer.alloc(420, 1)]);
}

// --- write files, drive the server -----------------------------------------
const dir = await mkdtemp(join(tmpdir(), "gguf-mcp-smoke-"));
const gg = join(dir, "tiny.gguf");
const noext = join(dir, "sha256-abc123"); // Ollama-style extension-less blob
const st = join(dir, "tiny.safetensors");
await writeFile(gg, tinyGguf());
await writeFile(noext, tinyGguf());
await writeFile(st, tinySafetensors());

const client = new Client({ name: "smoke", version: "0.0.0" });
await client.connect(new StdioClientTransport({ command: process.execPath, args: ["dist/index.js"] }));

const { tools } = await client.listTools();
console.log("tools:", tools.map((t) => t.name).join(", "));

const calls = [
  ["inspect_model", { path: gg }],
  ["inspect_model", { path: noext }], // magic sniffing
  ["inspect_model", { path: st }],
  ["list_tensors", { path: gg, filter: "attn" }],
  ["estimate_vram", { path: gg, context_length: 1024 }],
  ["get_metadata", { path: gg, filter: "tokenizer" }],
  ["inspect_model", { path: join(dir, "missing.gguf") }], // must be isError
];

for (const [name, args] of calls) {
  const res = await client.callTool({ name, arguments: args });
  console.log(`\n== ${name}(${args.filter ?? args.context_length ?? ""}) ${res.isError ? "(isError)" : ""}\n${res.content[0].text.slice(0, 450)}`);
}

await client.close();
await rm(dir, { recursive: true, force: true });
console.log("\nSMOKE OK");
