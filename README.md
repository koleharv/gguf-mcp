# gguf-mcp

An [MCP](https://modelcontextprotocol.io) server that inspects **local model files** — GGUF and safetensors — so Claude and other LLMs can answer questions about the models on your disk:

- *"What is this .gguf? Architecture, quantization, parameter count?"*
- *"Will this model fit in my 12 GB GPU at 8k context?"*
- *"What tensors are inside, with what shapes?"*
- *"Show me its chat template / RoPE settings / tokenizer config."*

**Headers only.** The parser never touches tensor data, so inspecting a 70 GB model takes milliseconds and a few MiB of I/O. No network, no API keys, no telemetry — your files never leave your machine.

## Quick start

**Claude Code**

```bash
claude mcp add gguf -- npx -y gguf-mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gguf": {
      "command": "npx",
      "args": ["-y", "gguf-mcp"]
    }
  }
}
```

The same `npx` invocation works in Cursor, Windsurf, and any other MCP client.

## Tools

| Tool | What it does |
|------|--------------|
| `inspect_model` | One-call summary: format, architecture, parameters, quantization, context length, file size, tensor count |
| `list_tensors` | Tensor names, shapes, and storage types — filterable (`attn`, `blk.0`, ...) |
| `estimate_vram` | Fit check: exact weights size + modeled fp16 KV cache for your chosen context length |
| `get_metadata` | The GGUF key-value store (or safetensors `__metadata__`), filterable by key |

Paths can be a `.gguf` file, a `.safetensors` file, a `*.safetensors.index.json`, or a model directory (sharded HuggingFace layouts are aggregated across shards). Extension-less GGUF blobs — like the ones in Ollama's `~/.ollama/models/blobs` — are detected by magic bytes.

## Design notes

- **Context-friendly by construction.** A tokenizer vocabulary is 100k+ strings; metadata arrays are returned as `{count, sample}` summaries and long strings (chat templates) are truncated with a marker. The full data stays on disk where it belongs.
- **Honest estimates.** `estimate_vram` reports exact on-disk weight bytes plus the standard KV-cache formula (2 × layers × context × KV heads × head dim × 2 bytes), and says what it excludes rather than faking precision.
- **Defensive parsing.** Magic checks, version checks (incl. big-endian detection), truncation detection, and sanity caps on header sizes — malformed files produce specific, actionable errors.
- **Zero runtime dependencies** beyond the MCP SDK and zod. The GGUF binary reader and safetensors parser are hand-rolled and unit-tested against synthetic files built in the test suite — no fixtures, no downloads.

## Development

```bash
npm install
npm test                 # offline unit tests (vitest) — synthetic model files
npm run build            # tsc → dist/
node scripts/smoke.mjs   # end-to-end: generates models, drives the server over stdio
```

Architecture: [`src/gguf.ts`](src/gguf.ts) (binary header parser + VRAM math) and [`src/safetensors.ts`](src/safetensors.ts) (JSON header + shard index) are pure logic with no MCP imports; [`src/index.ts`](src/index.ts) is the MCP wiring and path/format detection.

## Out of scope

Tensor statistics (would require reading data), PyTorch `.bin` (pickle — unsafe by design), ONNX, and remote HuggingFace queries (HuggingFace has an official MCP server for that).

## License

MIT
