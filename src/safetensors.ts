// safetensors header parser (https://github.com/huggingface/safetensors)
// Layout: u64-LE header length, then that many bytes of JSON. Tensor data
// follows and is never read. Sharded models are handled via *.index.json.

import { open, readFile, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { humanBytes, humanCount, type TensorInfo } from "./gguf.js";

const HEADER_CAP = 256 << 20;

export interface SafetensorsModel {
  format: "safetensors";
  fileSizeBytes: number;
  weightsBytes: number;
  paramCount: number;
  tensors: TensorInfo[];
  dtypes: Record<string, number>;
  metadata: Record<string, string>;
  shards: number;
}

async function readHeader(path: string): Promise<Record<string, any>> {
  const fh = await open(path, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    const { bytesRead } = await fh.read(lenBuf, 0, 8, 0);
    if (bytesRead < 8) throw new Error("Not a safetensors file (too short)");
    const len = lenBuf.readBigUInt64LE(0);
    if (len > BigInt(HEADER_CAP)) throw new Error("safetensors header exceeds sanity cap — file is likely corrupt");
    const jsonBuf = Buffer.alloc(Number(len));
    const read = await fh.read(jsonBuf, 0, Number(len), 8);
    if (read.bytesRead < Number(len)) throw new Error("Unexpected end of file while reading safetensors header");
    try {
      return JSON.parse(jsonBuf.toString("utf8"));
    } catch {
      throw new Error("Not a safetensors file (header is not valid JSON)");
    }
  } finally {
    await fh.close();
  }
}

function emptyModel(): SafetensorsModel {
  return {
    format: "safetensors",
    fileSizeBytes: 0,
    weightsBytes: 0,
    paramCount: 0,
    tensors: [],
    dtypes: {},
    metadata: {},
    shards: 0,
  };
}

async function addFile(model: SafetensorsModel, path: string): Promise<void> {
  const header = await readHeader(path);
  model.fileSizeBytes += (await stat(path)).size;
  model.shards += 1;
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") {
      Object.assign(model.metadata, info);
      continue;
    }
    const shape: number[] = info.shape ?? [];
    model.tensors.push({ name, shape, type: info.dtype ?? "?" });
    model.paramCount += shape.reduce((a: number, b: number) => a * b, 1);
    model.dtypes[info.dtype] = (model.dtypes[info.dtype] ?? 0) + 1;
    if (Array.isArray(info.data_offsets)) model.weightsBytes += info.data_offsets[1] - info.data_offsets[0];
  }
}

export async function parseSafetensors(path: string): Promise<SafetensorsModel> {
  const model = emptyModel();
  await addFile(model, path);
  return model;
}

export async function parseSafetensorsIndex(indexPath: string): Promise<SafetensorsModel> {
  const index = JSON.parse(await readFile(indexPath, "utf8"));
  const weightMap: Record<string, string> = index.weight_map ?? {};
  const shardFiles = [...new Set(Object.values(weightMap))];
  if (shardFiles.length === 0) throw new Error("safetensors index has an empty weight_map");
  const model = emptyModel();
  for (const shard of shardFiles) {
    await addFile(model, join(dirname(indexPath), shard));
  }
  return model;
}

export interface SafetensorsSummary {
  format: "safetensors";
  parameters: string;
  dtypes: Record<string, number>;
  fileSize: string;
  tensorCount: number;
  shards: number;
  metadata: Record<string, string>;
}

export function summarizeSafetensors(m: SafetensorsModel): SafetensorsSummary {
  return {
    format: "safetensors",
    parameters: humanCount(m.paramCount),
    dtypes: m.dtypes,
    fileSize: humanBytes(m.fileSizeBytes),
    tensorCount: m.tensors.length,
    shards: m.shards,
    metadata: m.metadata,
  };
}
