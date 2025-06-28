"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCw,
  Upload,
  AlertCircle,
  Activity,
  Cpu,
  Layers,
  HardDrive,
  Image as ImageIcon,
  ZapOff,
} from "lucide-react";
import JSZip from "jszip";

// --- Type Definitions for gpu.js (to be imported dynamically) ---
interface IKernel<T extends unknown[], U> {
  (...args: T): U;
}
interface GPU {
  createKernel<T extends unknown[], U>(
    fn: unknown,
    settings?: unknown,
  ): IKernel<T, U>;
}
type Input = unknown;
type GpuConstructor = new (settings?: unknown) => GPU;
type InputConstructor = new (value: unknown, size: unknown) => Input;

// --- Application-specific Type Definitions ---
type VoxelVolume = number[][][];
type Mesh = { vertices: number[][]; triangles: number[][] };
type Projection = { angle: number; data: number[][] };
type ModelType = "gear" | "cube" | "sphere" | "custom";
type SystemStatusType = "ok" | "warn" | "error";
type SystemStatus = { text: string; type: SystemStatusType };

// =====================================================================================
// UTILITY & PARSING CLASSES
// =====================================================================================
class STLParser {
  public parseASCII(text: string): Mesh {
    const vertices: number[][] = [];
    const triangles: number[][] = [];
    let currentTriangle: number[] = [];
    text.split("\n").forEach((line) => {
      const trimmed = line.trim();
      if (trimmed.startsWith("vertex")) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 4) {
          vertices.push([
            parseFloat(parts[1]),
            parseFloat(parts[2]),
            parseFloat(parts[3]),
          ]);
          currentTriangle.push(vertices.length - 1);
          if (currentTriangle.length === 3) {
            triangles.push([...currentTriangle]);
            currentTriangle = [];
          }
        }
      }
    });
    return { vertices, triangles };
  }
  public parseBinary(buffer: ArrayBuffer): Mesh {
    const dataView = new DataView(buffer);
    const triangleCount = dataView.getUint32(80, true);
    const vertices: number[][] = [];
    const triangles: number[][] = [];
    let offset = 84;
    for (let i = 0; i < triangleCount; i++) {
      offset += 12;
      const triangle: number[] = [];
      for (let j = 0; j < 3; j++) {
        vertices.push([
          dataView.getFloat32(offset, true),
          dataView.getFloat32(offset + 4, true),
          dataView.getFloat32(offset + 8, true),
        ]);
        triangle.push(vertices.length - 1);
        offset += 12;
      }
      triangles.push(triangle);
      offset += 2;
    }
    return { vertices, triangles };
  }
  public meshToVoxels(mesh: Mesh, gridSize: number): VoxelVolume {
    const { vertices } = mesh;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    vertices.forEach((v) => {
      for (let i = 0; i < 3; i++) {
        min[i] = Math.min(min[i], v[i]);
        max[i] = Math.max(max[i], v[i]);
      }
    });
    const size = max.map((m, i) => m - min[i]);
    const scale = (gridSize * 0.8) / Math.max(...size);
    const voxels: VoxelVolume = Array(gridSize)
      .fill(0)
      .map(() =>
        Array(gridSize)
          .fill(0)
          .map(() => Array(gridSize).fill(0)),
      );
    vertices.forEach((vertex) => {
      const x = Math.floor(
        (vertex[0] - min[0]) * scale + (gridSize - size[0] * scale) / 2,
      );
      const y = Math.floor(
        (vertex[1] - min[1]) * scale + (gridSize - size[1] * scale) / 2,
      );
      const z = Math.floor(
        (vertex[2] - min[2]) * scale + (gridSize - size[2] * scale) / 2,
      );
      if (
        x >= 0 &&
        x < gridSize &&
        y >= 0 &&
        y < gridSize &&
        z >= 0 &&
        z < gridSize
      ) {
        voxels[z][y][x] = 1;
      }
    });
    return voxels;
  }
}

class CALProjector {
  public volumeSize: number;
  public numProjections: number;
  public projections: Projection[];
  public progressCallback: ((progress: number) => void) | null;
  private gpu: GPU;
  private radonKernel: IKernel<[Input, number, number], number[][]>;
  private InputClass: InputConstructor;

  constructor(
    GpuClass: GpuConstructor,
    InputClass: InputConstructor,
    volumeSize = 128,
    numProjections = 180,
  ) {
    this.volumeSize = volumeSize;
    this.numProjections = numProjections;
    this.projections = [];
    this.progressCallback = null;
    this.gpu = new GpuClass();
    this.InputClass = InputClass;
    this.radonKernel = this._createRadonKernel();
  }

  private _createRadonKernel(): IKernel<[Input, number, number], number[][]> {
    return this.gpu.createKernel(
      function (
        this: { thread: { x: number; y: number } },
        volume: number[][][],
        theta: number,
        size: number,
      ): number {
        const u = this.thread.x;
        const v = this.thread.y;
        const angleRad = theta * (Math.PI / 180);
        const cosTheta = Math.cos(angleRad);
        const sinTheta = Math.sin(angleRad);
        const center = size / 2;
        const uCentered = u - center;
        const vCentered = v - center;
        let lineIntegral = 0;
        for (let t = -size * 0.7; t < size * 0.7; t += 0.5) {
          const x = uCentered * cosTheta - t * sinTheta + center;
          const y = vCentered + center;
          const z = uCentered * sinTheta + t * cosTheta + center;
          if (
            x >= 0 &&
            x < size - 1 &&
            y >= 0 &&
            y < size - 1 &&
            z >= 0 &&
            z < size - 1
          ) {
            const x0 = Math.floor(x),
              y0 = Math.floor(y),
              z0 = Math.floor(z);
            const dx = x - x0,
              dy = y - y0,
              dz = z - z0;
            const c000 = volume[z0][y0][x0];
            const c100 = volume[z0][y0][x0 + 1];
            const c010 = volume[z0][y0 + 1][x0];
            const c110 = volume[z0][y0 + 1][x0 + 1];
            const c001 = volume[z0 + 1][y0][x0];
            const c101 = volume[z0 + 1][y0][x0 + 1];
            const c011 = volume[z0 + 1][y0 + 1][x0];
            const c111 = volume[z0 + 1][y0 + 1][x0 + 1];
            const c00 = c000 * (1 - dx) + c100 * dx;
            const c01 = c001 * (1 - dx) + c101 * dx;
            const c10 = c010 * (1 - dx) + c110 * dx;
            const c11 = c011 * (1 - dx) + c111 * dx;
            const c0 = c00 * (1 - dy) + c10 * dy;
            const c1 = c01 * (1 - dy) + c11 * dy;
            lineIntegral += (c0 * (1 - dz) + c1 * dz) * 0.5;
          }
        }
        return lineIntegral;
      },
      { output: [this.volumeSize, this.volumeSize] },
    );
  }

  public setProgressCallback(callback: (progress: number) => void): void {
    this.progressCallback = callback;
  }

  public async generateProjections(volume: VoxelVolume): Promise<Projection[]> {
    this.projections = [];
    const flatVolume = volume.flat(2);
    const inputVolume = new this.InputClass(flatVolume, [
      this.volumeSize,
      this.volumeSize,
      this.volumeSize,
    ]);
    for (let i = 0; i < this.numProjections; i++) {
      const angle = (i / this.numProjections) * 360;
      // The error originates here, so the calling function needs the try...catch
      const projectionData = this.radonKernel(
        inputVolume,
        angle,
        this.volumeSize,
      );
      this.projections.push({ angle, data: projectionData as number[][] });
      this.progressCallback?.((i + 1) / this.numProjections);
      if (i % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    this.normalizeProjections();
    return this.projections;
  }

  private normalizeProjections(): void {
    let globalMin = Infinity;
    let globalMax = -Infinity;
    this.projections.forEach((proj) => {
      proj.data.forEach((row) => {
        row.forEach((val) => {
          if (val < globalMin) globalMin = val;
          if (val > globalMax) globalMax = val;
        });
      });
    });
    const range = globalMax - globalMin || 1;
    this.projections.forEach((proj) => {
      proj.data = proj.data.map((row) =>
        row.map((val) => Math.round(((val - globalMin) / range) * 255)),
      );
    });
  }

  public createTestModel(modelType: ModelType | string): VoxelVolume {
    const size = this.volumeSize;
    const model: VoxelVolume = Array(size)
      .fill(0)
      .map(() =>
        Array(size)
          .fill(0)
          .map(() => Array(size).fill(0)),
      );
    const center = size / 2;
    switch (modelType) {
      case "gear":
        const outerRadius = size / 3,
          innerRadius = size / 6,
          teeth = 8,
          height = size * 0.6;
        const startZ = (size - height) / 2,
          endZ = startZ + height;
        for (let z = Math.floor(startZ); z < Math.floor(endZ); z++) {
          for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
              const dx = x - center,
                dy = y - center,
                dist = Math.sqrt(dx * dx + dy * dy),
                angle = Math.atan2(dy, dx);
              const toothAngle = ((angle + Math.PI) / (2 * Math.PI)) * teeth,
                toothPhase = toothAngle - Math.floor(toothAngle);
              const radiusMod = toothPhase < 0.5 ? 1.0 : 0.7,
                effectiveRadius =
                  innerRadius + (outerRadius - innerRadius) * radiusMod;
              if (dist <= effectiveRadius) model[z][y][x] = 1;
            }
          }
        }
        break;
      case "cube":
        const cubeSize = size * 0.4,
          start = (size - cubeSize) / 2,
          end = start + cubeSize;
        for (let z = Math.floor(start); z < Math.floor(end); z++)
          for (let y = Math.floor(start); y < Math.floor(end); y++)
            for (let x = Math.floor(start); x < Math.floor(end); x++)
              model[z][y][x] = 1;
        break;
      case "sphere":
        const radius = size * 0.3;
        for (let z = 0; z < size; z++)
          for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++)
              if (
                Math.sqrt(
                  Math.pow(x - center, 2) +
                    Math.pow(y - center, 2) +
                    Math.pow(z - center, 2),
                ) <= radius
              )
                model[z][y][x] = 1;
        break;
    }
    return model;
  }
}

// =====================================================================================
// REACT UI COMPONENT
// =====================================================================================
function CALProjectorSimulator(): React.JSX.Element {
  const [Gpu, setGpu] = useState<{
    GPU: GpuConstructor;
    Input: InputConstructor;
  } | null>(null);
  const projectorRef = useRef<CALProjector | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [projections, setProjections] = useState<Projection[]>([]);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [volumeSize, setVolumeSize] = useState<number>(128);
  const [numProjections, setNumProjections] = useState<number>(180);
  const [modelType, setModelType] = useState<ModelType>("gear");
  const [customModel, setCustomModel] = useState<VoxelVolume | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isZipping, setIsZipping] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    text: "INITIALIZING",
    type: "warn",
  });
  const [statusMessage, setStatusMessage] = useState<string>(
    "Loading core libraries...",
  );
  const [canvasSize] = useState<number>(512);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [showCrosshairs, setShowCrosshairs] = useState<boolean>(true);

  useEffect(() => {
    import("gpu.js")
      .then((GpuModule) => {
        // Handle both named and default exports with proper type assertion
        const gpuModule = GpuModule as Record<string, unknown>;
        const GPU =
          gpuModule.GPU ||
          (gpuModule.default as Record<string, unknown>)?.GPU ||
          gpuModule.default;
        const Input =
          gpuModule.Input ||
          (gpuModule.default as Record<string, unknown>)?.Input ||
          ((gpuModule.default as Record<string, unknown>) &&
            (gpuModule.default as Record<string, unknown>).Input);
        setGpu({
          GPU: GPU as GpuConstructor,
          Input: Input as InputConstructor,
        });
      })
      .catch((e) => {
        console.error("Failed to load gpu.js", e);
        setSystemStatus({ text: "ERROR", type: "error" });
        setStatusMessage("GPU library failed to load. Please refresh.");
      });
  }, []);

  useEffect(() => {
    if (Gpu && !projectorRef.current) {
      const proj = new CALProjector(
        Gpu.GPU,
        Gpu.Input,
        volumeSize,
        numProjections,
      );
      projectorRef.current = proj;
      setSystemStatus({ text: "STANDBY", type: "warn" });
      setStatusMessage("Ready to generate.");
      generateProjections(proj);
    }
  }, [Gpu]); // eslint-disable-line

  const generateProjections = async (
    projInstance?: CALProjector | null,
  ): Promise<void> => {
    const projector = projInstance || projectorRef.current;
    if (!projector) {
      setStatusMessage("Projector not initialized...");
      return;
    }
    projector.volumeSize = volumeSize;
    projector.numProjections = numProjections;
    setIsGenerating(true);
    setProgress(0);
    setSystemStatus({ text: "GENERATING", type: "error" });
    setStatusMessage("Compiling GPU kernel...");

    // FIX: Wrap the GPU-intensive operation in a try...catch block
    try {
      await new Promise((r) => setTimeout(r, 10));
      projector.setProgressCallback((p) => {
        setProgress(p);
        setStatusMessage(
          `Processing angle ${Math.round(p * numProjections)}/${numProjections}`,
        );
      });
      const volume = customModel || projector.createTestModel(modelType);
      const projs = await projector.generateProjections(volume);
      setProjections(projs);
      setCurrentFrame(0);
      setSystemStatus({ text: "READY", type: "ok" });
      setStatusMessage(`Generated ${numProjections} projections.`);
    } catch (error) {
      console.error("GPU Error:", error);
      // Provide a user-friendly error message
      setSystemStatus({ text: "GPU ERROR", type: "error" });
      setStatusMessage("Hardware limit reached. Try a lower resolution.");
      setProjections([]); // Clear any partial or old projections
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFileUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSystemStatus({ text: "LOADING MODEL", type: "warn" });
    setStatusMessage(`Parsing ${file.name}...`);
    const parser = new STLParser();
    try {
      if (!file.name.toLowerCase().endsWith(".stl"))
        throw new Error("Invalid file type. Please upload an STL file.");
      const arrayBuffer = await file.arrayBuffer();
      const text = new TextDecoder().decode(arrayBuffer.slice(0, 80));
      const mesh: Mesh = text.startsWith("solid")
        ? parser.parseASCII(new TextDecoder().decode(arrayBuffer))
        : parser.parseBinary(arrayBuffer);
      setStatusMessage("Voxelizing model...");
      await new Promise((r) => setTimeout(r, 10));
      const voxels = parser.meshToVoxels(mesh, volumeSize);
      setCustomModel(voxels);
      setModelType("custom");
      setSystemStatus({ text: "MODEL LOADED", type: "ok" });
      setStatusMessage(`${file.name} successfully voxelized.`);
    } catch (error) {
      console.error("Error loading model:", error);
      setSystemStatus({ text: "ERROR", type: "error" });
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Failed to load or parse model.",
      );
    }
    if (e.target) e.target.value = "";
  };

  const drawProjection = useCallback(
    (projection?: Projection): void => {
      const canvas = canvasRef.current;
      if (!canvas || !projection) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const size = projection.data.length;
      const scale = canvasSize / size;
      const imageData = ctx.createImageData(canvasSize, canvasSize);
      const data = imageData.data;
      for (let y = 0; y < canvasSize; y++) {
        for (let x = 0; x < canvasSize; x++) {
          const projX = Math.floor(x / scale),
            projY = Math.floor(y / scale);
          const value = projection.data[projY]?.[projX] || 0;
          const i = (y * canvasSize + x) * 4;
          data[i] = value;
          data[i + 1] = value;
          data[i + 2] = value;
          data[i + 3] = 255;
        }
      }
      ctx.putImageData(imageData, 0, 0);
      ctx.strokeStyle = "rgba(234, 88, 12, 0.5)";
      ctx.lineWidth = 1;
      if (showGrid) {
        ctx.globalAlpha = 0.3;
        for (let i = 0; i <= size; i += 16) {
          ctx.beginPath();
          ctx.moveTo(i * scale, 0);
          ctx.lineTo(i * scale, canvasSize);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(0, i * scale);
          ctx.lineTo(canvasSize, i * scale);
          ctx.stroke();
        }
      }
      if (showCrosshairs) {
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.moveTo(canvasSize / 2, 0);
        ctx.lineTo(canvasSize / 2, canvasSize);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, canvasSize / 2);
        ctx.lineTo(canvasSize, canvasSize / 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    },
    [canvasSize, showGrid, showCrosshairs],
  );

  const exportImagesAsZip = async (): Promise<void> => {
    if (projections.length === 0 || isZipping) return;
    setIsZipping(true);
    setSystemStatus({ text: "EXPORTING", type: "warn" });
    const zip = new JSZip();
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = volumeSize;
    offscreenCanvas.height = volumeSize;
    const offscreenCtx = offscreenCanvas.getContext("2d");
    if (!offscreenCtx) {
      setIsZipping(false);
      setSystemStatus({ text: "ERROR", type: "error" });
      setStatusMessage("Could not create off-screen context for export.");
      return;
    }
    for (let i = 0; i < projections.length; i++) {
      const proj = projections[i];
      const imageData = offscreenCtx.createImageData(volumeSize, volumeSize);
      for (let y = 0; y < volumeSize; y++) {
        for (let x = 0; x < volumeSize; x++) {
          const val = proj.data[y][x];
          const index = (y * volumeSize + x) * 4;
          imageData.data[index] = val;
          imageData.data[index + 1] = val;
          imageData.data[index + 2] = val;
          imageData.data[index + 3] = 255;
        }
      }
      offscreenCtx.putImageData(imageData, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) =>
        offscreenCanvas.toBlob(resolve, "image/png"),
      );
      if (blob) zip.file(`projection_${String(i).padStart(4, "0")}.png`, blob);
      setStatusMessage(`Zipping image ${i + 1}/${projections.length}`);
      if (i % 10 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    setStatusMessage("Compressing ZIP file...");
    const zipBlob = await zip.generateAsync({
      type: "blob",
      compression: "DEFLATE",
      compressionOptions: { level: 6 },
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(zipBlob);
    link.download = `cal_projections_${modelType}_${volumeSize}x${numProjections}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    setIsZipping(false);
    setSystemStatus({ text: "READY", type: "ok" });
    setStatusMessage(`Export complete.`);
  };

  useEffect(() => {
    if (isPlaying && projections.length > 0) {
      const id = setInterval(
        () => setCurrentFrame((p) => (p + 1) % projections.length),
        1000 / 60,
      );
      return () => clearInterval(id);
    }
  }, [isPlaying, projections.length]);
  useEffect(() => {
    drawProjection(projections[currentFrame]);
  }, [currentFrame, projections, showGrid, showCrosshairs, drawProjection]);

  const isLibraryLoading = !Gpu;
  const isBusy = isGenerating || isZipping || isLibraryLoading;
  const statusColors: Record<SystemStatusType, string> = {
    ok: "bg-green-500",
    warn: "bg-yellow-500",
    error: "bg-red-500",
  };
  const PlayPauseIcon = isPlaying ? Pause : Play;
  const ExportIcon = isZipping ? RotateCw : ImageIcon;
  const GenerateIcon = isGenerating ? RotateCw : HardDrive;

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-300 font-mono select-none">
      <header className="flex-shrink-0 bg-gray-950/70 border-b border-gray-800 px-6 py-3 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-orange-500 tracking-wider">
              C.A.L. SIMULATOR
            </h1>
            <p className="text-xs text-gray-500">
              Computed Axial Lithography // Projection Generation System
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${isBusy ? "animate-pulse" : ""} ${statusColors[systemStatus.type]}`}
              />
              <span className="text-gray-400">{systemStatus.text}</span>
            </div>
            <div className="text-gray-500">
              {new Date().toISOString().split("T")[0]}{" "}
              {new Date().toTimeString().split(" ")[0]}
            </div>
          </div>
        </div>
      </header>
      <main className="flex flex-1 overflow-hidden">
        <div className="flex-1 p-4 md:p-6 flex flex-col">
          <div className="bg-black/30 border border-gray-800 rounded-lg h-full flex flex-col">
            <div className="border-b border-gray-800 px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Activity className="text-orange-500" size={16} />
                <span className="text-sm font-semibold text-gray-300">
                  PROJECTION VIEWPORT
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowGrid((s) => !s)}
                  className={`px-3 py-1 text-xs rounded ${showGrid ? "bg-orange-600/80 text-white" : "bg-gray-700/50 hover:bg-gray-700"}`}
                >
                  GRID
                </button>
                <button
                  onClick={() => setShowCrosshairs((s) => !s)}
                  className={`px-3 py-1 text-xs rounded ${showCrosshairs ? "bg-orange-600/80 text-white" : "bg-gray-700/50 hover:bg-gray-700"}`}
                >
                  RETICLE
                </button>
              </div>
            </div>
            <div className="flex-1 p-4 flex items-center justify-center relative overflow-hidden">
              <canvas
                ref={canvasRef}
                width={canvasSize}
                height={canvasSize}
                className="border border-gray-700"
                style={{ imageRendering: "pixelated" }}
              />
              {isBusy && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 backdrop-blur-sm">
                  <div className="text-center">
                    <div className="text-orange-500 text-lg mb-4">
                      {systemStatus.text}...
                    </div>
                    <div className="w-64 h-1 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-orange-500 transition-all duration-150"
                        style={{
                          width: isGenerating ? `${progress * 100}%` : "100%",
                        }}
                      />
                    </div>
                    <div className="text-xs text-gray-400 mt-2">
                      {statusMessage}
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="border-t border-gray-800 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsPlaying((p) => !p)}
                    className="bg-orange-600 hover:bg-orange-500 px-4 py-2 rounded text-sm flex items-center gap-2 disabled:opacity-50"
                    disabled={isBusy}
                  >
                    <PlayPauseIcon size={16} />
                    {isPlaying ? "PAUSE" : "PLAY"}
                  </button>
                  <button
                    onClick={() => setCurrentFrame(0)}
                    className="bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded text-sm flex items-center gap-2 disabled:opacity-50"
                    disabled={isBusy}
                  >
                    <RotateCw size={16} />
                    RESET
                  </button>
                  <button
                    onClick={exportImagesAsZip}
                    className="bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded text-sm flex items-center gap-2 disabled:opacity-50"
                    disabled={isBusy || projections.length === 0}
                  >
                    <ExportIcon
                      size={16}
                      className={isZipping ? "animate-spin" : ""}
                    />
                    {isZipping ? "PACKAGING..." : "EXPORT IMAGES"}
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-x-6 text-xs text-right">
                  <div>
                    <span className="text-gray-500">ANGLE: </span>
                    <span className="text-orange-400 w-16 inline-block">
                      {projections[currentFrame]?.angle.toFixed(1) ?? "0.0"}°
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">FRAME: </span>
                    <span className="text-orange-400 w-16 inline-block">
                      {projections.length > 0 ? currentFrame + 1 : 0} /{" "}
                      {projections.length}
                    </span>
                  </div>
                </div>
              </div>
              <input
                type="range"
                min="0"
                max={projections.length > 0 ? projections.length - 1 : 0}
                value={currentFrame}
                onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700"
                disabled={isBusy}
              />
            </div>
          </div>
        </div>
        <aside className="w-80 flex-shrink-0 bg-gray-950/50 border-l border-gray-800 p-4 overflow-y-auto">
          <div className="space-y-6">
            <div className="space-y-4 p-4 bg-black/20 rounded-lg border border-gray-800">
              <div className="flex items-center gap-3">
                <Layers className="text-orange-500" size={16} />
                <h3 className="text-sm font-semibold">MODEL CONFIGURATION</h3>
              </div>
              <select
                value={modelType}
                onChange={(e) => {
                  setModelType(e.target.value as ModelType);
                  setCustomModel(null);
                }}
                disabled={isBusy}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:ring-orange-500 focus:border-orange-500 disabled:opacity-50"
              >
                <option value="gear">GEAR</option>
                <option value="cube">CUBE</option>
                <option value="sphere">SPHERE</option>
                {customModel && <option value="custom">CUSTOM MODEL</option>}
              </select>
              <input
                ref={fileInputRef}
                type="file"
                accept=".stl"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded text-sm flex items-center justify-center gap-2 disabled:opacity-50"
                disabled={isBusy}
              >
                <Upload size={16} />
                UPLOAD STL
              </button>
            </div>
            <div className="space-y-4 p-4 bg-black/20 rounded-lg border border-gray-800">
              <div className="flex items-center gap-3">
                <Cpu className="text-orange-500" size={16} />
                <h3 className="text-sm font-semibold">SYSTEM PARAMETERS</h3>
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-2">
                  VOXEL RESOLUTION: {volumeSize}³
                </label>
                <input
                  type="range"
                  min="64"
                  max="256"
                  step="64"
                  value={volumeSize}
                  onChange={(e) => setVolumeSize(parseInt(e.target.value))}
                  className="w-full"
                  disabled={isBusy}
                />
                {/* FIX: Add a user-facing warning for high-resolution settings */}
                {volumeSize === 256 && (
                  <div className="mt-2 text-xs text-yellow-400/80 bg-yellow-900/30 p-2 rounded flex items-start gap-2">
                    <ZapOff size={20} className="flex-shrink-0" />
                    <span>
                      High resolution is hardware-intensive and may fail on some
                      devices. If generation fails, please select a lower
                      resolution.
                    </span>
                  </div>
                )}
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-2">
                  PROJECTION COUNT: {numProjections}
                </label>
                <input
                  type="range"
                  min="36"
                  max="720"
                  step="36"
                  value={numProjections}
                  onChange={(e) => setNumProjections(parseInt(e.target.value))}
                  className="w-full"
                  disabled={isBusy}
                />
              </div>
              <button
                onClick={() => generateProjections()}
                disabled={isBusy}
                className="w-full bg-orange-600 hover:bg-orange-500 px-4 py-3 rounded text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-50 disabled:bg-orange-900"
              >
                <GenerateIcon
                  size={16}
                  className={isGenerating ? "animate-spin" : ""}
                />
                {isGenerating ? "PROCESSING..." : "GENERATE PROJECTIONS"}
              </button>
            </div>
            <div className="space-y-2 p-4 bg-black/20 rounded-lg border border-gray-800">
              <div className="flex items-center gap-3 mb-2">
                <AlertCircle className="text-orange-500" size={16} />
                <h3 className="text-sm font-semibold">TECHNICAL INFO</h3>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">COMPUTE:</span>
                <span>{isGenerating ? "GPU" : "IDLE"}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">ALGORITHM:</span>
                <span>RADON TRANSFORM</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">INTERPOLATION:</span>
                <span>TRILINEAR</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">STEP SIZE:</span>
                <span>0.5 VOXELS</span>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default CALProjectorSimulator;
