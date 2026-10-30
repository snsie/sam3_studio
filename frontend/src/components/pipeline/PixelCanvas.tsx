import { useEffect, useRef } from "react";

interface PixelCanvasProps {
  width: number;
  height: number;
  /** RGBA bytes, width × height × 4. */
  pixels: Uint8ClampedArray | null;
  className?: string;
  label?: string;
  /** Pixelated upscaling (default) shows the actual metric grid. */
  smooth?: boolean;
}

/** Paints a small RGBA buffer into a canvas that CSS scales to display size. */
export function PixelCanvas({ width, height, pixels, className, label, smooth = false }: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context || !pixels) {
      return;
    }
    context.putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
  }, [pixels, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={`pixel-canvas${smooth ? " is-smooth" : ""}${className ? ` ${className}` : ""}`}
      role={label ? "img" : undefined}
      aria-label={label}
    />
  );
}
