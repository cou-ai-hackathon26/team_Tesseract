const MAX_DIMENSION = 1280;
const JPEG_QUALITY = 0.82;

export interface PreparedImage {
  /** Downscaled image ready to upload. */
  blob: Blob;
  /** Object URL for the local preview, owned by the caller. */
  previewUrl: string;
  width: number;
  height: number;
}

/**
 * Downscale + re-encode a selected photo before upload.
 *
 * The server does its own downscale and EXIF strip, so this is not the last word
 * on size - it just avoids spending a rural mobile connection on a 6 MB original
 * that will be resized anyway.
 */
export async function prepareImage(file: File): Promise<PreparedImage> {
  const bitmap = await loadImage(await readAsDataUrl(file));
  if (!bitmap) {
    // Undecodable (e.g. HEIC in an unsupported browser): send the original.
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      width: 0,
      height: 0,
    };
  }

  const scale = Math.min(1, MAX_DIMENSION / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d');
  if (!context) {
    return { blob: file, previewUrl: URL.createObjectURL(file), width: bitmap.width, height: bitmap.height };
  }

  context.drawImage(bitmap, 0, 0, width, height);

  const blob = await toBlob(canvas);
  return { blob, previewUrl: URL.createObjectURL(blob), width, height };
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => resolve(blob ?? new Blob()),
      'image/jpeg',
      JPEG_QUALITY,
    );
  });
}

export function readAsDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('read_failed'));
    reader.readAsDataURL(file);
  });
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = src;
  });
}
