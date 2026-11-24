import { ImagePlus, Leaf } from 'lucide-react';
import { useId, useRef, useState } from 'react';

interface ImageDropzoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED = 'image/jpeg,image/png,image/webp,image/*';

export function ImageDropzone({ onFile, disabled = false }: ImageDropzoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) onFile(file);
  }

  return (
    <div>
      <input
        ref={inputRef}
        id={inputId}
        className="sr-only"
        type="file"
        accept={ACCEPTED}
        capture="environment"
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = '';
        }}
      />

      <button
        type="button"
        className={isDragging ? 'dropzone dropzone--drag' : 'dropzone'}
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (!disabled) handleFiles(event.dataTransfer.files);
        }}
      >
        <span className="dropzone__frame" aria-hidden="true">
          <span className="dropzone__corner dropzone__corner--tl" />
          <span className="dropzone__corner dropzone__corner--tr" />
          <span className="dropzone__corner dropzone__corner--bl" />
          <span className="dropzone__corner dropzone__corner--br" />
          <span className="dropzone__icon">
            <ImagePlus size={22} strokeWidth={1.9} />
          </span>
          <span className="dropzone__leaf-hint">
            <Leaf size={13} strokeWidth={2} />
            একটি পাতা ফ্রেমের মাঝে রাখুন
          </span>
        </span>
        <span className="dropzone__label">ছবি নির্বাচন করুন</span>
        <span className="dropzone__helper">ছবি নির্বাচন করুন অথবা এখানে টেনে আনুন</span>
        <span className="dropzone__tip">
          ভালো আলোতে একটি আক্রান্ত পাতা ফ্রেমের ভেতরে রেখে ছবি তুলুন
        </span>
      </button>
    </div>
  );
}
