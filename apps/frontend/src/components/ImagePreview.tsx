import { RefreshCw } from 'lucide-react';

interface ImagePreviewProps {
  src: string;
  alt: string;
  onReplace: () => void;
  disabled?: boolean;
}

export function ImagePreview({ src, alt, onReplace, disabled = false }: ImagePreviewProps) {
  return (
    <div className="preview">
      <img className="preview__img" src={src} alt={alt} />
      <div style={{ textAlign: 'center' }}>
        <button type="button" className="preview__replace" onClick={onReplace} disabled={disabled}>
          <RefreshCw size={14} strokeWidth={2.2} aria-hidden="true" />
          ছবি পরিবর্তন করুন
        </button>
      </div>
    </div>
  );
}
