import heroImage from '../assets/hero-rice-panicle.jpg';
import { CropUploadCard } from '../components/CropUploadCard';

export function HomePage() {
  return (
    <div className="page">
      <h1 className="greeting">আপনার ফসলের ছবি দিয়ে রোগ শনাক্ত করুন।</h1>

      <div className="home-grid">
        <div className="hero">
          <img
            className="hero__img"
            src={heroImage}
            alt="সবুজ ধানের শীষ"
            width={2000}
            height={1500}
            decoding="async"
            fetchPriority="high"
          />
        </div>

        <CropUploadCard />
      </div>
    </div>
  );
}
