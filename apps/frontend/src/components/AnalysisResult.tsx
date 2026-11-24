import type { AnalysisResult } from '@foshol-lens/contracts';
import {
  AlertTriangle,
  BookOpen,
  ChevronLeft,
  MessageSquare,
  ShieldAlert,
  UserCheck,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { mediaUrl } from '../lib/api-client';
import { buildSpeechText, formatBengaliDate } from '../lib/format';
import { AudioPlayer } from './AudioPlayer';
import { SpeakCardButton } from './SpeakCardButton';
import { WeatherCard } from './WeatherCard';

function buildTreatmentSpeech(result: AnalysisResult): string {
  if (!result.treatment) return '';
  return buildSpeechText([
    result.treatment.summary,
    ...result.treatment.steps,
    result.ppeReminder ? `নিরাপত্তা নির্দেশনা: ${result.ppeReminder}` : null,
  ]);
}

export function AnalysisResultView({ result }: { result: AnalysisResult }) {
  const { diagnosis, uncertainty, narrative } = result;
  const imageSrc = mediaUrl(result.imageUrl);
  const treatmentSpeech = buildTreatmentSpeech(result);

  return (
    <div className="page">
      <Link to="/history" className="back-link">
        <ChevronLeft size={17} aria-hidden="true" />
        ইতিহাসে ফিরে যান
      </Link>

      <h1 className="page-heading" style={{ marginTop: 12 }}>
        বিশ্লেষণের ফলাফল
      </h1>
      <p className="page-subtitle">
        {result.cropName} • {formatBengaliDate(result.createdAt)}
      </p>

      <div className="result">
        {imageSrc && (
          <div className="result__image">
            <img src={imageSrc} alt="বিশ্লেষণ করা ফসলের ছবি" />
          </div>
        )}

        <div className="result__body">
          {uncertainty.isUncertain && (
            <div className="notice">
              <div>
                <p className="notice__title">রোগ নিশ্চিতভাবে শনাক্ত করা যায়নি</p>
                <p className="notice__text">
                  {uncertainty.message ||
                    'ছবিটি থেকে নিশ্চিতভাবে রোগ শনাক্ত করা যায়নি। অনুগ্রহ করে আরও পরিষ্কার ছবি দিন।'}
                </p>
                <Link to="/" className="button-secondary" style={{ marginTop: 14 }}>
                  নতুন ছবি দিন
                </Link>
              </div>
            </div>
          )}

          {result.farmerQuestion && (
            <div className="question-chip" aria-label="আপনার প্রশ্ন">
              <div className="question-chip__label">
                <MessageSquare size={15} strokeWidth={2.1} aria-hidden="true" />
                <span>আপনার প্রশ্ন</span>
              </div>
              <p className="question-chip__text">{result.farmerQuestion}</p>
            </div>
          )}

          {result.cropMismatch?.detected && (
            <div className="notice mismatch-banner" role="alert">
              <div>
                <div className="notice__header">
                  <AlertTriangle size={18} strokeWidth={2.2} aria-hidden="true" />
                  <p className="notice__title">ফসলের অমিল শনাক্ত হয়েছে</p>
                </div>
                <p className="notice__text">{result.cropMismatch.messageBn}</p>
                <div className="mismatch-banner__meta">
                  <span className="badge badge--warn">
                    ছবিতে: {result.cropMismatch.photoCropBn}
                  </span>
                  <span className="badge badge--warn">
                    প্রশ্নে: {result.cropMismatch.questionCropBn}
                  </span>
                </div>
              </div>
            </div>
          )}

          {narrative && (
            <section className="card card--narrative">
              <p className="card__eyebrow">আপনার জন্য পরামর্শ</p>
              <p className="narrative">{narrative}</p>
              {result.audio && <AudioPlayer audio={result.audio} analysisId={result.id} />}
            </section>
          )}

          {diagnosis && !uncertainty.isUncertain && (
            <section className="card">
              <p className="card__eyebrow">সম্ভাব্য রোগ</p>
              <h2 className="card__title">{diagnosis.diseaseLabel}</h2>
              {diagnosis.confidence !== null && (
                <p className="confidence__note">
                  ছবি বিশ্লেষণে মিলেছে {Math.round(diagnosis.confidence * 100)}% নিশ্চয়তায়।
                </p>
              )}
              <p className="confidence__note">এটি একটি AI-উৎপন্ন আনুমানিক ফলাফল।</p>
            </section>
          )}

          {result.symptoms.length > 0 && (
            <section className="card">
              <h2 className="card__label">লক্ষণ</h2>
              <ul className="list">
                {result.symptoms.map((symptom) => (
                  <li key={symptom}>{symptom}</li>
                ))}
              </ul>
            </section>
          )}

          {result.treatment && (
            <section className="card card--plain">
              <div className="card__header-row">
                <h2 className="card__label">
                  {uncertainty.isUncertain ? 'সাধারণ পরামর্শ' : 'প্রস্তাবিত চিকিৎসা'}
                </h2>
                {treatmentSpeech && (
                  <SpeakCardButton
                    text={treatmentSpeech}
                    label="শুনুন"
                    ariaLabel="প্রস্তাবিত চিকিৎসা শুনুন"
                  />
                )}
              </div>
              <p className="card__text">{result.treatment.summary}</p>
              <ul className="list">
                {result.treatment.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ul>
              {result.treatment.source && (
                <p className="confidence__note">তথ্যসূত্র: {result.treatment.source}</p>
              )}
            </section>
          )}

          {result.ppeReminder && (
            <section className="card card--warn" aria-label="নিরাপত্তা নির্দেশনা">
              <div className="ppe-card__header">
                <ShieldAlert size={18} strokeWidth={2.1} aria-hidden="true" />
                <h2 className="card__label">নিরাপত্তা নির্দেশনা (PPE)</h2>
              </div>
              <p className="card__text">{result.ppeReminder}</p>
            </section>
          )}

          {result.weather ? (
            <WeatherCard weather={result.weather} />
          ) : (
            // The district was chosen, so silence looks like a bug. Being
            // explicit also avoids implying the weather was checked.
            result.district && (
              <section className="card card--plain">
                <p className="card__eyebrow">{result.district} জেলার আবহাওয়া</p>
                <p className="card__text">
                  আবহাওয়ার তথ্য এখন পাওয়া যাচ্ছে না। স্প্রে করার আগে আকাশের দিকে তাকিয়ে
                  দেখুন।
                </p>
              </section>
            )
          )}

          {result.prevention.length > 0 && (
            <section className="card">
              <h2 className="card__label">প্রতিরোধ</h2>
              <ul className="list">
                {result.prevention.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
          )}

          {result.ragSources && result.ragSources.length > 0 && (
            <section className="card card--plain" aria-label="BARI ও BRRI তথ্যসূত্র">
              <div className="rag-sources__header">
                <BookOpen size={17} strokeWidth={2.1} aria-hidden="true" />
                <h2 className="card__label">তথ্যসূত্র (BARI/BRRI)</h2>
                <span className="badge badge--ok">যাচাইকৃত নির্দেশিকা</span>
              </div>
              <ul className="rag-sources__list">
                {result.ragSources.map((src) => (
                  <li key={src.chunkId} className="rag-source">
                    <div className="rag-source__meta">
                      <span className="rag-source__title">{src.sourceTitle}</span>
                      {src.culturalOnly && (
                        <span className="badge badge--warn">রাসায়নিকমুক্ত পরামর্শ</span>
                      )}
                    </div>
                    <p className="rag-source__section">{src.section}</p>
                    <p className="rag-source__excerpt">{src.excerpt}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {result.referToOfficer ? (
            <div className="notice" role="note">
              <div>
                <div className="notice__header">
                  <UserCheck size={18} strokeWidth={2.1} aria-hidden="true" />
                  <p className="notice__title">
                    উপ-সহকারী কৃষি কর্মকর্তা (SAAO)-এর পরামর্শ নিন
                  </p>
                </div>
                <p className="notice__text">
                  {uncertainty.isUncertain
                    ? 'রোগটি নিশ্চিতভাবে শনাক্ত করা যায়নি। আন্দাজে কোনো ওষুধ স্প্রে না করে সরাসরি স্থানীয় উপ-সহকারী কৃষি কর্মকর্তা (SAAO) অথবা উপজেলা কৃষি অফিসে যোগাযোগ করুন।'
                    : result.cropMismatch?.detected
                      ? 'ছবি ও প্রশ্নের ফসলে অমিল থাকায় সঠিক ফসল যাচাই করে ব্যবস্থা নিতে স্থানীয় উপ-সহকারী কৃষি কর্মকর্তা (SAAO) অথবা উপজেলা কৃষি অফিসে যোগাযোগ করুন।'
                      : 'এই রোগে সাধারণ ছত্রাকনাশক স্প্রে কার্যকর নয়। মাঠ পর্যায়ের সঠিক ব্যবস্থাপনার জন্য দ্রুত স্থানীয় উপ-সহকারী কৃষি কর্মকর্তা (SAAO)-এর সাথে পরামর্শ করুন।'}
                </p>
              </div>
            </div>
          ) : (
            result.needsProfessionalVerification &&
            !uncertainty.isUncertain && (
              <div className="notice">
                <div>
                  <p className="notice__title">পেশাদার যাচাই প্রয়োজন</p>
                  <p className="notice__text">
                    চূড়ান্ত সিদ্ধান্তের জন্য স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।
                  </p>
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
