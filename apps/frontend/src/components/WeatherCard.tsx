import type {
  HourlySprayWindow,
  LocationCandidate,
  SpraySafety,
  Weather,
} from '@foshol-lens/contracts';
import { Building2, Clock, CloudRain, MapPin, Mic, Wind } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../lib/api-client';
import { buildSpeechText, toBengaliDigits } from '../lib/format';
import { SpeakCardButton } from './SpeakCardButton';

const SAFETY_LABELS: Record<SpraySafety, string> = {
  green: 'নিরাপদ',
  amber: 'সতর্কতা',
  red: 'ঝুঁকিপূর্ণ',
};

function parseHour(timeIso: string): number {
  const match = timeIso.match(/T(\d{2}):/);
  return match ? Number(match[1]) : 0;
}

function formatHourBn(hour: number): string {
  if (hour === 0) return 'রাত ১২টা';
  if (hour < 6) return `ভোর ${toBengaliDigits(hour)}টা`;
  if (hour < 12) return `সকাল ${toBengaliDigits(hour)}টা`;
  if (hour === 12) return 'দুপুর ১২টা';
  if (hour <= 14) return `দুপুর ${toBengaliDigits(hour - 12)}টা`;
  if (hour < 18) return `বিকেল ${toBengaliDigits(hour - 12)}টা`;
  if (hour === 18) return 'সন্ধ্যা ৬টা';
  return `রাত ${toBengaliDigits(hour - 12)}টা`;
}

function formatShortHourBn(hour: number): string {
  const h12 = hour % 12 === 0 ? 12 : hour % 12;
  return toBengaliDigits(h12);
}

function dayLabelBn(dayIndex: number): string {
  if (dayIndex === 0) return 'আজ';
  if (dayIndex === 1) return 'আগামীকাল';
  return 'পরশু';
}

interface DayGroup {
  dateKey: string;
  label: string;
  slots: Array<{ window: HourlySprayWindow; index: number; hour: number }>;
}

function groupByDay(windows: HourlySprayWindow[]): DayGroup[] {
  const groups: DayGroup[] = [];
  for (let i = 0; i < windows.length; i += 1) {
    const win = windows[i];
    if (!win) continue;
    const dateKey = win.timeIso.slice(0, 10) || String(Math.floor(i / 24));
    const hour = parseHour(win.timeIso);
    let group = groups.find((g) => g.dateKey === dateKey);
    if (!group) {
      group = {
        dateKey,
        label: dayLabelBn(groups.length),
        slots: [],
      };
      groups.push(group);
    }
    group.slots.push({ window: win, index: i, hour });
  }
  return groups;
}

function defaultSelectedIndex(windows: HourlySprayWindow[]): number {
  const safeDayIdx = windows.findIndex((win) => {
    const h = parseHour(win.timeIso);
    return win.safety === 'green' && h >= 6 && h < 18;
  });
  return safeDayIdx >= 0 ? safeDayIdx : 0;
}

function buildWeatherSpeech(weather: Weather): string {
  const placeName = weather.resolvedPlace || weather.district;
  const locationLine =
    weather.locationSource === 'voice' || weather.locationSource === 'gps'
      ? `${placeName} এলাকার আবহাওয়া।`
      : `${weather.district} জেলার আবহাওয়া।`;
  const verdictLine = weather.sprayWindowOk
    ? 'এখন স্প্রে করার অনুকূল সময়।'
    : 'এখনই স্প্রে করবেন না।';
  return buildSpeechText([
    locationLine,
    verdictLine,
    weather.reason,
    weather.safestWindowSummaryBn,
  ]);
}

function isActiveCandidate(candidate: LocationCandidate, weather: Weather): boolean {
  if (weather.resolvedPlace && candidate.label === weather.resolvedPlace) {
    return true;
  }
  if (
    typeof weather.gridLatitude === 'number' &&
    typeof weather.gridLongitude === 'number' &&
    Math.abs(candidate.latitude - weather.gridLatitude) < 0.25 &&
    Math.abs(candidate.longitude - weather.gridLongitude) < 0.25 &&
    candidate.district === weather.district
  ) {
    return true;
  }
  return false;
}

/**
 * Weather facts plus the deterministic 12h verdict and 48h spray timeline.
 *
 * The verdict and hourly slots are computed server-side in code. The LLM only
 * phrases them, so this card and the spoken narrative can never disagree about
 * whether spraying is sensible.
 */
export function WeatherCard({ weather }: { weather: Weather }) {
  const [activeWeather, setActiveWeather] = useState<Weather>(weather);
  const [switchingLabel, setSwitchingLabel] = useState<string | null>(null);
  const [switchError, setSwitchError] = useState<string | null>(null);

  useEffect(() => {
    setActiveWeather(weather);
    setSwitchingLabel(null);
    setSwitchError(null);
  }, [weather]);

  const blocking = !activeWeather.sprayWindowOk;
  const windows = activeWeather.hourlyWindows ?? [];
  const [selectedIndex, setSelectedIndex] = useState(() => defaultSelectedIndex(windows));

  useEffect(() => {
    setSelectedIndex(defaultSelectedIndex(windows));
  }, [activeWeather.district, activeWeather.resolvedPlace, windows.length]);

  const candidates =
    weather.locationCandidates && weather.locationCandidates.length > 0
      ? weather.locationCandidates
      : (activeWeather.locationCandidates ?? []);

  async function handleSelectCandidate(candidate: LocationCandidate) {
    if (switchingLabel) return;
    setSwitchingLabel(candidate.label);
    setSwitchError(null);
    try {
      const updated = await api.getWeather(
        candidate.latitude,
        candidate.longitude,
        candidate.label,
      );
      setActiveWeather({
        ...updated,
        locationSource: 'voice',
        resolvedPlace: updated.resolvedPlace ?? candidate.label,
        locationCandidates: candidates,
      });
    } catch {
      setSwitchError('এই এলাকার আবহাওয়া এখন লোড করা যায়নি। আবার চেষ্টা করুন।');
    } finally {
      setSwitchingLabel(null);
    }
  }

  const dayGroups = groupByDay(windows);
  const selectedSlot = windows[selectedIndex] ?? windows[0];
  const selectedDayLabel =
    dayGroups.find((group) => group.slots.some((slot) => slot.index === selectedIndex))?.label ??
    'আজ';
  const weatherSpeech = buildWeatherSpeech(activeWeather);

  const headerEyebrow =
    activeWeather.locationSource === 'voice'
      ? `${activeWeather.resolvedPlace || activeWeather.district} — আবহাওয়া`
      : activeWeather.locationSource === 'gps'
        ? `${activeWeather.district} — আবহাওয়া`
        : `${activeWeather.district} জেলার আবহাওয়া`;

  return (
    <section className={`card ${blocking ? 'card--warn' : 'card--plain'}`}>
      <div className="card__header-row">
        <div>
          <p className="card__eyebrow">{headerEyebrow}</p>

          {activeWeather.locationSource === 'voice' && (
            <span className="voice-location-badge">
              <Mic size={12} strokeWidth={2.2} aria-hidden="true" />
              ভয়েস থেকে শনাক্তকৃত এলাকা
            </span>
          )}

          <h2 className="card__label">
            {activeWeather.sprayWindowOk ? 'এখন স্প্রে করার অনুকূল সময়' : 'এখনই স্প্রে করবেন না'}
          </h2>
        </div>

        <SpeakCardButton
          text={weatherSpeech}
          label="শুনুন"
          ariaLabel="আবহাওয়া ও স্প্রে নির্দেশনা শুনুন"
        />
      </div>

      <p className="card__text">{activeWeather.reason}।</p>

      {(activeWeather.resolvedPlace ||
        (typeof activeWeather.gridLatitude === 'number' &&
          typeof activeWeather.gridLongitude === 'number')) && (
        <div className="weather-grid-meta">
          <MapPin size={14} strokeWidth={2.1} aria-hidden="true" />
          <span>
            {activeWeather.resolvedPlace ? `${activeWeather.resolvedPlace}` : activeWeather.district}
            {typeof activeWeather.gridLatitude === 'number' &&
              typeof activeWeather.gridLongitude === 'number' &&
              ` • গ্রিড: ${toBengaliDigits(activeWeather.gridLatitude.toFixed(4))}, ${toBengaliDigits(activeWeather.gridLongitude.toFixed(4))}`}
          </span>
        </div>
      )}

      {activeWeather.extensionOfficeBn && (
        <div className="weather-extension-office">
          <Building2 size={14} strokeWidth={2.1} aria-hidden="true" />
          <span>{activeWeather.extensionOfficeBn}</span>
        </div>
      )}

      {candidates.length > 1 && (
        <div className="location-candidates" aria-label="অন্য এলাকা নির্বাচন">
          <p className="location-candidates__label">অন্য এলাকা? পরিবর্তন করুন:</p>
          <div className="location-candidates__chips">
            {candidates.map((cand) => {
              const active = isActiveCandidate(cand, activeWeather);
              const loadingThis = switchingLabel === cand.label;
              return (
                <button
                  key={`${cand.label}-${cand.latitude}-${cand.longitude}`}
                  type="button"
                  className={`location-chip${active ? ' location-chip--active' : ''}`}
                  aria-pressed={active}
                  disabled={switchingLabel !== null}
                  onClick={() => handleSelectCandidate(cand)}
                >
                  {loadingThis ? 'লোড হচ্ছে...' : cand.label}
                </button>
              );
            })}
          </div>
          {switchError && <p className="location-candidates__error">{switchError}</p>}
        </div>
      )}

      <div className="weather-facts">
        {typeof activeWeather.rainProbabilityNext12h === 'number' && (
          <span className="weather-fact">
            <CloudRain size={17} strokeWidth={2.1} aria-hidden="true" />
            বৃষ্টির সম্ভাবনা {toBengaliDigits(activeWeather.rainProbabilityNext12h)}%
          </span>
        )}
        {typeof activeWeather.windSpeedMax12h === 'number' && (
          <span className="weather-fact">
            <Wind size={17} strokeWidth={2.1} aria-hidden="true" />
            বাতাস {toBengaliDigits(Math.round(activeWeather.windSpeedMax12h))} কিমি/ঘণ্টা
          </span>
        )}
      </div>

      {windows.length > 0 && (
        <div className="spray-timeline" aria-label="৪৮ ঘণ্টার স্প্রে সময়সূচি">
          <div className="spray-timeline__header">
            <h3 className="spray-timeline__title">৪৮ ঘণ্টার স্প্রে সময়সূচি</h3>
            <div className="spray-timeline__legend" aria-hidden="true">
              <span className="spray-legend-item">
                <span className="spray-dot spray-dot--green" />
                নিরাপদ
              </span>
              <span className="spray-legend-item">
                <span className="spray-dot spray-dot--amber" />
                সতর্কতা
              </span>
              <span className="spray-legend-item">
                <span className="spray-dot spray-dot--red" />
                ঝুঁকিপূর্ণ
              </span>
            </div>
          </div>

          {activeWeather.safestWindowSummaryBn && (
            <div className="spray-timeline__safest">
              <Clock size={15} strokeWidth={2.1} aria-hidden="true" />
              <span>{activeWeather.safestWindowSummaryBn}</span>
            </div>
          )}

          <div className="spray-timeline__days">
            {dayGroups.map((group) => (
              <div key={group.dateKey} className="spray-timeline__day">
                <span className="spray-timeline__day-label">{group.label}</span>
                <div className="spray-timeline__grid" role="list">
                  {group.slots.map(({ window: slot, index, hour }) => {
                    const isSelected = index === selectedIndex;
                    const hourLabel = formatHourBn(hour);
                    const safetyLabel = SAFETY_LABELS[slot.safety];
                    return (
                      <button
                        key={`${slot.timeIso}-${index}`}
                        type="button"
                        role="listitem"
                        className={`spray-slot spray-slot--${slot.safety}${
                          isSelected ? ' spray-slot--selected' : ''
                        }`}
                        title={`${group.label} ${hourLabel} — ${safetyLabel}`}
                        aria-label={`${group.label} ${hourLabel}: ${safetyLabel}, বৃষ্টি ${toBengaliDigits(slot.rainProb)}%`}
                        aria-pressed={isSelected}
                        onClick={() => setSelectedIndex(index)}
                        onMouseEnter={() => setSelectedIndex(index)}
                        onFocus={() => setSelectedIndex(index)}
                      >
                        <span className="spray-slot__hour">{formatShortHourBn(hour)}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {selectedSlot && (
            <div className="spray-timeline__detail" aria-live="polite">
              <span className="spray-timeline__detail-time">
                {selectedDayLabel} {formatHourBn(parseHour(selectedSlot.timeIso))}
              </span>
              <span className={`spray-badge spray-badge--${selectedSlot.safety}`}>
                {SAFETY_LABELS[selectedSlot.safety]}
              </span>
              <span className="spray-timeline__detail-metrics">
                বৃষ্টি {toBengaliDigits(selectedSlot.rainProb)}% ({toBengaliDigits(selectedSlot.rainMm)} মিমি) • বাতাস{' '}
                {toBengaliDigits(Math.round(selectedSlot.windKmh))} কিমি/ঘণ্টা
              </span>
            </div>
          )}
        </div>
      )}

      {activeWeather.source && <p className="confidence__note">তথ্যসূত্র: {activeWeather.source}</p>}
    </section>
  );
}
