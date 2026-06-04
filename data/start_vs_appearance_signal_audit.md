# Start vs. appearance signal audit

Denne audit skiller startchance, appearance/indhop og availability tydeligere ad. Der er ikke koert optimizer, strategi-output eller frontend.

## Kort konklusion

- Maignans lave startchance skyldtes, at normal rotation/friendly-bench blev vaegtet som svagt startsignal. Den er rettet med eksisterende raw start-security og tidligere canonical layer som kilde.
- Schjelderup havde hoej appearance, men for hoej startchance. Start er nu skilt fra indhop.
- Manu Kone havde skadefravaer, der ikke slog haardt nok igennem. Han er nu high_risk med lavere availability/start.
- Patrick Wimmer og Ismaila Sarr var overpromoveret af canonical availability-splitten og er sat tilbage mod eksisterende raw national start-security.

## Sanity cases

| Spiller | Start foer | Start efter | Conditional foer | Conditional efter | Appearance efter | Risk efter | Handling |
|---|---:|---:|---:|---:|---:|---|---|
| Mike Maignan | 0.4842 | 0.82 | 0.5135 | 0.95 | 0.95 | low_risk | restore established-GK start signal from existing raw start-security row and prior canonical layer |
| Andreas Schjelderup | 0.6607 | 0.3333 | 0.7143 | 0.4545 | 0.7333 | medium_risk | separate appearance_prob from start_prob using documented squad/start/sub split |
| Manu Koné | 0.7771 | 0.38 | 0.8571 | 0.55 | 0.45 | high_risk | mark high_risk and cap start/appearance signals until injury context clears |
| Jurrien Timber | 0.6497 | 0.6497 | 0.7381 | 0.7381 |  | medium_risk | keep current signal; review only if new usage data arrives |
| Deniz Undav | 0.3287 | 0.3287 | 0.3684 | 0.3684 |  | medium_risk | keep current signal; review only if new usage data arrives |
| Ismael Saibari | 0.6814 | 0.6814 | 0.7857 | 0.7857 |  | high_risk | review injury/absence context before optimizer rerun |
| Ismaila Sarr | 0.7935 | 0.6015 | 0.8636 | 0.6526 | 0.768 | medium_risk | use existing raw country start-security row as safer start layer |
| Patrick Wimmer | 0.8397 | 0.5787 | 0.9 | 0.6477 | 0.8085 | medium_risk | use existing raw country start-security row as safer start layer |

## Rettede hoej-sikkerhedsfejl

- Mike Maignan: normal rotation/friendly bench was treated like weak starter evidence -> restore established-GK start signal from existing raw start-security row and prior canonical layer.
- Patrick Wimmer: canonical availability split over-promoted an uncertain starter -> use existing raw country start-security row as safer start layer.
- Manu Koné: recent injury absences were not strong enough availability negatives -> mark high_risk and cap start/appearance signals until injury context clears.
- Ismaila Sarr: canonical availability split over-promoted an uncertain starter -> use existing raw country start-security row as safer start layer.
- Andreas Schjelderup: sub appearances inflated start probability -> separate appearance_prob from start_prob using documented squad/start/sub split.

## Ikke rettet, men plausibelt markeret

- Deniz Undav: no high-confidence data error found; keep current signal; review only if new usage data arrives.
- Ismael Saibari: high start signal conflicts with high availability risk; review injury/absence context before optimizer rerun.
- Jurrien Timber: no high-confidence data error found; keep current signal; review only if new usage data arrives.

## Bubble-audit flags

- Foer denne start-audit: 27
- Efter genkoersel: 27

## Note om datagrundlag

De ra Transfermarkt-matchfiler, som de gamle batch/classification-scriptnavne peger paa, findes ikke i repoet. Rettelserne er derfor lagt som et lille dokumenteret context-override-lag og tilsluttet det eksisterende merge-script, saa fremtidige merge-koersler kan bevare samme skelnen mellem start, indhop og skadefravaer.
