# Start Probability Availability Split - Sanityrapport

Denne rapport er afledt af `player_pool_v1.json`, `player_start_security_nt.csv` og `start_probability_availability_split_report.csv`. Den ændrer ikke modeldata.

## 1. Dækning

| Kilde | Antal med conditional_start_prob + availability_prob | Total rækker/spillere |
| --- | --- | --- |
| player_pool_v1.json | 1207 | 1244 |
| player_start_security_nt.csv | 1207 | 2632 |
| start_probability_availability_split_report.csv | 1207 | 1213 |

## 2. Top 30 - new_start_prob steg mest

| Spiller | Land | Old | New | Delta | Conditional | Availability | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| David Cabezas | ECU | 0.0794 | 0.8427 | +0.7633 | 0.9000 | 0.8182 | medium_risk |
| Lukas Provod | HOLDET_584 | 0.2500 | 0.9439 | +0.6939 | 0.9439 |  | unknown |
| Tomas Soucek | HOLDET_584 | 0.2500 | 0.9219 | +0.6719 | 0.9394 | 0.9469 | low_risk |
| Gue-sung Cho | KOR | 0.1530 | 0.8209 | +0.6679 | 0.9048 | 0.7353 | medium_risk |
| Matt Turner | USA | 0.0183 | 0.6763 | +0.6580 | 0.7465 | 0.7315 | medium_risk |
| Mohamed Al-Mannai | QAT | 0.2500 | 0.8985 | +0.6485 | 0.9700 | 0.7895 | medium_risk |
| Tahith Chong | CUW | 0.2500 | 0.8985 | +0.6485 | 0.9700 | 0.7895 | medium_risk |
| Charles Pickel | COD | 0.1410 | 0.7895 | +0.6485 | 0.8438 | 0.8163 | medium_risk |
| In-beom Hwang | KOR | 0.2500 | 0.8970 | +0.6470 | 0.9700 | 0.7850 | medium_risk |
| Musab Al-Juwayr | KSA | 0.2500 | 0.8952 | +0.6452 | 0.9700 | 0.7797 | medium_risk |
| Kwasi Sibo | GHA | 0.2500 | 0.8851 | +0.6351 | 0.9700 | 0.7500 | medium_risk |
| Matej Kovar | HOLDET_584 | 0.2500 | 0.8843 | +0.6343 | 0.8843 |  | unknown |
| Yan Diomande | CIV | 0.2500 | 0.8814 | +0.6314 | 0.9700 | 0.7391 | medium_risk |
| Matthew Garbett | NZL | 0.2500 | 0.8803 | +0.6303 | 0.9700 | 0.7358 | medium_risk |
| Lucas Herrington | AUS | 0.2500 | 0.8795 | +0.6295 | 0.9700 | 0.7333 | medium_risk |
| Ime Okon | RSA | 0.2500 | 0.8795 | +0.6295 | 0.9700 | 0.7333 | medium_risk |
| Josue Casimir | HAI | 0.2500 | 0.8757 | +0.6257 | 0.9700 | 0.7222 | medium_risk |
| Ralph Priso | CAN | 0.2500 | 0.8730 | +0.6230 | 0.9700 | 0.7143 | unknown |
| Haissem Hassan | EGY | 0.2500 | 0.8730 | +0.6230 | 0.9700 | 0.7143 | unknown |
| Victor Munoz | ESP | 0.2500 | 0.8730 | +0.6230 | 0.9700 | 0.7143 | unknown |
| Nathan Ngoy | BEL | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Joan Garcia | ESP | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Lachlan Bayliss | NZL | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Mostafa Ziko | EGY | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Marcelo Flores | CAN | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Rayan Elloumi | TUN | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Hamza Abdelkarim | EGY | 0.2500 | 0.8655 | +0.6155 | 0.9700 | 0.6923 | unknown |
| Lucas Mendes | QAT | 0.2500 | 0.8649 | +0.6149 | 0.9545 | 0.7317 | medium_risk |
| Abdel Rahman Al-Talalga | JOR | 0.1115 | 0.7262 | +0.6147 | 0.7692 | 0.8400 | medium_risk |
| Alvaro Fidalgo | MEX | 0.2500 | 0.8639 | +0.6139 | 0.9700 | 0.6875 | medium_risk |

## 3. Top 30 - new_start_prob faldt mest

| Spiller | Land | Old | New | Delta | Conditional | Availability | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Orlando Mosquera | PAN | 0.9758 | 0.4613 | -0.5145 | 0.5169 | 0.6929 | medium_risk |
| Nawaf Al-Aqidi | KSA | 0.9800 | 0.4863 | -0.4937 | 0.5532 | 0.6548 | medium_risk |
| Joel Ordonez | ECU | 0.9800 | 0.4914 | -0.4886 | 0.5333 | 0.7755 | medium_risk |
| Ivan Basic | BIH | 0.9500 | 0.4913 | -0.4587 | 0.5385 | 0.7500 | medium_risk |
| Ken Sema | SWE | 0.8280 | 0.3948 | -0.4332 | 0.4590 | 0.6000 | high_risk |
| Juan Quintero | COL | 0.4520 | 0.0446 | -0.4074 | 0.0500 | 0.6923 | unknown |
| Jeremy Arevalo | ECU | 0.4410 | 0.0446 | -0.3964 | 0.0500 | 0.6923 | unknown |
| Gregor Kobel | SUI | 0.8100 | 0.4300 | -0.3800 | 0.5000 | 0.6000 | high_risk |
| Ali Yousif | IRQ | 0.4180 | 0.0386 | -0.3794 | 0.0500 | 0.3500 | high_risk |
| Jonathan Tah | GER | 0.9426 | 0.5671 | -0.3755 | 0.6395 | 0.6763 | medium_risk |
| Hicham Boudaoui | ALG | 0.8800 | 0.5192 | -0.3608 | 0.6038 | 0.6000 | high_risk |
| Alan Franco | ECU | 0.8100 | 0.4500 | -0.3600 | 0.5000 | 0.7143 | unknown |
| Chemsdine Talbi | MAR | 0.4290 | 0.0792 | -0.3498 | 0.0909 | 0.6333 | high_risk |
| Anthony Ralston | SCO | 0.9800 | 0.6311 | -0.3489 | 0.7200 | 0.6471 | high_risk |
| Patrick Pentz | AUT | 0.7300 | 0.3832 | -0.3468 | 0.4048 | 0.8475 | medium_risk |
| Alexis Duarte | PAR | 0.4560 | 0.1104 | -0.3456 | 0.1429 | 0.3500 | high_risk |
| Mike Maignan | FRA | 0.8200 | 0.4842 | -0.3358 | 0.5135 | 0.8367 | medium_risk |
| Finn Surman | NZL | 0.9800 | 0.6455 | -0.3345 | 0.7143 | 0.7250 | medium_risk |
| Aymen Dahmen | TUN | 0.8265 | 0.5017 | -0.3248 | 0.5634 | 0.6870 | medium_risk |
| John Mercado | ECU | 0.4200 | 0.0987 | -0.3213 | 0.1111 | 0.6800 | medium_risk |
| John Souttar | SCO | 0.8340 | 0.5160 | -0.3180 | 0.6000 | 0.6000 | high_risk |
| Elisha Owusu | GHA | 0.8370 | 0.5235 | -0.3135 | 0.6087 | 0.6000 | high_risk |
| Nikola Vasilj | BIH | 0.7827 | 0.4836 | -0.2991 | 0.5000 | 0.9062 | medium_risk |
| Luc De Fougerolles | CAN | 0.6340 | 0.3375 | -0.2965 | 0.3810 | 0.6744 | medium_risk |
| Mathieu Choiniere | CAN | 0.7680 | 0.4739 | -0.2941 | 0.5455 | 0.6250 | high_risk |
| Joel Waterman | CAN | 0.5800 | 0.2867 | -0.2933 | 0.3333 | 0.6000 | high_risk |
| Yaimar Medina | ECU | 0.4540 | 0.1614 | -0.2926 | 0.1818 | 0.6786 | medium_risk |
| Martin Experience | HAI | 0.8250 | 0.5324 | -0.2926 | 0.6190 | 0.6000 | high_risk |
| Kenny McLean | SCO | 0.9440 | 0.6629 | -0.2811 | 0.7708 | 0.6000 | high_risk |
| Ørjan Nyland | NOR | 0.9000 | 0.6193 | -0.2807 | 0.6667 | 0.7971 | medium_risk |

## 4. Top 30 - høj conditional_start_prob, lav availability_prob

| Spiller | Land | Old | New | Delta | Conditional | Availability | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Assane Diao Diaoune | SEN | 0.2500 | 0.7493 | +0.4993 | 0.9700 | 0.3500 | high_risk |
| Jovo Lukic | BIH | 0.2500 | 0.7493 | +0.4993 | 0.9700 | 0.3500 | high_risk |
| Alexander Bernhardsson | SWE | 0.3760 | 0.6928 | +0.3168 | 0.8889 | 0.3696 | high_risk |
| Mohamed Toure | AUS | 0.3717 | 0.6883 | +0.3166 | 0.8750 | 0.3902 | high_risk |
| Jeremy Antonisse | CUW | 0.4800 | 0.7223 | +0.2423 | 0.9091 | 0.4130 | high_risk |
| Yuito Suzuki | JPN | 0.4378 | 0.6857 | +0.2479 | 0.8571 | 0.4286 | high_risk |
| Jürgen Locadia | CUW | 0.2500 | 0.7207 | +0.4707 | 0.8889 | 0.4595 | high_risk |
| Hossein Abarghouei | IRN | 0.4880 | 0.7913 | +0.3033 | 0.9700 | 0.4737 | high_risk |
| Mustafa Eskihellac | TUR | 0.4070 | 0.8002 | +0.3932 | 0.9700 | 0.5000 | high_risk |
| Noni Madueke | ENG | 0.4300 | 0.7302 | +0.3002 | 0.8571 | 0.5769 | high_risk |
| Adalberto Carrasquilla | PAN | 0.2500 | 0.8342 | +0.5842 | 0.9700 | 0.6000 | high_risk |
| Ahmed Nadhir Benbouali | ALG | 0.2500 | 0.8342 | +0.5842 | 0.9700 | 0.6000 | high_risk |
| Folarin Balogun | USA | 0.3006 | 0.8342 | +0.5336 | 0.9700 | 0.6000 | high_risk |
| Merchas Doski | IRQ | 0.8750 | 0.8342 | -0.0408 | 0.9700 | 0.6000 | high_risk |
| Mohamed Rabie Hrimat | MAR | 0.2500 | 0.8342 | +0.5842 | 0.9700 | 0.6000 | high_risk |
| Mohammad Mohebi | IRN | 0.6189 | 0.8342 | +0.2153 | 0.9700 | 0.6000 | high_risk |
| Ryan Thomas | NZL | 0.5287 | 0.8342 | +0.3055 | 0.9700 | 0.6000 | high_risk |
| Yoane Wissa | COD | 0.4091 | 0.8342 | +0.4251 | 0.9700 | 0.6000 | high_risk |
| Yoel Barcenas | PAN | 0.2500 | 0.8342 | +0.5842 | 0.9700 | 0.6000 | high_risk |
| Kosta Barbarouses | NZL | 0.4540 | 0.8303 | +0.3763 | 0.9655 | 0.6000 | high_risk |
| Tyler Adams | USA | 0.4913 | 0.8298 | +0.3385 | 0.9649 | 0.6000 | high_risk |
| Elias Achouri | TUN | 0.5417 | 0.8269 | +0.2852 | 0.9615 | 0.6000 | high_risk |
| Salem Al-Dawsari | KSA | 0.5574 | 0.8245 | +0.2671 | 0.9588 | 0.6000 | high_risk |
| Iñaki Williams | GHA | 0.3250 | 0.8226 | +0.4976 | 0.9565 | 0.6000 | high_risk |
| Weston McKennie | USA | 0.4398 | 0.8215 | +0.3817 | 0.9552 | 0.6000 | high_risk |
| Hakim Ziyech | MAR | 0.2500 | 0.8184 | +0.5684 | 0.9516 | 0.6000 | high_risk |
| Viktor Gyökeres | SWE | 0.5828 | 0.8180 | +0.2352 | 0.9512 | 0.6000 | high_risk |
| Amine Gouiri | ALG | 0.2992 | 0.8170 | +0.5178 | 0.9500 | 0.6000 | high_risk |
| Christian Pulisic | USA | 0.4692 | 0.8170 | +0.3478 | 0.9500 | 0.6000 | high_risk |
| Ben Old | NZL | 0.5620 | 0.8147 | +0.2527 | 0.9474 | 0.6000 | high_risk |

## 5. Eksempler på profiler der håndteres bedre

| Spiller | Land | Old | New | Delta | Conditional | Availability | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cristian Romero | ARG | 0.7715 | 0.8499 | +0.0784 | 0.9259 | 0.7654 | medium_risk |
| Erling Haaland | NOR | 0.9700 | 0.8883 | -0.0817 | 0.9700 | 0.7595 | medium_risk |
| Antonio Nusa | NOR | 0.7000 | 0.8157 | +0.1157 | 0.8889 | 0.7647 | medium_risk |
| Manuel Neuer | GER | 0.9500 | 0.8454 | -0.1046 | 0.8988 | 0.8302 | medium_risk |
| Oliver Baumann | GER | 0.0200 | 0.3420 | +0.3220 | 0.3878 | 0.6628 | medium_risk |
| Assane Diao Diaoune | SEN | 0.2500 | 0.7493 | +0.4993 | 0.9700 | 0.3500 | high_risk |
| Jovo Lukic | BIH | 0.2500 | 0.7493 | +0.4993 | 0.9700 | 0.3500 | high_risk |
| Jeremy Antonisse | CUW | 0.4800 | 0.7223 | +0.2423 | 0.9091 | 0.4130 | high_risk |

Cristian Romero er et godt sanity-eksempel, fordi hans conditional_start_prob ligger højt, mens availability holdes separat.

## 6. Antal spillere pr. availability_risk

| availability_risk | Antal |
| --- | --- |
| high_risk | 512 |
| low_risk | 57 |
| medium_risk | 592 |
| unknown | 52 |

## 7. Min/max/mean

| Felt | N | Min | Max | Mean |
| --- | --- | --- | --- | --- |
| old_start_prob | 1213 | 0.0000 | 0.9832 | 0.4772 |
| new_start_prob | 1213 | 0.0386 | 0.9470 | 0.6153 |
| conditional_start_prob | 1213 | 0.0500 | 0.9700 | 0.6891 |
| availability_prob | 1207 | 0.3500 | 0.9469 | 0.6832 |

## 8. Kort vurdering

- Risk-labels er nu mindre ekstreme: medium bliver standardområdet for spillere med rimelig availability, mens high i højere grad markerer lav availability eller tydelige absence-signaler.
- Startformlen og selve sandsynlighedsfelterne er ikke ændret af denne sanityrapport.
- Overordnet ser kalibreringen mere brugbar ud som advarselslabel end den tidligere fordeling, hvor high_risk dominerede for kraftigt.
