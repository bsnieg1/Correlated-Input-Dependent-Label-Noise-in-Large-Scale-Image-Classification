# CLAUDE.md — praca inżynierska: HET (heteroskedastyczny szum etykiet)

Kontekst dla Claude Code. Projekt: praca inżynierska Bartka (telekomunikacja, AGH Kraków) oparta na
**Collier i in., „Correlated Input-Dependent Label Noise in Large-Scale Image Classification”, CVPR 2021**
(arXiv:2105.10305). Język pracy i komunikacji: **polski**. Kod, nazwy w kodzie i komentarze: angielski.

Stan na 2026-10-28: **etap planowania zakończony, kodu jeszcze nie ma.** Wszystko poniżej oznaczone
„planowane” to ustalenia, nie istniejące pliki.

---

## 1. Cel pracy

Zaimplementować warstwę wyjściową HET i sprawdzić, czy sieć, która **uczy się, gdzie ludzie mylą etykiety**,
działa lepiej niż zwykła sieć, gdy część etykiet treningowych jest błędna.

Pytania badawcze:
1. Czy HET daje wyższą dokładność na **czystym** zbiorze testowym niż zwykła sieć (CE)?
2. Czy jest lepiej skalibrowany (NLL, ECE)?
3. Czy sam odnajduje właściwe pary mylonych klas (MNIST: 1–7, 4–9, 3–8; ANIMAL-10N: 5 par zwierząt)?
4. **Główne pytanie:** czy korelacje między klasami (macierz V) dają coś ponad sam rozrzut (HET-diag, R=0)?
   W artykule ta przewaga była najmniejsza: **+0,6 p.p.** na ImageNet (78,7 → 79,3).
5. Czy model wskazuje błędnie oznaczone przykłady (AUROC)?

Sukcesem jest każdy rzetelny wynik — także „HET wygrywa nieznacznie albo wcale w małej skali”.
Praca to **weryfikacja metody**, nie dowód, że jest lepsza.

Proponowany tytuł: *Modelowanie skorelowanego, zależnego od wejścia szumu etykiet w klasyfikacji obrazów
za pomocą heteroskedastycznej warstwy wyjściowej.*

## 2. Założenia

- Błędy etykiet nie są losowe: dotyczą podobnych klas i częściej trudnych zdjęć (szum NNAR wg Frénay i Verleysen 2014 — zależny od klasy **i** od cech obrazu).
- Zbiór testowy jest czysty; walidacja jest zaszumiona (realistyczny scenariusz).
- W parze < 50% zamienionych etykiet (przy 50% klasy są nierozróżnialne); górna granica w eksperymentach: 45%.
- Trening od zera, jedna karta GPU, bez dużych modeli pretrenowanych.
- Mała skala: 10 klas, dziesiątki tysięcy obrazów (artykuł: do 21k klas, 300M obrazów).
- HET **nie przewidzi klasy, której nie ma w etykietach** (np. gatunku, gdy etykieta to tylko „ptak”).

## 3. Metoda — jak działa HET (skrót)

Głowica zastępuje ostatnią warstwę liniową. Z cech h (ResNet-18: D=512) trzy warstwy liniowe liczą:

| Wyjście | Kształt | Znaczenie |
|---|---|---|
| μ(x) | K | oceny klas (jak zwykła sieć) |
| V(x) | K × R | kierunki wspólnego chwiania się ocen = pary pomyłek |
| d(x) = softplus(·) + 1e-3 | K | niezależny rozrzut każdej klasy |

- Kowariancja: **Σ(x) = V(x)V(x)ᵀ + diag(d(x)²)** (analiza czynnikowa; stąd `MCSoftmaxDenseFA` w edward2).
- Reparametryzacja: **u = μ + V·ε_R + d⊙ε_K**, ε ~ N(0, I), S losowań.
- p(y|x) ≈ (1/S) Σₛ softmax(uₛ/τ). Strata: −log p(ỹ|x).
- Model generatywny: ỹ = argmax(μ + ε); zwykły softmax = szczególny przypadek (szum Gumbela i.i.d.).
- Mechanizm: błędna, ale wiarygodna etykieta (kot→ryś) daje słabszy gradient na μ (w przykładzie 1,23 → 0,49),
  a „energia” idzie w V(x). Nieprawdopodobna pomyłka (kot→ciężarówka) — pełna kara. Duże V obniża p(poprawna),
  więc model buduje niepewność tylko tam, gdzie etykiety w danych faktycznie się mieszają.
- Koszt: D·K·(R+2) parametrów; ResNet-18, K=10, R=5 → ~36 tys. (~0,3% sieci). Problem dopiero przy tysiącach klas (→ HET-XL).

**Interpretacja:** nie interpretować pojedynczych kolumn V (V i V·Q dla obrotu Q dają to samo VVᵀ).
Interpretować korelacje z Σ: corr_ij = Σ_ij / √(Σ_ii Σ_jj). Silnie ujemna korelacja = klasy się wymieniają.

## 4. Porównywane metody

| Metoda | Opis | Rola |
|---|---|---|
| CE | zwykła warstwa liniowa + entropia krzyżowa | baseline |
| LS | label smoothing α = 0,1 | baseline (regularyzacja) |
| HET-diag | Σ = diag(d²), R = 0 (odpowiada „Het. Diag” = Collier i in. 2020) | izoluje wkład korelacji |
| HET | Σ = VVᵀ + diag(d²) | badana metoda |

Opcjonalnie (jeśli czas): `parameter_efficient`, τ uczone (jak HET-XL — sprawdzone tylko w dużej skali, ryzyko przeuczenia).

## 5. Datasety i scenariusze

### 5.1 MNIST — eksperyment kontrolowany (sugestia promotora)
60k/10k, 28×28. Mała CNN (2 warstwy splotowe). Minuty na GPU. Znamy prawdziwe etykiety → znamy „odpowiedź”.

| Scenariusz | Ustawienie | Oczekiwanie |
|---|---|---|
| S0 | czysty | wszystkie ≈ 99%; test poprawności kodu |
| S1 | pary 1↔7, 4↔9, 3↔8, zależne od klasy, 10/30/45% | precision@3 = 3/3; korelacja rośnie z poziomem |
| S2 | te same pary, **zależne od obrazu**: P(zamiany 1→7) ∝ p(7|x) z czystego modelu pomocniczego, przeskalowane do średnio 30% | tu HET powinien najwyraźniej wygrać z HET-diag |
| S3 | kontrola negatywna: 30% szumu symetrycznego | brak wyróżnionych par; HET ≈ HET-diag |

Ablacja na S1-30%: R ∈ {1, 2, 3, 5}, τ ∈ {0,5; 1; 2}.

### 5.2 ANIMAL-10N — eksperyment właściwy
Song i in., ICML 2019 (SELFIE). https://dm.kaist.ac.kr/datasets/animal-10n
- 10 klas w 5 parach: kot/ryś, jaguar/gepard, wilk/kojot, szympans/orangutan, chomik/świnka morska.
- 50k train / 5k test, RGB 64×64; realny szum ludzki ~8% (ręczna kontrola: 6,44%); test uważany za czysty.
- ResNet-18 dostosowany do 64×64 (pierwszy splot 3×3, bez max-poolingu), 100 epok, SGD + cosine, batch 128.

| Scenariusz | Ustawienie | Po co |
|---|---|---|
| A1 | oryginalne etykiety | wynik główny |
| A2 | + 20% sztucznych zamian wewnątrz par | wyższy szum + znana część błędów → AUROC |
| A3 | ablacja R ∈ {1, 3, 5, 9}, τ | czy R = liczba par to dobry wybór |

Punkt odniesienia z literatury (VGG-19, nie ResNet-18 — tylko orientacyjnie; arXiv:2206.13140):
CE 79,4%, Co-teaching 80,2%, SELFIE 81,8%, PLC 83,4%, Dropout+Co-teaching 84,5%.

### 5.3 Opcjonalne
- CIFAR-10N/100N (Wei i in. ICLR 2022): Aggregate 9,03%, Random1 17,23%, Worst 40,21%, CIFAR-100N fine 40,20%.
  Wcześniej planowany jako główny zbiór — **zdegradowany do opcji** po decyzji o MNIST + zwierzęta.
- Ptaki (np. CUB-200-2011): rozszerzenie drobnoziarniste, mały zbiór; tylko jeśli zostanie czas.

## 6. Metodyka i metryki

- Podział: 90% train / 10% val (oba zaszumione); czysty test uruchamiany raz, na koniec. Hiperparametry tylko na val.
- **3 seedy** na konfigurację; raport: średnia ± std. Różnica < ~2 std ≠ przewaga.
- Raportować top-1 z **najlepszej i ostatniej** epoki (różnica = zapamiętywanie szumu).
- S = 1000 losowań MC (trening i test).

| Grupa | Metryka | Lepiej |
|---|---|---|
| Klasyfikacja | top-1 (czysty test), top-1 w parach (MNIST), top-2, spadek best→last | ↑ / ↓ |
| | pomyłki wewnątrz par vs między parami (ANIMAL-10N, z macierzy pomyłek) | więcej wewnątrz |
| Niepewność | NLL; ECE (15 przedziałów) + wykres niezawodności | ↓ |
| Struktura szumu (HET) | precision@k par (k=3 MNIST, k=5 ANIMAL), Spearman: siła szumu vs korelacja | ↑ |
| Wykrywanie błędów | AUROC, score = −log p(ỹ|x) na zbiorze treningowym (tylko gdzie szum jest znany: MNIST, A2) | ↑ |
| Koszt | liczba parametrów, czas epoki | — |

- Korelacje par: uśredniać Σ→corr po obrazach, dla pary (i,j) najlepiej po obrazach z prawdziwą klasą i lub j.
- Dwa tryby predykcji HET do porównania: średnia MC vs softmax(μ(x)) („czysta” odpowiedź).
- Wykresy do pracy: heatmapa uśrednionej korelacji, wykres niezawodności, krzywe top-1 po epokach.

## 7. Struktura kodu (planowane)

```
het-thesis/
├── configs/            # YAML na każdy eksperyment (pełna konfiguracja zapisywana obok wyników)
├── src/
│   ├── data.py         # MNIST, ANIMAL-10N, (CIFAR-N); wstrzykiwanie szumu: pary / zależny od obrazu / symetryczny
│   ├── models.py       # SmallCNN (MNIST), ResNet-18 dla 64×64
│   ├── heads.py        # LinearHead, HETHead(R, tau, S), HETDiag = HETHead(R=0)
│   ├── train.py        # pętla treningowa, logowanie best/last epoch
│   └── metrics.py      # top-1/top-2, NLL, ECE, precision@k, Spearman, AUROC
├── tests/              # pytest dla warstwy HET
├── analysis/           # notebooki: tabele, heatmapy korelacji, wykresy niezawodności
└── results/            # CSV + config każdego przebiegu
```

Stos: **PyTorch + torchvision**, YAML (argparse lub Hydra), W&B lub TensorBoard + CSV, pytest.
Planowane komendy: `pytest tests/`, `python -m src.train --config configs/<exp>.yaml --seed <n>`.

### Szkic głowicy (punkt wyjścia)
```python
class HETHead(nn.Module):
    def __init__(self, D, K, R, tau=1.0, S=1000):
        super().__init__()
        self.mu = nn.Linear(D, K); self.d = nn.Linear(D, K)
        self.V = nn.Linear(D, K * R) if R > 0 else None
        self.K, self.R, self.tau, self.S = K, R, tau, S

    def forward(self, h):                                   # h: [B, D] -> log p(y|x): [B, K]
        B = h.shape[0]
        mu = self.mu(h)
        d = F.softplus(self.d(h)) + 1e-3
        u = mu[:, None] + d[:, None] * torch.randn(B, self.S, self.K, device=h.device)
        if self.R > 0:
            V = self.V(h).view(B, self.K, self.R)
            u = u + torch.einsum('bkr,bsr->bsk', V, torch.randn(B, self.S, self.R, device=h.device))
        logp = F.log_softmax(u / self.tau, dim=-1)
        return torch.logsumexp(logp, dim=1) - math.log(self.S)   # -> F.nll_loss(out, y)
```

### Testy (wymagane, materiał do rozdz. 5.4)
- kształty tensorów; R=0 ≡ HET-diag; d→0, V=0 → softmax(μ/τ);
- zgodność z referencją edward2 `MCSoftmaxDenseFA` przy tych samych wagach i próbkach szumu: różnica < 1e-5;
- MNIST S0 ≈ 99%; CIFAR/ANIMAL baseline CE w oczekiwanym zakresie — **nie iść dalej, dopóki baseline nie działa**.

## 8. Podjęte decyzje i powody

| Decyzja | Powód |
|---|---|
| Własna implementacja w PyTorch, nie TF/edward2 | zrozumienie kodu linijka po linijce (obrona); edward2 korzysta z `tf.keras`, ryzyko problemów z Keras 3; Colab/PLGrid bez kombinowania |
| edward2 jako referencja w teście | dowód wierności oryginałowi; dobry materiał do rozdz. 5.4 |
| Strata przez `logsumexp(log_softmax) − log S` | stabilniejsze niż `log(clip(mean(softmax), 1e-7))` z edward2; opisać jako świadomą poprawkę |
| Głowica w float32 nawet przy AMP | stabilność numeryczna |
| MNIST + ANIMAL-10N zamiast CIFAR-N jako głównego | sugestia promotora; MNIST daje znaną „odpowiedź”, ANIMAL-10N ma wbudowaną strukturę 5 par i realny szum |
| HET-diag jako obowiązkowy baseline | jedyny sposób na wyizolowanie wkładu korelacji (najsłabiej udokumentowana część metody) |
| Zaszumiona walidacja, czysty test raz | realistyczny scenariusz, brak wycieku informacji |
| best + last epoch | standard w literaturze o szumie; mierzy zapamiętywanie |
| Scenariusz S2 (szum zależny od obrazu) | szum S1 jest tylko zależny od klasy; S2 testuje to, w czym HET ma przewagę |
| Kontrola negatywna S3 | jeśli HET „widzi” pary przy szumie losowym — błąd w kodzie lub analizie |
| Interpretacja przez korelacje Σ, nie kolumny V | niejednoznaczność obrotu V |
| 100 epok + cosine zamiast 200 | ~2× krócej, niewielka strata; dłuższy trening i tak sprzyja zapamiętywaniu |

## 9. Budżet obliczeń

- MNIST: ~150 przebiegów po 1–2 min → jeden wieczór.
- ANIMAL-10N: 4 metody × 3 seedy (A1) + A2 + ablacja → ok. 10–20 h na RTX 3090; Colab T4 ~2–3× wolniej.
- Opcja: PLGrid / Cyfronet (A100/H100) — grant zwykle zakłada promotor.
- Przyspieszenia: kilka przebiegów równolegle na jednej karcie; dane i augmentacja na GPU.
- Zmierzyć czas epoki przy pierwszym teście i przemnożyć.

## 10. Literatura (zweryfikowana) i obowiązkowe korekty

Dokumenty: plan pracy https://claude.ai/code/artifact/53afe66d-28e4-40a9-9ac8-557fd739a8fd ·
streszczenia 22 pozycji https://claude.ai/code/artifact/2a3948d7-676d-4c16-b7c7-89b6e03d2cd8 ·
raport weryfikacji: `reports/Weryfikacja literatury do inżynierki.md` (sesja Claude).

Rdzeń (dane sprawdzone w źródłach pierwotnych):
1. Collier M., Mustafa B., Kokiopoulou E., Jenatton R., Berent J., CVPR 2021, s. 1551–1560, DOI 10.1109/CVPR46437.2021.00160.
2. Collier i in., *A Simple Probabilistic Method for Deep Classification under Input-Dependent Label Noise*, preprint arXiv:2003.06778, 2020 (brak recenzowanej publikacji).
3. Kendall A., Gal Y., NIPS 30, 2017, s. 5574–5584 (bez DOI).
4. Song H., Kim M., Park D., Shin Y., Lee J.-G., IEEE TNNLS t. 34, nr 11, 2023, s. 8135–8153, DOI 10.1109/TNNLS.2022.3152527.
5. Collier i in., *Massively Scaling Heteroscedastic Classifiers*, ICLR 2023 (HET-XL).
6. Song, Kim, Lee, *SELFIE*, ICML 2019 (ANIMAL-10N); LeCun i in. 1998 (MNIST); Guo i in. ICML 2017 (ECE);
   Frénay, Verleysen, TNNLS t. 25, nr 5, 2014, s. 845–869 (taksonomia NCAR/NAR/NNAR); Goodfellow i in. 2016.

Korekty, których trzymać się w tekście pracy:
- Kendall i Gal **nie wprowadzili** rozróżnienia aleatoryczna/epistemiczna (cytują Der Kiureghian i Ditlevsen 2009) — połączyli je w jednym frameworku dla wizji.
- „Het. Diag” w artykule cytuje Collier 2020, nie Kendall i Gal; „R = 0” to oznaczenie z tej pracy, nie z artykułu.
- +2,6 p.p. tylko z kontekstem: ResNet-152, 270 epok (79,3 vs 76,7); nad HET-diag tylko +0,6 p.p.
- HET modeluje korelacje w przybliżeniu **niskiego rzędu**, nie „pełne”.
- W artykule metoda nazywa się „Het. Full”/„Het. PE”; skrót HET pochodzi z pracy o HET-XL.
- HET-XL: „14×” tylko dla JFT-4B (JFT-300M ~9×); uczone τ sprawdzone tylko w dużej skali.
- NNAR zależy od cech **i** prawdziwej klasy. CIFAR-100N ~40% dotyczy etykiet fine.
- Zhu i in. NeurIPS 2024 (DINOv2 + regresja logistyczna): najlepsi na 5 z 6 zestawów CIFAR-N, nie na wszystkich.
- Teza „DivideMix osiąga najlepsze wyniki na CIFAR z szumem syntetycznym” — **niepotwierdzona**, usunąć lub podać źródło.
- „Oral na CVPR 2021” — tylko komentarz autorów na arXiv; nie eksponować.
- Słowo „CIFAR” nie występuje w artykule bazowym → luka, którą wypełnia praca (mała skala, znana struktura szumu).

## 11. Spis treści (aktualny)

1. Wstęp (motywacja, cel i zakres, układ) · 2. Podstawy teoretyczne (CNN/ResNet, softmax i CE, niepewność
aleatoryczna/epistemiczna, typy szumu, zapamiętywanie, rozkład normalny i Monte Carlo) · 3. Metody uczenia
z błędnymi etykietami wg 5 grup Songa (architektury, regularyzacja, odporne straty, korekcja straty, selekcja
próbek) + modele heteroskedastyczne · 4. Metoda HET (model, kowariancja niskiego rzędu, MC i reparametryzacja,
temperatura, koszt, interpretacja Σ) · 5. Implementacja (środowisko, warstwy, generowanie szumu, weryfikacja)
· 6. Eksperymenty (metodyka, MNIST, ANIMAL-10N, hiperparametry, analiza pomyłek) · 7. Dyskusja (porównanie
z artykułem, ograniczenia) · 8. Podsumowanie i kierunki rozwoju. Objętość ~45–60 stron.

## 12. Aktualny stan

- [x] Temat, cel, założenia, plan i spis treści
- [x] Literatura: 22 pozycje ze streszczeniami; dane bibliograficzne rdzenia zweryfikowane
- [x] Wybór datasetów (MNIST + ANIMAL-10N), metod, metryk, scenariuszy
- [ ] Dokument z planem nieaktualny w dwóch miejscach: sekcja 1 „Zakres” i sekcja 4 „Stan wiedzy” nadal mówią o CIFAR; korekty literatury z §10 nienaniesione
- [ ] Pozycje „do sprawdzenia” w dokumencie z literaturą (He, Kingma, Patrini, GCE, SCE, Co-teaching, DivideMix, Müller, LeCun, Guo, Arpit, Zhang 2017) — potwierdzić strony w DBLP/Scholar
- [x] Kod MNIST: heads/models/train/metrics/plots + testy; S0 (seed 0) ≈ 99,2% dla CE/HET-diag/HET
- [x] Szum (2026-11-02): `src/noise.py` (S1 pary, S2 zależny od obrazu, S3 symetryczny), cache w `data/noise/`,
      configi z `configs/make_mnist_configs.py`. S2: model pomocniczy z cross-fittingiem (2 foldy, 3 epoki),
      P(zamiany) ∝ softmax(logit/T=4)[partner], limit 0,45. Seed szumu przesunięty o SEED_OFFSET
      (bez tego S3 zamieniał całą walidację — ten sam randperm co podział).
- [x] Metryki szumu w summary.json: AUROC (train), precision@k par, korelacje par uśrednione po klasach i/j
- [x] Pełna siatka MNIST (2026-11-02): S0–S3 × CE/LS/HET-diag/HET × 3 seedy + ablacja R×τ na S1-30% (105 przebiegów,
      ~1,5 h na GTX 1060). `python -m src.run_grid --pattern "mnist_s*"`, potem `python -m analysis.aggregate`
      → results/tables/*.md, results/figures/*.png. Główne: HET wygrywa wyraźnie tylko przy 45% i w ostatniej epoce;
      korelacje znajdują pary (precision@3 = 1 przy ≥30%, 0 w S0/S3); softmax(μ) ma dużo niższe ECE niż średnia MC.

Do potwierdzenia z promotorem:
- termin oddania (roboczo: koniec stycznia 2027);
- „false ptak”: błędny gatunek (HET pomoże) czy tylko etykieta „ptak” bez gatunku (HET nie pomoże — inny problem)?
- zgodność tematu z kierunkiem telekomunikacja; ewentualny pomost: szum etykiet w klasyfikacji ruchu sieciowego (np. CIC-IDS2017) jako dodatkowy rozdział;
- dostęp do PLGrid.

## 13. Następne kroki

1. Repo `het-thesis/` wg §7; środowisko PyTorch; pobrać MNIST i ANIMAL-10N.
2. Baseline CE: MNIST czysty ≈ 99%; ANIMAL-10N ≈ 79–80% — dopiero potem dalej.
3. `heads.py` (HETHead, HETDiag) + testy z §7, w tym porównanie z edward2.
4. `data.py`: szum w parach (S1), zależny od obrazu (S2, wymaga czystego modelu pomocniczego), symetryczny (S3).
5. `metrics.py`: wszystkie metryki z §6, przetestowane na małych przykładach ręcznych (np. AUROC 8/9 = 0,89).
6. Siatka MNIST (S0–S3 + ablacja) → pierwsze tabele i heatmapy; pokazać promotorowi.
7. ANIMAL-10N A1 → A2 → A3; zmierzyć czas epoki i zaplanować resztę budżetu.
8. Równolegle: poprawić dokument z planem (sekcje 1 i 4, korekty z §10), pisać rozdz. 2–3.

Harmonogram roboczy: implementacja do ~15.11 · MNIST + start ANIMAL-10N do ~6.12 · ablacje i analiza do ~20.12
· rozdz. 6–7 do ~10.01 · wstęp, podsumowanie, poprawki, oddanie ~24.01.2027.

## 14. Wskazówki dla Claude Code

- Pisz do Bartka po polsku, zwięźle, z konkretnymi liczbami.
- Nie zmieniaj ustaleń z §8 bez pytania; jeśli coś przeczy planowi, zaproponuj i uzasadnij.
- Każdy eksperyment: config w YAML, seed, zapis wyników do CSV — wyniki muszą być odtwarzalne.
- Nie raportuj wyników z testu przy doborze hiperparametrów.
- Przy tekstach do pracy trzymaj się korekt z §10; nie dopisuj liczb z literatury bez źródła.
