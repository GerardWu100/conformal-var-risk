---
title: "VaR conforme adaptative : la calibration a un prix"
description: "Un audit walk-forward de la Value at Risk conforme adaptative face à la simulation historique, aux modèles GARCH et à la simulation historique filtrée."
date: 2026-07-12
image: images/cover-conformal-var.png
categories: ["Quantitative Finance", "Risk Management"]
---

Une prévision de Value at Risk à un jour trace une borne sous le rendement de demain. Si le modèle vise une queue inférieure de 5 %, environ cinq rendements sur cent devraient franchir cette borne pendant une période d'évaluation longue et stable. Trop de franchissements indiquent une sous-estimation du risque. Trop peu peuvent sembler rassurants, mais une borne trop basse renchérit inutilement le capital et les limites de trading.

J'ai comparé une prévision conforme adaptative à quatre modèles classiques de risque de marché. Tous reçoivent les mêmes rendements passés et prévoient les mêmes dates. La méthode conforme a produit moins de violations dans cet échantillon, mais aussi la pire perte quantile. Moins de violations ne signifie donc pas une meilleure prévision.

## Définir l'objet prévu

Soit $P_t$ le cours de clôture d'un actif au jour de bourse $t$. Son rendement logarithmique à un jour, exprimé en rendement décimal, vaut

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

Soit $q_{t,\alpha}$ la prévision du quantile d'ordre $\alpha$ de $r_t$, où $\alpha=0.05$ désigne la queue inférieure à 5 %. Le modèle vise

$$
\Pr(r_t < q_{t,\alpha}) \approx \alpha.
$$

La Value at Risk (VaR) transforme cette borne de rendement en perte non négative :

$$
\operatorname{VaR}_{t,\alpha}=\max(-q_{t,\alpha},0).
$$

Par exemple, $q_{t,0.05}=-0.02$ implique une VaR à un jour de 2 %. Une violation, aussi appelée dépassement, se produit lorsque le rendement réalisé vérifie $r_t<q_{t,\alpha}$. Le backtest évalue le quantile brut, même dans le cas inhabituel où il est positif ; la troncature ne s'applique qu'à l'affichage de la VaR comme perte.

## Des barres minute à une horloge de prévision unique

Le jeu de données suivi contient les barres minute de séance régulière d'AAPL, JPM, TSLA et SPY de 2019 à 2023. Le pipeline prend la dernière clôture de chaque séance, calcule les rendements logarithmiques quotidiens et forme une série de portefeuille équipondéré.

Il calcule aussi la variance réalisée à partir de rendements intrajournaliers synchronisés. Soit $r_{t,i,j}$ le rendement logarithmique intrajournalier du constituant $j$ pendant l'intervalle $i$ du jour $t$, et soit $N=4$ le nombre de constituants. Le rendement intrajournalier du portefeuille équipondéré est

$$
r^{p}_{t,i}=\frac{1}{N}\sum_{j=1}^{N}r_{t,i,j}.
$$

La variance réalisée quotidienne du portefeuille, exprimée en rendement décimal au carré par jour, devient

$$
\operatorname{RV}^{p}_t=\sum_i\left(r^{p}_{t,i}\right)^2
=\frac{1}{N^2}\sum_i\sum_{j=1}^{N}\sum_{k=1}^{N}r_{t,i,j}r_{t,i,k}.
$$

L'indice $k$ désigne un second constituant. Les produits croisés où $j\ne k$ portent donc la covariance intrajournalière. Une version précédente moyennait les variances réalisées des constituants. Elle omettait ces termes et utilisait une mauvaise pondération. Le pipeline corrigé calcule d'abord le rendement du portefeuille, puis son carré. Ces variables de variance réalisée sont retardées et disponibles pour de futures extensions ; les cinq modèles testés ici ajustent directement les rendements quotidiens passés.

L'ordre walk-forward est strict. À chaque date de prévision, le modèle s'ajuste sur les 1 150 rendements précédents, prévoit le rendement suivant, enregistre le résultat, puis met à jour son état adaptatif.

```python
training_returns = asset_series.iloc[
    evaluation_index - calibration_window : evaluation_index
].to_numpy(dtype=float)
realized_return = float(asset_series.iloc[evaluation_index])

model.fit(training_returns)
lower_quantile = model.predict_lower_quantile(alpha=alpha)
is_violation = realized_return < lower_quantile
model.observe(realized_return=realized_return, alpha=alpha)
```

Aucune information observée au jour $t$ n'entre dans la borne qui sert à juger ce même jour.

## Les hypothèses des cinq modèles

La simulation historique prend le quantile empirique d'ordre $\alpha$ des 250 derniers rendements. Elle suppose que leur distribution reste pertinente pour demain.

Le modèle Generalized Autoregressive Conditional Heteroskedasticity, abrégé GARCH, fait varier la volatilité dans le temps. Pour un GARCH(1,1), soit $\mu_t$ la moyenne conditionnelle, $\varepsilon_t=r_t-\mu_t$ le choc de rendement et $\sigma_t^2$ la variance conditionnelle. La récurrence est

$$
\sigma_{t+1}^2=\omega+\beta\sigma_t^2+\delta\varepsilon_t^2,
$$

où $\omega>0$, $\beta\ge 0$ et $\delta\ge 0$ sont des paramètres estimés. Si $F^{-1}(\alpha)$ est le quantile de queue inférieure de la loi des innovations standardisées, alors

$$
q_{t+1,\alpha}=\mu_{t+1}+\sigma_{t+1}F^{-1}(\alpha).
$$

La comparaison retient des innovations gaussiennes et de Student-$t$. La loi de Student-$t$ autorise des queues plus épaisses.

La simulation historique filtrée ajuste aussi un GARCH, mais rééchantillonne les chocs standardisés empiriques au lieu d'imposer une queue gaussienne ou de Student-$t$. Elle associe une prévision paramétrique de volatilité à une distribution non paramétrique des chocs.

## Construire la borne conforme adaptative

Le modèle conforme part d'une moyenne mobile. Soit $m=20$ la fenêtre de moyenne et soit $\hat{\mu}_u$ le centre prévu pour la date de calibration $u$ :

$$
\hat{\mu}_u=\frac{1}{m}\sum_{j=1}^{m}r_{u-j}.
$$

Son score de non-conformité unilatéral ne conserve que les erreurs défavorables :

$$
s_u=\max(\hat{\mu}_u-r_u,0).
$$

La fenêtre conforme contient 500 rendements. Après les 20 observations nécessaires à la moyenne mobile, il reste $n=500-20=480$ scores. On les trie selon $s_{(1)}\le\cdots\le s_{(n)}$. Si $a_t$ est la probabilité de queue interne courante, le rang corrigé pour l'échantillon fini vaut

$$
k_t=\min\left(n,\left\lceil(n+1)(1-a_t)\right\rceil\right).
$$

La prochaine borne inférieure de rendement est

$$
q_{t,\alpha}=\hat{\mu}_t-s_{(k_t)}.
$$

La correction en $n+1$ sélectionne une statistique d'ordre observée au lieu d'un percentile interpolé. C'est la correction usuelle du conformal split. Elle ne crée pas de garantie inconditionnelle à échantillon fini pour cette expérience : les scores mobiles se chevauchent et les rendements financiers dépendants dans le temps ne respectent pas l'hypothèse d'échangeabilité de la théorie conforme classique.

```python
sample_size = len(scores)
rank = int(np.ceil((sample_size + 1) * (1.0 - alpha)))
clipped_rank = int(np.clip(rank, 1, sample_size))
adjustment = np.partition(scores, clipped_rank - 1)[clipped_rank - 1]
lower_quantile = center - adjustment
```

L'adaptation modifie $a_t$ après chaque résultat. On définit $I_t=1$ en cas de violation et $I_t=0$ sinon. Avec la probabilité de queue cible $\alpha$ et le taux d'apprentissage $\gamma=0.005$, la mise à jour est

$$
a_{t+1}=\operatorname{clip}\left(a_t+\gamma(\alpha-I_t),0.001,0.999\right).
$$

Après une violation, $\alpha-I_t<0$ et $a_{t+1}$ diminue. Le rang $k_{t+1}$ augmente, le score retenu devient plus grand et la borne suivante descend. Les journées calmes inversent ce mouvement par pas de $\gamma\alpha$.

Il s'agit d'une règle de risque inspirée du conformal adaptatif, pas d'une affirmation selon laquelle les rendements boursiers seraient « sans distribution » au sens courant. Le résultat formel de Gibbs et Candès contrôle la fréquence de mauvaise couverture à long terme sous les conditions précisées dans leur article ; ce backtest de sept mois doit toujours justifier ses conclusions par les données.

## Évaluer la calibration et l'utilité

Le taux de violation empirique sur $T$ prévisions est

$$
\hat{p}=\frac{1}{T}\sum_{t=1}^{T}I_t.
$$

Soit $K=\sum_{t=1}^{T}I_t$ le nombre de violations. Sous une probabilité de violation constante $p$, la vraisemblance de Bernoulli vaut

$$
\mathcal{L}(p)=p^K(1-p)^{T-K}.
$$

La calibration demande si $\hat{p}$ est compatible avec $\alpha$. On note $\operatorname{LR}_{\mathrm{UC}}$ la statistique de rapport de vraisemblance de Christoffersen pour la couverture inconditionnelle :

$$
\operatorname{LR}_{\mathrm{UC}}=-2\log\left(\frac{\mathcal{L}(\alpha)}{\mathcal{L}(\hat{p})}\right),
$$

dont la loi asymptotique sous l'hypothèse nulle est un chi carré à un degré de liberté. La couverture conditionnelle ajoute un test de transition du premier ordre afin de repérer les violations groupées. Une p-value supérieure à 5 % indique seulement que l'échantillon n'a pas rejeté le modèle ; elle ne prouve pas sa bonne calibration.

La perte quantile, ou perte pinball, mesure aussi l'utilité de la prévision. On note $q_t=q_{t,\alpha}$ le quantile inférieur prévu :

$$
L_{\alpha}(r_t,q_t)=(\alpha-I_t)(r_t-q_t).
$$

Les deux cas expliquent le compromis. Lorsque $r_t\ge q_t$, $I_t=0$ et le coût vaut $\alpha(r_t-q_t)$. Lorsque $r_t<q_t$, il vaut $(1-\alpha)(q_t-r_t)$. Une borne très basse évite les violations, mais paie un petit coût pendant presque toutes les journées ordinaires.

## Prudente, mais moins précise

La fenêtre de calibration de 1 150 jours laisse 153 prévisions par série, du 31 mai au 29 décembre 2023. Les comptes regroupés ci-dessous combinent quatre actifs et le portefeuille pour former un total descriptif de 765 prévisions. Il ne s'agit pas de 765 essais indépendants, puisque les actifs et le portefeuille partagent les mêmes chocs de marché.

![Taux de violation observés par modèle et probabilité de queue](images/01_violation_rates.png)

Les lignes pointillées indiquent les probabilités de queue nominales. Les barres situées en dessous signalent des prévisions prudentes dans cet échantillon, pas nécessairement de meilleures prévisions.

| Modèle | Violations à 5 % | Taux à 5 % | Violations à 1 % | Taux à 1 % |
|---|---:|---:|---:|---:|
| Simulation historique | 42 | 5.49% | 5 | 0.65% |
| GARCH gaussien | 24 | 3.14% | 6 | 0.78% |
| GARCH de Student | 24 | 3.14% | 6 | 0.78% |
| Simulation historique filtrée | 27 | 3.53% | 4 | 0.52% |
| Conforme adaptative | 22 | 2.88% | 0 | 0.00% |

La simulation historique s'approche le plus de la cible groupée à 5 %. La méthode conforme adaptative produit le moins de violations et la VaR moyenne la plus large : 2.48 % pour la queue à 5 % et 3.74 % pour celle à 1 %. La réduction des dépassements consomme donc davantage de largeur de capital.

L'échantillon a peu de puissance à 1 %. Pour une série, $T=153$ et le nombre attendu de violations sous une bonne calibration vaut $T\alpha=1.53$. La probabilité de n'en observer aucune est

$$
\Pr(K=0\mid T=153,\alpha=0.01)=(1-\alpha)^T=0.99^{153}=21.49\%.
$$

Zéro violation n'a rien d'étonnant sous l'hypothèse nulle. Sous l'hypothèse d'indépendance binomiale, l'intervalle exact de Clopper-Pearson à 95 % pour zéro violation sur 153 va de 0 % à 2.38 %, ce qui contient la cible de 1 %. Toutes les p-values de Christoffersen calculées actif par actif dépassent aussi 5 %, mais elles souffrent du même manque d'observations.

![Perte pinball moyenne par modèle et probabilité de queue](images/02_quantile_loss.png)

À 5 %, la simulation historique filtrée obtient la plus faible perte pinball moyenne, soit 13.21 points de base de rendement. La méthode conforme adaptative ferme la marche à 14.09 points de base. À 1 %, la simulation historique mène avec 3.24 points de base, contre 3.84 pour la méthode conforme. Le graphique des violations favorise à lui seul la prudence ; la perte pinball révèle son coût.

![Rendements de SPY et borne conforme adaptative à 5 %](images/03_spy_forecast_path.png)

SPY franchit la borne conforme à 5 % cinq fois sur 153 prévisions, soit un taux de 3.27 %. L'intervalle exact à 95 % pour ce taux va de 1.07 % à 7.46 %. La borne turquoise évolue lentement parce qu'elle associe un centre sur 20 jours, 480 scores de calibration et la mise à jour adaptative. Les points rouges marquent les dates qui abaissent la probabilité de queue interne.

## Ce que l'expérience permet d'affirmer

Les périodes configurées pour la crise du COVID et le choc de taux de 2022 ne produisent aucune observation hors échantillon : elles se terminent avant la première prévision de mai 2023. Les qualifier de stress tests serait incorrect. Une comparaison crédible en période de stress exige davantage d'historique avant 2019 ou une fenêtre de calibration plus courte, choisie sans consulter les résultats de test.

Les cinq séries de prévisions dépendent les unes des autres. Les violations regroupées servent donc à visualiser les résultats, pas à multiplier artificiellement la taille de l'échantillon. L'inférence de couverture doit rester au niveau de chaque série ou intégrer explicitement cette dépendance.

Le taux d'apprentissage demande aussi une analyse de sensibilité. Pour la cible à 1 %, une journée calme augmente $a_t$ de seulement $0.00005$, alors qu'une violation le réduit de $0.00495$. Cette asymétrie est voulue, mais sept mois ne suffisent pas à décrire son comportement dans plusieurs régimes de volatilité.

L'Expected Shortfall (ES) est la perte moyenne conditionnelle dans la queue. Le projet la rapporte comme diagnostic empirique secondaire, tandis que le score conforme cible un quantile, pas l'ES. Cette construction ne fournit aucune garantie conforme sur l'ES.

Je ne considère pas ce résultat comme une victoire de la nouvelle méthode. La VaR conforme adaptative réduit les violations sur ces 153 dates, puis perd sur la perte quantile. Avec un échantillon aussi court, la conclusion doit rester modeste : la mise à jour a déplacé le compromis entre calibration et précision. Une évaluation plus longue doit montrer si la largeur supplémentaire se justifie.

## Références

- Gibbs, I. and Candès, E. (2021), [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
- Christoffersen, P. (1998), [Evaluating Interval Forecasts](https://www.jstor.org/stable/2527341).
- Bollerslev, T. (1986), [Generalized Autoregressive Conditional Heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1).
- Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999), [VaR without Correlations for Portfolios of Derivative Securities](https://doi.org/10.1002/(SICI)1096-9934(199908)19:5%3C583::AID-FUT5%3E3.0.CO;2-S).
- Koenker, R. and Bassett, G. (1978), [Regression Quantiles](https://www.jstor.org/stable/1913643).
- Andersen, T., Bollerslev, T., Diebold, F. and Labys, P. (2003), [Modeling and Forecasting Realized Volatility](https://doi.org/10.1111/1468-0262.00418).
- Basel Committee on Banking Supervision (2019), [Minimum capital requirements for market risk](https://www.bis.org/bcbs/publ/d457.htm).
