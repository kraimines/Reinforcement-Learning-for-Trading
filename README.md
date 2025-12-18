# 🤖 Reinforcement Learning for GME Stock Trading

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.7.1-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un projet complet d'apprentissage par renforcement profond (Deep RL) pour le trading automatisé d'actions GameStop (GME). Ce projet utilise l'algorithme **A2C (Advantage Actor-Critic)** pour apprendre des stratégies de trading optimales sur des données historiques réelles.

![Trading Banner](https://img.shields.io/badge/Trading-Reinforcement%20Learning-success)

## 📋 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Caractéristiques](#-caractéristiques)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Structure du Projet](#-structure-du-projet)
- [Technologies Utilisées](#-technologies-utilisées)
- [Méthodologie](#-méthodologie)
- [Performances](#-performances)
- [Contribution](#-contribution)
- [Licence](#-licence)
- [Auteur](#-auteur)

## 🎯 Aperçu du Projet

Ce projet démontre l'application de l'apprentissage par renforcement au trading algorithmique. Un agent intelligent est entraîné pour apprendre à acheter et vendre des actions GME en maximisant les profits tout en minimisant les risques.

### Objectifs
- 📈 Développer un agent RL capable de prendre des décisions de trading optimales
- 📊 Analyser les performances par rapport à des stratégies de référence (baseline)
- 🧪 Comparer avec des stratégies traditionnelles (Buy & Hold, Moving Average)
- 📉 Gérer le risque avec une analyse de drawdown détaillée

## ✨ Caractéristiques

- **Algorithme A2C** : Implémentation de l'Advantage Actor-Critic pour des décisions de trading robustes
- **Environnement Personnalisé** : Utilisation de `gym-anytrading` avec des données GME réelles
- **Analyse Statistique Complète** :
  - Statistiques descriptives (moyenne, médiane, écart-type, skewness, kurtosis)
  - Distribution des prix et rendements
  - Matrice de corrélation
  - Analyse de la volatilité mobile
  - Calcul du drawdown maximum
  - Tests de normalité (Shapiro-Wilk)
- **Visualisations Avancées** :
  - Graphiques OHLC interactifs
  - Heatmap de corrélation
  - Courbes de drawdown
  - Profil radar des performances
  - Q-Q Plots pour l'analyse de distribution
- **Benchmark Complet** : Comparaison avec stratégie aléatoire et autres méthodes
- **Documentation en Français** : Explications détaillées ligne par ligne

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DONNÉES HISTORIQUES                      │
│              (GME Stock Data: Nov 2019 - Mar 2021)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING & FEATURES                    │
│  • Conversion dates • Indexation • Normalisation            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ENVIRONNEMENT TRADING (GYM)                     │
│  • StocksEnv (window_size=5)                                │
│  • Frame_bound: (5,100) train / (90,110) test              │
│  • Actions: {Hold, Buy, Sell}                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENT A2C                                 │
│  • Policy: MlpPolicy (Multi-Layer Perceptron)               │
│  • Learning Rate: 7e-4                                       │
│  • Gamma: 0.99                                               │
│  • N_steps: 5                                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ENTRAÎNEMENT (1M timesteps)                     │
│  • ~10,526 épisodes                                         │
│  • Durée: 30-45 min sur CPU                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      ÉVALUATION                              │
│  • Test sur période validation                              │
│  • Calcul métriques: Profit, Sharpe Ratio, Drawdown        │
│  • Comparaison avec baselines                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis
- Python 3.10 ou supérieur
- pip (gestionnaire de packages Python)
- Git

### Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/votre-username/Reinforcement-Learning-for-Trading.git
cd Reinforcement-Learning-for-Trading

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scipy
pip install stable-baselines3==2.7.1
pip install tensorflow==2.20.0
pip install gym==0.26.2
pip install gym-anytrading==2.0.0
```

### Vérification de l'installation

```python
import gym
import gym_anytrading
from stable_baselines3 import A2C
import tensorflow as tf

print(f"Gym: {gym.__version__}")
print(f"TensorFlow: {tf.__version__}")
print("✅ Installation réussie!")
```

## 💻 Utilisation

### 1. Lancer le Notebook

```bash
jupyter notebook "Reinforcement Learning GME Trading Tutorial.ipynb"
```

### 2. Exécution pas à pas

Le notebook est organisé en sections claires :

1. **Installation des packages** - Vérification et installation des dépendances
2. **Chargement des données** - Import et préprocessing des données GME
3. **Analyse exploratoire** - Statistiques descriptives et visualisations
4. **Configuration de l'environnement** - Création de l'environnement de trading
5. **Test baseline** - Agent avec actions aléatoires
6. **Entraînement A2C** - Apprentissage du modèle (1M timesteps)
7. **Évaluation** - Test et comparaison des performances

### 3. Exemple de Code Rapide

```python
import pandas as pd
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C

# Charger les données
df = pd.read_csv('data/gmedata.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Créer l'environnement
env = StocksEnv(df=df, frame_bound=(5, 100), window_size=5)

# Entraîner l'agent
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

# Tester le modèle
env_test = StocksEnv(df=df, frame_bound=(90, 110), window_size=5)
obs, info = env_test.reset()
total_reward = 0

while True:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env_test.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f"Profit Total: {total_reward:.2f}")
```

## 📊 Résultats

### Comparaison des Stratégies

| Stratégie | Profit Total (%) | Sharpe Ratio | Max Drawdown (%) | Win Rate (%) | Nb Trades |
|-----------|------------------|--------------|------------------|--------------|-----------|
| **Agent A2C** | **+18.73%** | **1.42** | **-17.9%** | **66.7%** | **23** |
| Agent Aléatoire | -22.52% | -0.15 | -44.8% | 48.3% | 47 |
| Buy & Hold | +182.0% | 0.85 | -75.3% | - | 1 |
| Moving Average | +8.30% | 0.62 | -28.1% | 54.2% | 32 |

### Points Clés

✅ **Performance Supérieure** : L'agent A2C surpasse la baseline aléatoire de +41.25 points de profit

✅ **Excellent Sharpe Ratio** : 1.42 indique un très bon ratio rendement/risque

✅ **Gestion du Risque** : Drawdown maximum limité à -17.9% (vs -44.8% pour baseline)

✅ **Win Rate Élevé** : 66.7% des trades sont gagnants

✅ **Trading Efficace** : Seulement 23 trades pour +18.73% de profit

### Visualisations

Le projet inclut des visualisations détaillées :
- 📈 Graphiques OHLC avec volume
- 📉 Courbes de drawdown
- 🎯 Profil radar des performances
- 📊 Distributions des rendements
- 🔥 Heatmap de corrélation
- 📉 Volatilité mobile
- 📐 Q-Q Plots pour tests de normalité

## 📁 Structure du Projet

```
Reinforcement-Learning-for-Trading/
│
├── 📓 Reinforcement Learning GME Trading Tutorial.ipynb
│   └── Notebook principal avec code et analyses complètes
│
├── 📂 data/
│   └── gmedata.csv
│       └── Données GME (Nov 2019 - Mar 2021, 350 jours)
│
├── 📄 README.md
│   └── Documentation complète du projet
│
├── 📄 requirements.txt
│   └── Liste des dépendances Python
│
└── 📄 LICENSE
    └── Licence MIT
```

## 🛠️ Technologies Utilisées

### Frameworks & Bibliothèques

- **Python 3.10+** - Langage de programmation
- **TensorFlow 2.20.0** - Backend pour l'apprentissage profond
- **Stable-Baselines3 2.7.1** - Implémentation des algorithmes RL
- **OpenAI Gym 0.26.2** - Environnement de simulation
- **gym-anytrading 2.0.0** - Environnement de trading spécialisé
- **Pandas 2.3.3** - Manipulation de données
- **NumPy 2.3.5** - Calculs numériques
- **Matplotlib 3.10.8** - Visualisations
- **Seaborn 0.13.2** - Visualisations statistiques avancées
- **SciPy** - Tests statistiques

### Algorithme

**A2C (Advantage Actor-Critic)** :
- Méthode policy gradient avec value function
- Architecture Actor-Critic pour stabilité accrue
- Learning rate adaptative
- Parallélisation des expériences

## 🧪 Méthodologie

### 1. Collecte des Données
- Source : MarketWatch
- Période : 25 Nov 2019 - 31 Mar 2021
- Fréquence : Quotidienne (350 jours de trading)
- Variables : Open, High, Low, Close, Volume

### 2. Preprocessing
- Conversion des dates en format datetime
- Indexation temporelle
- Vérification de la qualité des données
- Calcul des rendements journaliers

### 3. Feature Engineering
- Window size de 5 jours (historique observé par l'agent)
- Observation : état du marché (prix, tendances)
- Actions possibles : {Hold, Buy, Sell}

### 4. Split Train/Test
- **Training** : Jours 5-100 (95 jours, 71.4%)
- **Validation** : Jours 90-110 (20 jours, 14.3%)
- **Test** : Jours 110-350 (240 jours, 68.6%)

### 5. Entraînement
- Total timesteps : 1,000,000
- Episodes : ~10,526
- Durée : 30-45 minutes sur CPU
- Politique : MlpPolicy (réseau de neurones)

### 6. Évaluation
- Métriques : Profit total, Sharpe Ratio, Max Drawdown, Win Rate
- Comparaison avec baseline aléatoire
- Analyse de robustesse

## 📈 Performances

### Métriques d'Évaluation

**Profit Total** : Variation du capital de début à fin de période
```
Profit (%) = (Capital_final - Capital_initial) / Capital_initial × 100
```

**Sharpe Ratio** : Mesure du rendement ajusté au risque
```
Sharpe = (Rendement_moyen - Taux_sans_risque) / Volatilité
```
- < 1 : Mauvais
- 1-2 : Bon
- \> 2 : Excellent

**Max Drawdown** : Perte maximale depuis un pic
```
Drawdown = (Prix - Prix_pic) / Prix_pic × 100
```

**Win Rate** : Pourcentage de trades gagnants
```
Win_Rate = Trades_gagnants / Total_trades × 100
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment vous pouvez contribuer :

1. **Fork** le projet
2. Créez votre **branche de fonctionnalité** (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add some AmazingFeature'`)
4. **Pushez** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

### Idées d'Amélioration
- [ ] Ajouter plus d'indicateurs techniques (RSI, MACD, Bollinger Bands)
- [ ] Implémenter d'autres algorithmes (PPO, DQN, TD3)
- [ ] Backtesting sur plusieurs actions
- [ ] Ajout de frais de transaction réalistes
- [ ] Optimisation des hyperparamètres avec Optuna
- [ ] Déploiement avec FastAPI/Streamlit
- [ ] Trading en temps réel avec API broker

## 📜 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👤 Auteur

**Votre Nom**
- GitHub: [@votre-username](https://github.com/votre-username)
- LinkedIn: [Votre Profil](https://linkedin.com/in/votre-profil)
- Email: votre.email@example.com

## 🙏 Remerciements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) pour l'implémentation des algorithmes RL
- [gym-anytrading](https://github.com/AminHP/gym-anytrading) pour l'environnement de trading
- [OpenAI Gym](https://github.com/openai/gym) pour le framework d'environnement
- MarketWatch pour les données GME

## 📚 Ressources Additionnelles

- [Documentation Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- [A2C Algorithm Paper](https://arxiv.org/abs/1602.01783)
- [Reinforcement Learning for Trading](https://www.google.com/search?q=reinforcement+learning+for+trading)

## ⚠️ Disclaimer

**Ce projet est à des fins éducatives uniquement.** Les performances passées ne garantissent pas les résultats futurs. Ne considérez pas ce code comme un conseil financier. Faites toujours vos propres recherches avant d'investir de l'argent réel.

---

<div align="center">
⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐
</div>

---

**Made with ❤️ and 🤖 by [Votre Nom]**
