# Étude d'Évolution de Voxtral Pod

Après analyse du codebase (v4.0.0), le projet dispose d'une base solide : transcription live/batch, isolation des clients, intégration API Albert, et système de mise à jour Git. Voici les axes d'évolution majeurs suggérés pour franchir un nouveau palier :

## 1. Intelligence Assistée (Post-traitement LLM)
Actuellement, le projet produit du texte brut. L'étape logique est l'intégration d'un LLM (local via Ollama/vLLM ou via Albert) :
- **Résumé automatique** : Générer un compte-rendu structuré dès la fin de la transcription.
- **Action Points** : Extraire automatiquement les décisions et tâches (TODO list).
- **Nettoyage "Smart"** : Supprimer les tics de langage (euh, bah, alors) sans altérer le sens.

## 2. Diarisation Avancée & Voice ID
La diarisation actuelle identifie "Speaker 0", "Speaker 1".
- **Profils Vocaux** : Permettre à l'utilisateur de nommer un speaker et de sauvegarder son "empreinte vocale". Lors de la prochaine réunion, le système reconnaîtrait automatiquement "Olivier" ou "Julie".
- **Correction Manuelle** : Une interface permettant de glisser-déposer des segments d'un speaker à un autre en cas d'erreur de l'IA.

## 3. Expérience Utilisateur (UX/UI)
- **Synchronisation Audio/Texte** : Dans la vue historique, cliquer sur une phrase pour jouer l'audio correspondant précisément à ce timestamp.
- **Visualisation Waveform** : Afficher la forme d'onde audio pour aider à repérer les silences ou les moments de parole intense.
- **Édition Collaborative** : Permettre à plusieurs utilisateurs de corriger la transcription en temps réel (via WebSockets).

## 4. Recherche Sémantique (RAG)
Si le volume de transcriptions augmente :
- **Moteur de recherche intelligent** : Rechercher dans l'historique non pas par mots-clés, mais par concept (ex: "Quelle était l'échéance du projet Alpha ?").
- **Tableau de bord** : Statistiques sur le temps de parole par intervenant.

## 5. Architecture & Sécurité
- **Authentification forte** : Passer des UUIDs localStorage à un vrai système OIDC (Keycloak/Auth0) pour un usage en entreprise.
- **Support Multi-GPU** : Optimiser la file d'attente des jobs batch pour répartir la charge sur plusieurs cartes graphiques.

---
> [!TIP]
> L'intégration du **résumé automatique** est probablement l'évolution ayant le plus fort "ROI" (retour sur investissement) pour l'utilisateur final.
