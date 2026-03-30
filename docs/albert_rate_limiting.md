# Gestion du Rate Limiting pour l'API Albert

## Problème Identifié

L'API Albert impose une limite de 1000 requêtes par jour. Lorsque cette limite est dépassée, l'API retourne une erreur 429 (Too Many Requests). Les anciennes implémentations ne géraient pas correctement cette situation, entraînant des erreurs non traitées et des interruptions de service.

## Solution Implémentée

Nous avons mis en place un système complet de gestion du rate limiting avec les composants suivants :

### 1. Rate Limiter Intelligente (`AlbertRateLimiter`)

- **Gestion des 429 consécutifs** : Détecte les erreurs 429 et compte leur nombre
- **Basculage en mode mock** : Après 5 429 consécutifs, bascule automatiquement en mode mock pendant 1 heure
- **Vérification de la clé API** : Si aucune clé n'est configurée, utilise automatiquement le mode mock
- **Réinitialisation intelligente** : Rétablit le mode normal après le délai d'attente

### 2. Améliorations dans `_transcribe_albert`

- **Vérification préalable** : Avant de tenter une transcription, vérifie si le mode mock doit être utilisé
- **Message d'erreur clair** : Retourne un message explicite indiquant le basculement en mode mock
- **Gestion robuste des erreurs** : Le système ne bloque plus les sessions live en cas de quota dépassé

### 3. Améliorations dans `live.py`

- **Notification client** : Envoie un message d'erreur spécifique au client lorsque le quota est dépassé
- **Message utilisateur** : Affiche "Quota API dépassé (1000 req/jour). Réessayez demain."

## Configuration

### Variables d'environnement

```bash
# Clé API Albert (optionnelle)
ALBERT_API_KEY="votre_cle_api"

# URL de base de l'API Albert
ALBERT_BASE_URL="https://albert.api.etalab.gouv.fr/v1"

# Modèle à utiliser
ALBERT_MODEL_ID="openai/whisper-large-v3"
```

## Comportement du Système

### Normal
- Utilise l'API Albert avec les paramètres configurés
- Gère les erreurs 5xx avec retry automatique
- Gère les erreurs réseau avec retry exponentiel

### Rate Limiting Actif
- Détecte 5 erreurs 429 consécutives
- Bascule en mode mock pendant 1 heure
- Retourne une transcription simulée avec message d'erreur
- Envoie un message d'erreur au client dans les sessions live

### Pas de Clé API
- Utilise automatiquement le mode mock
- Fonctionne même sans accès réseau

## Exemples de Messages

### Mode Mock Actif
```
[*] Utilisation du mode mock pour Albert (quota dépassé ou pas de clé)
[MODE MOCK] Quota API dépassé. Utilisation du mode de secours.
```

### Message Client
```
Quota API dépassé (1000 req/jour). Réessayez demain.
```

## Bonnes Pratiques

1. **Surveillance** : Le rate limiter affiche les statistiques dans la console
2. **Reprise automatique** : Le système revient automatiquement au mode normal
3. **Expérience utilisateur** : Les clients reçoivent des messages clairs
4. **Compatibilité** : Le système fonctionne même en mode offline
5. **Optimisation des appels API** : Les partials sont désactivés par défaut pour réduire les appels fréquents à l'API Albert
6. **Rate limiting strict** : Les requêtes sont limitées à une par seconde avec verrouillage thread-safe pour éviter les conflits

## Dépannage

### Si vous voyez des messages "MODE MOCK"
- Vérifiez votre clé API Albert
- Vérifiez que vous n'avez pas dépassé le quota quotidien
- Patientez jusqu'à 1 heure pour la reprise automatique

### Pour tester le comportement
```bash
# Activer le mode mock forcé
export ALBERT_API_KEY=""
# Ou simuler plusieurs 429