# Instructions pour le téléchargement manuel du modèle WeSpeaker

## Problème rencontré
Le test échoue car le téléchargement du modèle WeSpeaker depuis Hugging Face échoue avec une erreur 407 "authentication required" liée à un proxy.

## Fichier à télécharger
- **URL du modèle** : https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx
- **Nom du fichier** : model.onnx

## Emplacement du fichier
Le fichier doit être placé dans le répertoire suivant :
```
%USERPROFILE%\.wespeaker\en\model.onnx
```

## Instructions de téléchargement manuel

1. **Ouvrir un navigateur web** et accéder à l'URL du modèle :
   ```
   https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx
   ```

2. **Se connecter à Hugging Face** si nécessaire (utiliser votre compte Hugging Face existant)

3. **Télécharger le fichier** `model.onnx` (le fichier devrait se télécharger automatiquement)

4. **Créer le répertoire** `%USERPROFILE%\.wespeaker\en` si celui-ci n'existe pas encore

5. **Placer le fichier téléchargé** dans le répertoire créé :
   ```
   %USERPROFILE%\.wespeaker\en\model.onnx
   ```

## Alternative : Utilisation de curl avec proxy
Si vous préférez télécharger via la ligne de commande avec proxy :

```bash
curl -L -H "Authorization: Bearer VOTRE_JETON_HUGGING_FACE" \
  -o %USERPROFILE%\.wespeaker\en\model.onnx \
  https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx
```

Remplacer `VOTRE_JETON_HUGGING_FACE` par votre token Hugging Face valide.

## Vérification
Après le téléchargement, vous pouvez vérifier que le fichier existe :
```bash
dir %USERPROFILE%\.wespeaker\en\model.onnx
```

## Note importante
Une fois ce fichier téléchargé manuellement, les tests devraient fonctionner normalement. Ce fichier est utilisé par le moteur de diarisation CPU (`LightDiarizationEngine`) lorsqu'il est utilisé dans le mode CPU.