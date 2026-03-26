Principe absolu pour ce projet : toujours privilégier une lib pip 
existante plutôt que de réimplémenter la logique manuellement.
Chaque fois que tu es tenté d'écrire plus de 10 lignes pour faire
quelque chose, cherche d'abord si une lib pip le fait en 1-2 lignes.
mets les js dans un fichier js (pour pouvoir linter)
mets les <style> dans un css (pour pouvoir linter)
n'utilise pas de cdn au pire demande moi de télécharger le fichier pour l'hoster dans le répertoire static
privilégie les contrôles HTML5 plutot qu'une explosion de javascript
privilégie le server side rendering quand c'est possible. (pour diminuer la part de javascript)