# Test de débruitage d'images avec NAFNet

Ce script utilise le modèle NAFNet pour débruiter des images contenant du bruit et calcule les différentes métriques pour analyser le résultat. Il est conçu pour être exécuté principalement sur Google Colab, mais peut également être testé sur des machines locales (à vos risques et périls).

### Sources
- Google Colab Notebook (Base) : [NAFNet Colab Notebook](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)
- Code source NAFNet : [NAFNet](https://github.com/megvii-research/NAFNet)
- Base de données pour bruit sel et poivre : [Kaggle](https://www.kaggle.com/datasets/rajneesh231/salt-and-pepper-noise-images)


### Configuration des dossiers
Ce script utilise les fichiers/dossiers suivants :

- gt/ : Contient les images originales.
- noisy/salt_pepper/ : Contient les images bruitées avec le bruit de sel et poivre.
- noisy/gaussian/ : Contient les images bruitées avec le bruit gaussien.
- requirements : fichier pour installer les dépendances pour NIMA

Le script va créer (si ce n'est pas déjà fait) les dossiers de sorties

- denoisy/nafnet/salt_pepper/ : Dossier de sortie pour les images débruitées à partir des images bruitées sel et poivre.
- denoisy/nafnet/gaussian/ : Dossier de sortie pour les images débruitées à partir des images bruitées gaussiennes.

### Préparation des données

Vous avez deux solutions :
- Soit mettre les dossiers gt et noisy dans votre google drive et juste dérouler le script et créer un dossier NIMA où vous mettez le fichier requirements dedans.
- Ou alors changer le morceau de code qui monte google drive et créer vous-même les dossiers/fichiers mais assurez vous que ces dossiers soient présents et disponible dans l'environnement d'execution

### Execution

Executer le notebook et tout devrait *normalement* bien se passer. Ce script va calculer le PSNR, SSIM et le score NIMA moyen des images originaux et bruitées et débruitées pour évaluer le résultat.

### Perspectives

Ajouter un modèle plus récent comme CGNet pour comparer ou alors des métodes classiques.