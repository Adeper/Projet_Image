# Modèle NAFNet

Ce code contient une partie du code cloné depuis le dépôt GitHub suivant :  
**[NAFNet: Nonlinear Activation Free Network for Image Restoration, dépot de megvii-research](https://github.com/megvii-research/NAFNet/tree/main)**  

**Note importante** : Ceci notre création. Il appartient à ses auteurs originaux et a été cloné uniquement évaluer l'efficacité de leur modèles de débruitage dans le cadre de notre projet Image.

---

## Installation et configuration de l'environnement Python

Il est recommandé d'utiliser un environnement Python virtuel pour éviter les conflits avec les dépendances d'autres projets. Voici comment le configurer : 

### Mise en place de l'environnement

#### Sous Windows :
```bash
python -m venv nafnet_env
nafnet_env\Scripts\activate
```

#### Sous Linux :
```bash
python3 -m venv nafnet_env
source nafnet_env/bin/activate
```

### Installation des dépendances

Installez les packages nécessaires spécifiés dans le fichier requirements.txt :
```bash
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Execution des modèles :

Assurez-vous d'avoir les modèles correspondants avant de lancer la commande

```bash
python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
```
- --input_path: le chemin de l'image bruitée
- --output_path: le chemin où on veut sauvegarder l'image débruitée

---