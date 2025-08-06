# Yolo

## Annotation avec CVAT

L'annotation se fait avec la plateforme https://www.cvat.ai/


## Installation des paquets

Utiliser [pip](https://pip.pypa.io/en/stable/) pour installer les paquets nécessaires

```bash
pip install -r requirements.txt
```

## Créer le dataset tiling

Le tiling peut permettre d'améliorer la détection sur les objets plus petits. Le dataset tiled va permettre d'entraîner un autre Yolo.
Pour cela, il est nécessaire d'avoir un dossier avec les jeux de train valid et test. 
Il faut pour cette partie modifier le fichier test_yolo_tiler.py dans le dossier ./tiling_dataset/tests/
Il faut modifier le chemin (src_detection) vers le dataset qu'on veut tiler. On peut aussi choisir la taille des tiles (slice_wh) et le pourcentage de recouvrement entre les tiles (overlap_wh)

```bash
if test_detection:
    src_detection = "../data_1506/"
    dst_detection = "./data/test_tiled_640_640"

    config_detection = TileConfig(
        slice_wh=(640, 640),             # Slice width and height
        overlap_wh=(0.1, 0.1),           # Overlap width and height (10% overlap in this example, or 64x48 pixels)
        input_ext=".jpg",
        output_ext=None,
        annotation_type="object_detection",
        margins=(0, 0, 0, 0),            # Left, top, right, bottom
        include_negative_samples=False,   # Inlude negative samples
        copy_source_data=False,          # Copy original source data to target directory
    )
```

```bash
cd tiling_dataset
python ./tests/test_yolo_tiler.py
```

## Créer le crop dataset

Le crop dataset va nous permettre d'entraîner un autre yolo pour utiliser une approche temporelle. 
Il faut modifier le fichier ./crop_dataset/grape_crop_generator.py
Il faut modifier les chemins d'input et sortie. On lui donne les dossiers individuellement. Il va crop le dossier en entrée
Il va prendre les images et les labels et découper les bbox présentes dans la labelisation pour obtenir les bbox en image en y ajoutant ou non une expansion pour prendre plus de contexte. 
On peut également lui demander une proportion de bbox sans objet labelisé. (Conseillé pour l'apprentissage de nos modèles de détection pour comprendre différence background/objet) 
Il est possible de choisir le facteur d'expansion (>1 -> expansion positive), la taille des bbox images (après expansion, toutes les bbox sont ramenées à cette taille en pixel, carré. L'algo complète le vide par des bandes grises si besoin) et la proportion de bbox background (regarder la proportion donnée dans la console et ne pas se fier à la valeure du negative ratio qui est surestimé dans les paramètres)

```bash
if __name__ == "__main__":
    generator = GrapeCropGenerator(
        images_dir="../data_1506/test/images",           # Your original images folder
        labels_dir="../data_1506/test/labels",           # Your original labels folder  
        output_dir="./test_bbox/",   # Output folder for crops
        crop_size=320,                
        expansion_factor=1.6,         
        negative_ratio=0.6         
    )
    
    generator.process_dataset()
```

```bash
cd crop_dataset
python grape_crop_generator.py
```

## Entraînement du modèle

### Paramétrage d'Adastra et création environnement de travail

Transférer ses données sur son espace Adastra (CT1 est spécifique à mon compte je pense, ce sera peut-être un autre espace), les poids des modèles à entraîner (yolo11x.pt pour finetuning, yolo11-p2.yaml pour entraînement total) et le package ultralytics-main dans le dossier work sur Adastra (+ d'espace)
```bash
scp -rv /chemin/vers/son/jeu/de/donnée nom_sur_adastra@adastra.cines.fr:/work/CT1/id_compte/nom_sur_adastra/data/
scp -rv /chemin/vers/ultralytics-main nom_sur_adastra@adastra.cines.fr:/work/CT1/id_compte/nom_sur_adastra/
scp -rv /chemin/vers/weights_to_train nom_sur_adastra@adastra.cines.fr:/work/CT1/id_compte/nom_sur_adastra/weights
```
Se connecter à Adastra en ssh

Rajouter un fichier 'dataset.yaml' au même niveau que les dossier train, val et test (faire attention à bien faire la correspondance entre les labels numérique et les noms de classe)

```bash
path: ''
train: ./train/images
val: ./val/images
test: ./test/images
nc : nombre_de_classe

names:
  0: nom_de_la_classe_0
  1: nom_de_la_classe_1
  2: nom_de_la_classe_2
```

On a donc la structure suivante : 

```bash
data
    train
        images
        labels
    val
        images
        labels
    test
        images
        labels
    dataset.yaml
```
Créer un environnement virtuel et installer les paquets nécessaires

```bash
module load cray-python
python3.11 -m venv nom_env
source nom_env/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install ultralytics
```

### Lancer un entraînement

Créer un fichier bash 

```bash
nano train.sh
```

Remplir le fichier avec les informations suivantes en modifiant dans les lignes # : account, constraint (si on veut utilise une autre machine), time (nombre de temps pour entraînement : dépend du nombre d'images, d'epochs, de la résolution des images en entrée (imgsz), de la machine utilisée...)

Modifier également les paramètres d'entraînement : chemin vers données (vers le dossier dataset.yaml), chemin vers les poids du modèle, name pour choisir nom du dossier de sortie, nombre d'epochs (combien de fois le modèle voit le dataset entier (regarder les graphes de fin d'entrainement pour estimer le nombre d'epoch idéal), imgsz : taille des images en entrée du modèle (+grande, + de ressources nécessaires mais meilleur perf normalement), batch : +grand -> + de ressources nécessaire mais entrainement + rapide, save_period = 10 : sauvegarde des poids du modèle toutes les 10 epochs pour pouvoir reprendre entraînement depuis sauvegarde, cache = True : + rapide mais + gourmand)

```bash
#!/bin/bash
#SBATCH --account=cad16452
#SBATCH --job-name="test srun"
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3:00:00

module purge
module load cpe/24.07
source ~/nom_env/bin/activate
cd /work/CT1/id_compte/nom_sur_adastra/ultralytics-main/ultralytics

srun -- yolo train data=/work/CT1/id_compte/nom_sur_adastra/data/nom_dataset/dataset.yaml model=/work/CT1/id_compte/nom_sur_adastra/weights/yolo11x.pt name = "nom_sortie" epochs = 100 imgsz = 640 batch =  16 save_period = 50 save = True cache = True   
```

Lancer un job : 

```bash
sbatch train.sh  
```

### Lancer une session interactive 

Il est également possible de lancer une session interactive pour voir l'intérieur de la console lors de l'entraînement et ainsi debugger les différentes soucis. Avec sbatch, le code se lance mais on ne voit rien de l'intérieur. Ici on peut voir les messages d'erreur et l'avancement de l'entraînement (pour estimer la durée par exemple)

Il faut modifier le nom d'account, constraint : machine utilisée, time : durée de la session
squeue --me va indiquer le noeud sur lequel se connecter. 

```bash
salloc --account=cad16452 --constraint=MI250 --job-name="interactive" --nodes=1 --time=1:00:00 --exclusive
squeue --me
ssh <noeud> et non jobid
source ~/nom_env/bin/activate
cd /work/CT1/id_compte/nom_sur_adastra/ultralytics-main/ultralytics
yolo train data=/work/CT1/id_compte/nom_sur_adastra/data/nom_dataset/dataset.yaml model=/work/CT1/id_compte/nom_sur_adastra/weights/yolo11x.pt name = "nom_sortie" epochs = 100 imgsz = 640 batch =  16 save_period = 50 save = True cache = True  
```

### Reprendre un entraînement depuis une sauvegarde

Il est possible de reprendre un entraînement depuis une sauvegarde, que ce soit en session interactive ou avec sbatch. Il faut alors modifier la commande yolo train :

```bash
yolo train resume =True model ="./work/CT1/id_compte/nom_sur_adastra/runs/detect/nom_sortie/weights/last.pt"
```

### Transférer résultats de l'entraînement 

Se mettre sur un terminal de sa machine local et non plus sur Adastra
```bash
scp -rv nom_sur_adastra@adastra.cines.fr:/work/CT1/id_compte/nom_sur_adastra/runs/detect/nom_sortie /chemin/vers/sauvegarde
```

## Faire l'inférence du modèle sur un jeu de données

Après avoir regardé les différents graphs de l'entraînement réalisé, nous pouvons utiliser les poids du modèle entraîné et faire une inférence sur un jeu de données

folder : jeu de données sur lequel on fait l'inférence

main_model : chemin vers les poids du modèle entraîné

main_eval_iou 0.2 : si IoU>0.2 (intersection over union) alors prédiction correcte sur visualisation en mode évaluation

main_model_size : Taille des images en entrée du main_model

main_model_iou : filtre nms, supprime les bbox qui se chevauchent iou > main_model_iou en faisant l'inférence du modèle sur l'image

main_model_conf :  filtre à partir duquel une prédiction est gardée (baisser la confiance augmente le nombre de prédictions)


bbox_model : chemin vers modèle entraîné sur les bbox pour utiliser approche temporelle (facultatif)

memory_window : nombre d'images précédentes avec lequel les prédictions du modèle main sont comparées

bbox_eval_iou : pareil que main_eval_iou

expansion_factor : facteur d'agrandissement des bbox détectées par main model et qui seront données à manger au bbox_model (donner + de contexte pour regarder un peu autour)

target size : taille des bboxs après agrandissement en faisaint cv2.resize

bbox_model_iou : même que main_model_iou

bbox_model_size : taille des images en entrée du bbox_model (taille finale des bbox, donc après expansion factor qui seront données en entrée du modèle)

bbox_model_conf : même chose que main_model_conf


use_sahi : utiliser le tiling pour l'inférence (donner les poids du modèle tiled dans l'argument : main_model)

slice_height : hauteur des tiles

slice_width : largeur des tiles

overlap_height/width_ratio : recouvrement sur la hauteur/largeur


global_iou_threshold : eviter les chevauchement de prédictions entre les prédictions au seins de même modèle ou intermodèle (+ il est petit, + il est restrictif)


output : chemin de sauvegarde de l'inférence

no_vis : pas de sauvegarde des images avec bbox

no_crops : éviter sauvegarde des bbox prédites

ground_truth_folder : chemin vers vérité terrain pour visualiser en mode évaluation (dans le dossier visualisation, bbox de couleurs diff si vraies ou non et apparition des vraies bbox)

ignore_class_mismatch : ignorer la classe prédite pour le mode évaluation

```bash
python grape_detection.py --folder "../4j" --main_model "./Yolo/normal/normal_data_1440/weights/best.pt" --output "./results/4j_temp" --memory_window 20 --enhanced_vis --ignore_class_mismatch --bbox_model "./Yolo/bbox/crop_data_320_320_480_480_p2/weights/best.pt" --bbox_eval_iou 0.2 --main_eval_iou 0.2 --bbox_model_size 480 --expansion_factor 1.75 --target_size 320 --slice_height 512 --slice_width 512 --overlap_height_ratio 0.3 --overlap_width_ratio 0.3 --bbox_model_conf 0.4
```

On obtient à la fin un dossier contenant : 
- les bbox sur toute la série dans "crops"
- les bbox identifiées dans "id_tracking_results"
- fichiers txt qui contiennent les bbox prédites par date dans "labels"
- les images de la séries avec les bbox pour visualiser dans "visualizations"
- 2 fichiers json et xml qui contiennent des informations sur les prédictions pour chaque date (json) et chaque prédiction (xml)

## Evaluer les performances du modèle sur notre jeu de données

Dans cette partie nous allons évaluer les performances de notre modèle avec le jupyter notebook dans le dossier metrics. 
Il faut ainsi lancer la commande suivante en modifiant les chemins : 
predictions_dir : chemin vers les fichier labels des prédictions à évaluer

gt1_dir : chemin vers annotation évidente (celle qui va servir à déterminer les TP et FN)

gt2_dir : chemin vers annotation complète (celle qui va servir à déterminer les FP)

iou_threshold : valeurs limites de recouvrement pour évaluer correspondance entre predictions et annotations (modifier les valeurs si besoin. + les valeurs sont grandes, + exigeant sur précision des bbox) 

```bash
pred_metrics = metrics.metrics(predictions_dir="../results/train_corrected_novis/labels", gt1_dir="../metrics/1st_plan_evident", gt2_dir="../metrics/full_plan", iou_thresholds =iou_thresholds)
```

## Segmenter les grappes et analyser leurs features

il faut donner le dossier id_tracking_results que l'on veut analyser
beta : sensibilité du random_walker, 17000 est une bonne valeur avec nos bbox. Surement très différents pour des bbox d'autres objets. 
save_mask : pour sauvegarder les masques de segmentation

```bash
python .\grape_segmentation.py "./results/train_corrected/id_tracking_results" "./results/segmentation/train_corrected" --beta 17000 --mode "cg_mg" --save_masks 
```
Output :
boxplot_analysis : boite à moustache pour représenter distribution de toutes les grappes par date
combined_enhanced_color_analysis : courbes des moyennes des grappes par date
individual_grape_charts : courbes pour chaque id de grappe 
masks : sauvegarde des masques de segmentation par id 
multi_grape_time_series_data : fichier qui contient les données pour tracer les graphs
                                                                                                                                                                  

                    





