# CyTools_ImageStudio

Studio de génération et de retouche d’images par IA avec ComfyUI local ou distant, canvas d’inpainting, ControlNet, entraînement de LoRA et pilotage MCP.

ImageStudio est distribué sous [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only). Les licences des composants tiers figurent dans [LICENCES](LICENCES/SOURCES.txt).

Sous Windows, ouvrir **CyTools_ImageStudio.exe** une fois les dépendances installées. Le studio utilise CyUI pour l’interface et CyToolsCore 0.12.0 avec CyMCPSDK pour les jobs, la CLI et le service MCP. La procédure pour un nouveau poste figure en bas de cette page.

## Choisir ComfyUI

Dans **Options → Connexions ComfyUI**, choisir le moteur par défaut puis enregistrer. Le moteur du studio se lance au premier job.

| Moteur | Réglage |
| --- | --- |
| **ComfyUI déjà installé** | Choisir le fichier `main.py` et le Python qui lance cette installation. Le studio démarre un processus séparé avec ses propres dossiers de travail. |
| **Installation ImageStudio** | Choisir CUDA ou CPU et cliquer sur **Installer / reprendre ComfyUI**. Git et Python sont nécessaires. L’installation utilise ComfyUI v0.37.0 et un environnement PyTorch privé. |
| **Serveur local** | Indiquer l’adresse d’une instance en cours, par exemple `http://127.0.0.1:8188`. |
| **Serveur distant** | Indiquer l’URL de base du serveur ComfyUI. Un proxy HTTPS peut fournir l’authentification par jeton. Les images d’entrée sont envoyées au serveur choisi. |

## Réutiliser les modèles du PC

Dans **Options → Dossiers de recherche de modèles**, ajouter autant de dossiers que nécessaire. Le type **Bibliothèque de modèles ComfyUI** détecte les catégories sous un dossier `models` ou à la racine sélectionnée. Pour un dossier propre à un autre logiciel, choisir sa catégorie, par exemple `checkpoints`, `diffusion_models`, `loras` ou `controlnet`.

Les moteurs lancés par le studio lisent les modèles à leurs emplacements actuels. La **Bibliothèque d’écriture** reçoit les téléchargements et les LoRA que vous décidez d’y installer. Les serveurs externes exposent leurs propres chemins de recherche. Enregistrer les réglages puis cliquer sur **Actualiser les modèles**.

Pour utiliser Qwen Image Edit, sélectionner **Image vers image**, puis **Suggérer des modèles installés**. La recette nécessite le modèle de diffusion, l’encodeur de texte et le VAE compatibles.

L’application est livrée sans poids. L’onglet **Modèles** propose des fichiers Stable Diffusion, FLUX.1 et Qwen Image vérifiés par SHA256, ainsi qu’un téléchargement par URL HTTPS. Pour les modèles sous accès contrôlé, acceptez d’abord les conditions du fournisseur. Un jeton Hugging Face saisi dans le studio dure le temps de la session.

## Générer, retoucher et entraîner

- Dans **Studio**, choisir la tâche et les composants compatibles. **Suggérer des modèles installés** propose des recettes lorsque les poids nécessaires sont visibles sur le serveur.
- Pour retoucher, ouvrir une image, choisir **Retouche au masque** puis peindre en violet la zone à modifier. La molette zoome et Espace + glisser déplace la vue. Le canvas permet de réutiliser un résultat comme source.
- Ajouter des LoRA avec leur intensité et un ou plusieurs ControlNet avec carte préparée ou prétraitement Canny.
- Dans **Workflows**, ouvrir un export ComfyUI JSON ou un PNG avec métadonnées, analyser les nœuds et renseigner les entrées. Pour les graphes à sous-graphes ou widgets spéciaux, utiliser **Save (API Format)** dans ComfyUI. Les nœuds personnalisés doivent déjà être installés sur le moteur sélectionné.
- Dans **Entraînement LoRA**, sélectionner les images et leurs descriptions. Le studio peut reprendre un LoRA existant et installer les nouveaux poids dans la bibliothèque sur demande. L’entraînement guidé utilise les nœuds natifs d’un ComfyUI lancé par ImageStudio. Sur un serveur externe, soumettre un workflow d’entraînement compatible.
- **Exporter les fichiers du job** conserve les résultats et les métadonnées dans un dossier au choix. Les recettes JSON s’enregistrent dans **Options**, avec sauvegarde automatique facultative.

## Connecter un client IA

Dans **Options → MCP**, démarrer le service puis copier son URL dans le client. Le service utilise le profil local Free de CyToolsCore et partage le moteur de la fenêtre. La découverte MCP donne accès à `op_inspect_backend`, `op_suggest_models`, `op_generate`, `op_run_workflow` et `op_train_lora`. Le client suit les jobs avec `cy_get_job` et exporte les fichiers voulus avant `cy_finish_task`.

Pour utiliser un serveur MCP stdio autonome, fermer la fenêtre ou utiliser son URL MCP. Depuis le dossier de l’application, la commande stdio est la suivante.

```powershell
.\runtime\python\Scripts\python.exe .\app\tool.py mcp
```

## Mémoire

Pour les moteurs lancés par le studio, **Options** propose le calcul CPU, plusieurs profils VRAM, une réserve GPU, le déchargement après chaque job et l’arrêt complet du moteur. Le décodage VAE par tuiles se règle dans **Échantillonnage**. **Libérer la mémoire** sur un serveur externe envoie une demande au serveur sélectionné.

## Installation sur une autre machine

Sur Windows x64, installer Python 3.11 ou supérieur et récupérer le [wheel natif Windows de CyToolsCore 0.12.0](app/README.md#cytoolscore-platform-wheel). Depuis la racine du dépôt, lancer les commandes suivantes.

```powershell
powershell -ExecutionPolicy Bypass -File .\app\Install.ps1 -CoreWheel C:\chemin\vers\cytools_core-0.12.0-py3-none-win_amd64.whl
.\CyTools_ImageStudio.exe
```

Pour Linux x64, utiliser `sh app/install_linux.sh /chemin/vers/le-wheel-linux.whl`, puis `sh CyTools_ImageStudio.sh`. Linux demande un environnement graphique et les bibliothèques OpenGL requises par Dear ImGui. Le lancement Linux reste à valider.

[Couverture et vérifications](app/CapabilityCoverage.md) · [Installation développeur](app/README.md)
