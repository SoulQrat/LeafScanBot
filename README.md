[LeafScanBot](https://t.me/leafscan_bot) is a Telegram bot that identifies plant leaf diseases from photos (works for tomatoes, cucumbers, eggplants, watermelons, and peppers), as well as detects nutrient deficiencies.
The bot uses a fine-tuned *MobileNetV3-Large* model.

### Instalation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/LeafScanBot.git
   cd LeafScanBot
2. **Configure bot token**:
   Open `docker-compose.yml` and replace YOUR_BOT_TOKEN_HERE with your actual Telegram bot token.
3. **Build and run the Docker container**:
   ```bash
   docker-compose up --build -d

### Add or Modify Model

To add a new model or update existing ones, follow these steps:

1. **Open `model_registry.json`**  
   This file maps species, diseases, and nutrient classifiers to their corresponding model paths and labels.

2. **To add a new species**:
   - Update `"species_classifier"` model.
   - Add a new entry under `"species_labels"` using the next available index.
   - Make sure to **insert it in alphabetical order by species name**, not just append it.
   - Add a corresponding entry under `"disease_classifiers"`:
     - `"model_path"` pointing to the `.pth` file for the disease classifier.
     - `"disease_labels"` mapping output indices to class names.

4. **To update the nutrient model or labels**:
   - Modify the `"nutrients"` entry in `model_registry.json` to point to the new `.pth` model file if you're replacing it.
   - Update the `"nutrient_labels"` dictionary if your new model has different output classes.

5. **To replace an existing disease model**:
   - Replace the `.pth` file located at the path specified in `"model_path"` under the corresponding species in `"disease_classifiers"`.
   - If the number or names of classes have changed, update the `"disease_labels"` mapping accordingly.
