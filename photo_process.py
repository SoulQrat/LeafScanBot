import torch
import torch.nn.functional as F
import json

class PlantDiseaseRecognizer:
    def __init__(self, registry_path: str, device: str = "cpu"):
        self.device = device
        self.species_classifier = None
        self.species_labels = {}
        self.disease_classifiers = {}
        self.nutrient_model = None
        self.nutrient_labels = {}

        self._load_registry(registry_path)
        self._load_models()

    def _load_registry(self, registry_path: str):
        with open(registry_path, "r", encoding="utf-8") as f:
            self.registry = json.load(f)

        self.species_classifier_path = self.registry["species_classifier"]
        self.species_labels = self.registry["species_labels"]
        self.disease_registry = self.registry["disease_classifiers"]
        self.nutrient_model_path = self.registry["nutrients"]
        self.nutrient_labels = self.registry["nutrient_labels"]

    def _load_models(self):
        self.species_classifier = torch.load(self.species_classifier_path, map_location=self.device, weights_only=False)
        self.species_classifier.eval()

        for species_name, data in self.disease_registry.items():
            model_path = data["model_path"]
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.eval()
            self.disease_classifiers[species_name] = {
                "model": model,
                "labels": data["disease_labels"]
            }

        if self.nutrient_model_path:
            self.nutrient_model = torch.load(self.nutrient_model_path, map_location=self.device, weights_only=False)
            self.nutrient_model.eval()

    async def recognize(self, image_tensor: torch.Tensor, topk: int = 1):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        result = {}

        # Предсказание вида растения
        with torch.no_grad():
            species_logits = self.species_classifier(image_tensor)
            species_probs = F.softmax(species_logits, dim=1)
            topk_species_probs, topk_species_indices = torch.topk(species_probs, k=topk, dim=1)

        topk_species = []
        for prob, idx in zip(topk_species_probs[0], topk_species_indices[0]):
            species_name = self.species_labels.get(str(idx.item()), "неизвестно")
            topk_species.append((species_name, prob.item()))

        result["species_topk"] = topk_species
        best_species_name = topk_species[0][0]

        # Предсказание болезни
        if best_species_name != "неизвестно" and best_species_name in self.disease_classifiers:
            disease_model_info = self.disease_classifiers[best_species_name]
            disease_model = disease_model_info["model"]
            disease_labels = disease_model_info["labels"]

            with torch.no_grad():
                disease_logits = disease_model(image_tensor)
                disease_probs = F.softmax(disease_logits, dim=1)
                topk_disease_probs, topk_disease_indices = torch.topk(disease_probs, k=topk, dim=1)

            topk_diseases = []
            for prob, idx in zip(topk_disease_probs[0], topk_disease_indices[0]):
                disease_name = disease_labels.get(str(idx.item()), "неизвестная болезнь")
                topk_diseases.append((disease_name, prob.item()))

            result["disease_topk"] = topk_diseases
        else:
            result["disease_topk"] = []

        # Предсказание дефицита питательных веществ
        if self.nutrient_model:
            with torch.no_grad():
                nutrient_logits = self.nutrient_model(image_tensor)
                nutrient_probs = F.softmax(nutrient_logits, dim=1)
                topk_nutrient_probs, topk_nutrient_indices = torch.topk(nutrient_probs, k=topk, dim=1)

            topk_nutrients = []
            for prob, idx in zip(topk_nutrient_probs[0], topk_nutrient_indices[0]):
                nutrient_name = self.nutrient_labels.get(str(idx.item()), "неизвестный дефицит")
                topk_nutrients.append((nutrient_name, prob.item()))

            result["nutrient_topk"] = topk_nutrients
        else:
            result["nutrient_topk"] = []

        return result
