import torch
import numpy as np
from transformers import CamembertTokenizer, CamembertModel
import os
from tqdm import tqdm

def test_embedding_pipeline(model_path='data/models/finetuned-camembert', sample_size=100):
    print("🔍 Démarrage des tests de vérification des NaN...")
    
    # 1. Chargement des modèles
    print("\n1️⃣ Chargement des modèles...")
    try:
        model_name = 'camembert-base'
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
        if os.path.exists(model_path):
            model = CamembertModel.from_pretrained(model_path)
            print("✅ Modèle fine-tuné chargé depuis:", model_path)
        else:
            model = CamembertModel.from_pretrained(model_name)
            print("⚠️ Modèle base chargé (pas de fine-tuned trouvé)")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {str(e)}")
        return
    
    # 2. Préparation des données de test
    print("\n2️⃣ Préparation des données de test...")
    test_texts = [
        "Un livre très intéressant",
        "Un très long texte " * 100,  # Test avec un texte long
        "",  # Test avec un texte vide
        "😀 Test avec émojis 🎉",  # Test avec des caractères spéciaux
        "Test avec des     espaces multiples",
        "Test\navec\ndes\nretours\nà\nla\nligne",
    ]
    
    # 3. Tests systématiques
    print("\n3️⃣ Exécution des tests...")
    results = {
        'total_tested': 0,
        'nan_detected': 0,
        'nan_locations': []
    }
    
    def check_tensor_for_nan(tensor, step_name, text_sample):
        if torch.isnan(tensor).any():
            results['nan_detected'] += 1
            results['nan_locations'].append({
                'step': step_name,
                'text': text_sample[:100] + "..." if len(text_sample) > 100 else text_sample,
                'shape': tensor.shape,
                'nan_count': torch.isnan(tensor).sum().item()
            })
            return True
        return False

    with torch.no_grad():
        for text in tqdm(test_texts, desc="Testing samples"):
            results['total_tested'] += 1
            
            # Test de tokenization
            try:
                inputs = tokenizer(text, 
                                 return_tensors='pt',
                                 truncation=True, 
                                 padding=True, 
                                 max_length=512)
                
                # Test du forward pass
                outputs = model(**inputs)
                
                # Vérification des NaN à chaque étape
                check_tensor_for_nan(outputs.last_hidden_state, 
                                   "last_hidden_state", text)
                
                # Test de l'embedding CLS
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                check_tensor_for_nan(cls_embedding, "cls_embedding", text)
                
            except Exception as e:
                print(f"\n❌ Erreur lors du traitement du texte: {text[:100]}...")
                print(f"Error: {str(e)}")
    
    # 4. Rapport final
    print("\n4️⃣ Rapport final:")
    print(f"Total des échantillons testés: {results['total_tested']}")
    print(f"Nombre de NaN détectés: {results['nan_detected']}")
    
    if results['nan_locations']:
        print("\nDétails des NaN détectés:")
        for loc in results['nan_locations']:
            print(f"\n🔍 Étape: {loc['step']}")
            print(f"📝 Texte: {loc['text']}")
            print(f"📊 Shape: {loc['shape']}")
            print(f"❌ Nombre de NaN: {loc['nan_count']}")
    else:
        print("\n✅ Aucun NaN détecté !")

if __name__ == "__main__":
    test_embedding_pipeline()
