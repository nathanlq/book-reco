import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertModel
from sklearn.preprocessing import MultiLabelBinarizer
import json
from typing import List
import torch.nn as nn
import asyncio
from common.utils import reconnect

MODEL_OUTPUT_DIR = 'data/models/finetuned-camembert'

class BookDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        all_unique_labels = set()
        for label_list in labels:
            all_unique_labels.update(label_list)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_unique_labels))}
        
    def __len__(self):
        return len(self.texts)
    
    def create_label_vector(self, labels):
        vector = torch.zeros(len(self.label_to_idx))
        for label in labels:
            if label in self.label_to_idx:
                vector[self.label_to_idx[label]] = 1
        return vector
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vector = self.create_label_vector(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_vector
        }
    
class FinetunedCamemBERT(pl.LightningModule):
    def __init__(self, learning_rate: float = 2e-5):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.bert = CamembertModel.from_pretrained('camembert-base')
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),  # Ajout d'une normalisation
            nn.ReLU(),
            nn.Dropout(0.1),    # Ajout d'un dropout
            nn.Linear(768, 128),
            nn.LayerNorm(128)   # Normalisation finale
        )
        
        # Freeze first 8 layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]
        projected = self.projection(embeddings)
        # Normalisation L2 pour garantir des vecteurs unitaires
        return torch.nn.functional.normalize(projected, p=2, dim=1)
    
    def _compute_loss(self, batch):
        embeddings = self(batch['input_ids'], batch['attention_mask'])
        
        # Calcul de similarité cosinus (produit scalaire de vecteurs normalisés)
        similarity = torch.matmul(embeddings, embeddings.T)
        
        labels = batch['labels']
        label_similarity = torch.matmul(labels, labels.T) > 0
        
        # Temperature plus élevée pour plus de stabilité
        temperature = 0.25
        
        # Clip pour éviter les explosions numériques
        similarity_scaled = torch.clamp(similarity / temperature, min=-20, max=20)
        exp_sim = torch.exp(similarity_scaled)
        
        mask = torch.eye(len(embeddings), device=embeddings.device)
        
        positive_pairs = torch.sum(exp_sim * label_similarity * (1 - mask), dim=1)
        all_pairs = torch.sum(exp_sim * (1 - mask), dim=1)
        
        # Ajout d'un epsilon pour éviter la division par zéro
        eps = 1e-8
        loss = -torch.log((positive_pairs + eps) / (all_pairs + eps)).mean()
        
        # Vérification de NaN dans la loss
        if torch.isnan(loss):
            print("Warning: NaN detected in loss calculation")
            print(f"Similarity stats: min={similarity.min()}, max={similarity.max()}")
            print(f"Exp_sim stats: min={exp_sim.min()}, max={exp_sim.max()}")
            print(f"Positive pairs: {positive_pairs}")
            print(f"All pairs: {all_pairs}")
            # Retourner une perte alternative en cas de NaN
            return torch.tensor(10.0, device=loss.device, requires_grad=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss)
        # Ajout de logs pour le monitoring
        if batch_idx % 100 == 0:
            embeddings = self(batch['input_ids'], batch['attention_mask'])
            self.log('embedding_norm', torch.norm(embeddings[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
async def get_training_data():
    conn = await reconnect()
    rows = await conn.fetch("""
        SELECT product_title, resume, labels 
        FROM books 
        WHERE labels IS NOT NULL
    """)
    await conn.close()
    
    texts = [f"{row['product_title']} {row['resume']}".strip() for row in rows]
    labels = [json.loads(row['labels']) for row in rows]
    
    return texts, labels

async def main():
    torch.set_float32_matmul_precision('high')
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    
    texts, labels = await get_training_data()
    
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    train_size = int(0.8 * len(texts))

    train_dataset = BookDataset(
        texts[:train_size],
        labels[:train_size],
        tokenizer
    )
    
    model = FinetunedCamemBERT()

    val_dataset = BookDataset(
        texts[train_size:],
        labels[train_size:],
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=11)

    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='camembert-finetuned-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss'
        ),
        EarlyStopping(monitor='val_loss', patience=3)
    ]
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.bert.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
