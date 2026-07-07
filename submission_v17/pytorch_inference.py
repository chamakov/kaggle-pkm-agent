import torch
import torch.nn as nn

class PurePyTorchAgent(nn.Module):
    def __init__(self, num_cards=90, num_scalars=111, embedding_dim=32):
        super().__init__()
        
        # 1. Feature Extractor
        # Exactly matches CardEmbeddingExtractor
        self.embedding = nn.Embedding(num_embeddings=1310, embedding_dim=embedding_dim, padding_idx=0)
        
        features_dim = (num_cards * embedding_dim) + num_scalars
        
        # 2. MLP Extractor (Policy Net only, we don't need value net for inference)
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 3. Action Net
        self.action_net = nn.Linear(256, 700)
        
    def forward(self, card_ids, scalars):
        # Flatten embeddings
        embedded_cards = self.embedding(card_ids)
        embedded_cards_flat = embedded_cards.view(embedded_cards.size(0), -1)
        
        # Concat with scalars
        features = torch.cat([embedded_cards_flat, scalars], dim=1)
        
        # MLP
        latent_pi = self.policy_net(features)
        
        # Action logits
        logits = self.action_net(latent_pi)
        
        return logits

def load_agent(model_path="policy.pth"):
    model = PurePyTorchAgent()
    state_dict = torch.load(model_path, map_location="cpu")
    
    # We only need to load the weights relevant to the policy for inference
    inference_state_dict = {}
    inference_state_dict['embedding.weight'] = state_dict['features_extractor.embedding.weight']
    
    # MLP Policy
    for i in [0, 2, 4]:
        if f'mlp_extractor.policy_net.{i}.weight' in state_dict:
            inference_state_dict[f'policy_net.{i}.weight'] = state_dict[f'mlp_extractor.policy_net.{i}.weight']
            inference_state_dict[f'policy_net.{i}.bias'] = state_dict[f'mlp_extractor.policy_net.{i}.bias']
        elif f'mlp_extractor.shared_net.{i}.weight' in state_dict:
            inference_state_dict[f'policy_net.{i}.weight'] = state_dict[f'mlp_extractor.shared_net.{i}.weight']
            inference_state_dict[f'policy_net.{i}.bias'] = state_dict[f'mlp_extractor.shared_net.{i}.bias']
        else:
            raise KeyError(f"Could not find policy weights for layer {i} in state_dict")
    
    inference_state_dict['action_net.weight'] = state_dict['action_net.weight']
    inference_state_dict['action_net.bias'] = state_dict['action_net.bias']
    
    model.load_state_dict(inference_state_dict)
    model.eval()
    return model
