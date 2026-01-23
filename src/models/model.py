import torch
import torch.nn as nn
import numpy as np

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    This implementation combines Generalized Matrix Factorization (GMF) and a
    Multi-Layer Perceptron (MLP) to create the NeuMF architecture.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_layers: list, dropout: float):
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # GMF embeddings (named layers)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings (named layers)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim * (2 ** (len(hidden_layers)-1)) )
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim * (2 ** (len(hidden_layers)-1)) )
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * (2 ** len(hidden_layers)) # 2 * embedding_dim * 2**(len(hidden_layers)-1)
        for output_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        # Initialize weights
        self.init_weights(seed=42)

    def init_weights(self, seed: int):
        """
        Initializes model weights using a fixed seed for reproducibility.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def get_latent_features(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Returns the concatenated latent features from GMF and MLP paths,
        before the final prediction layer.

        Args:
            user (torch.Tensor): Tensor of user indices.
            item (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: The concatenated feature vector.
        """
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user)
        item_emb_gmf = self.item_embedding_gmf(item)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user)
        item_emb_mlp = self.item_embedding_mlp(item)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Concatenate GMF and MLP outputs
        return torch.cat([gmf_output, mlp_output], dim=-1)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            user (torch.Tensor): Tensor of user indices.
            item (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Raw output scores (logits).
        """
        latent_features = self.get_latent_features(user, item)
        
        # Final prediction
        prediction = self.predict_layer(latent_features)
        
        # Return raw logits
        return prediction.squeeze(-1)

if __name__ == '__main__':
    # Example usage:
    num_users = 100
    num_items = 50
    embedding_dim = 16
    hidden_layers = [64, 32, 16]
    dropout = 0.1

    model = NCF(num_users, num_items, embedding_dim, hidden_layers, dropout)
    model.init_weights(seed=42)

    # Create dummy input
    user_input = torch.LongTensor([0, 1, 2])
    item_input = torch.LongTensor([10, 20, 30])

    # Get raw logits
    logits = model(user_input, item_input)
    print("Logits:", logits)
    print("Logits shape:", logits.shape)

    # Get latent features
    features = model.get_latent_features(user_input, item_input)
    print("\nLatent Features shape:", features.shape)
