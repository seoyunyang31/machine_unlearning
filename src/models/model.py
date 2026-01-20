import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_layers, dropout):
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = 2 * embedding_dim
        for output_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        self._init_weights()

    def _init_weights(self):
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

    def forward(self, user, item):
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
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        
        # Final prediction
        prediction = self.predict_layer(concat_output)
        
        return prediction.squeeze()

if __name__ == '__main__':
    # Example usage:
    num_users = 100
    num_items = 50
    embedding_dim = 8
    hidden_layers = [64, 32, 16]
    dropout = 0.1

    model = NCF(num_users, num_items, embedding_dim, hidden_layers, dropout)

    # Create dummy input
    user_input = torch.LongTensor([0, 1, 2])
    item_input = torch.LongTensor([10, 20, 30])

    prediction = model(user_input, item_input)
    print(prediction)
    print(prediction.shape)
