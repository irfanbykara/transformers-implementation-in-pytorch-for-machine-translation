import torch
from data.data_module import Multi30kDataModule
from model.transformers import Transformer

class TransformerInference:
    def __init__(self, model, data_module, device="cpu", max_len=50):
        self.model = model
        self.data_module = data_module
        self.device = device
        self.max_len = max_len

        self.tokenizer_source = self.data_module.token_transform[self.data_module.SOURCE_LANGUAGE]
        self.vocab_source = self.data_module.vocab_transform[self.data_module.SOURCE_LANGUAGE]
        self.vocab_target = self.data_module.vocab_transform[self.data_module.TARGET_LANGUAGE]
        self.BOS_IDX = self.data_module.BOS_IDX
        self.EOS_IDX = self.data_module.EOS_IDX

        self.model.to(self.device)
        self.model.eval()

    def generate(self, source_sentence):
        # Tokenize and numericalize
        source_tokens = [self.BOS_IDX] + [self.vocab_source[token] for token in self.tokenizer_source(source_sentence)] + [self.EOS_IDX]
        source_tensor = torch.tensor(source_tokens, dtype=torch.long).unsqueeze(0).to(self.device)  # (1, seq_len)

        # Encode
        with torch.no_grad():
            source_embed = self.model.token_embedding(source_tensor)
            source_embed = self.model.pos_embedding(source_embed)
            memory = self.model.encoder(source_embed)

        # Start decoding
        target_indices = [self.BOS_IDX]

        for _ in range(self.max_len):
            target_tensor = torch.tensor(target_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            target_embed = self.model.token_embedding(target_tensor)
            target_embed = self.model.pos_embedding(target_embed)
            target_mask = self.model.generate_causal_mask(target_tensor.size(1)).to(self.device)

            with torch.no_grad():
                out = self.model.decoder(target_embed, memory, target_mask=target_mask)
                logits = self.model.output_linear(out)

            next_token = logits[0, -1].argmax().item()
            if next_token == self.EOS_IDX:
                break
            target_indices.append(next_token)

        tokens = [self.vocab_target.get_itos()[idx] for idx in target_indices[1:]]
        return " ".join(tokens)

if __name__ == "__main__":
    data_module = Multi30kDataModule()
    model = Transformer()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    inference_engine = TransformerInference(model, data_module, device=DEVICE)
    sentence = "A man is riding a horse in the countryside."
    translation = inference_engine.generate(sentence)

    print("Translation:", translation)
