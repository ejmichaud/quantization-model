
from tqdm.auto import tqdm
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_names = [
    "pythia-19m",
    "pythia-125m",
    "pythia-350m",
    "pythia-800m",
    "pythia-1.3b",
    "pythia-2.7b",
    "pythia-6.7b",
    "pythia-13b"
]

model_names_deduped = [
    "pythia-19m-deduped",
    "pythia-125m-deduped",
    "pythia-350m-deduped",
    "pythia-800m-deduped",
    "pythia-1.3b-deduped",
    "pythia-2.7b-deduped",
    "pythia-6.7b-deduped",
    "pythia-13b-deduped"
]

# steps = list(range(1000, 144000, 1000))

if __name__ == '__main__':
    sizes = dict()
    for model_name in tqdm(model_names):
        model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step143000",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step143000",
        )
        embedding_numel = model.gpt_neox.embed_in.weight.numel()
        unembed_numel = model.embed_out.weight.numel()
        total_numel = sum(p.numel() for p in model.parameters())
        sizes[model_name] = (total_numel - embedding_numel - unembed_numel, total_numel - embedding_numel, total_numel) 
        del model
    torch.save(sizes, "/om/user/ericjm/results/the-everything-machine/pythia-0/num_params.pt")

