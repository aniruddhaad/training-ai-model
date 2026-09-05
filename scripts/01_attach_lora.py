from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_path = r"C:\FinMod\models\SmolLM2-360M"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto"
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)