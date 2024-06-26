from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_and_save_model(model_name, save_directory):
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 保存模型到本地目录
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"模型 {model_name} 已保存到目录 {save_directory}")

def main():
    parser = argparse.ArgumentParser(description="加载并保存指定的模型")
    parser.add_argument(
        "--model", 
        choices=["facebook/opt-6.7b", "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"], 
        required=True, 
        help="要加载的模型名称"
    )
    parser.add_argument(
        "--save_dir", 
        required=True, 
        help="保存模型的本地目录",
        default='./'
    )
    args = parser.parse_args()

    load_and_save_model(args.model, args.save_dir)

if __name__ == "__main__":
    main()