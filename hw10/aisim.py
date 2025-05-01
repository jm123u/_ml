import ollama

def chat():
    model = "llama3:8b"  
    
    while (prompt := input("\n> ")) != "exit":
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        print(f"\n{response['message']['content']}\n")

if __name__ == "__main__":
    chat()