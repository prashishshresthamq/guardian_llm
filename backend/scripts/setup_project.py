# scripts/setup_project.py
"""
Complete setup script for Guardian LLM
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_guardian_llm():
    print("🚀 Setting up Guardian LLM...")
    
    # 1. Create necessary directories
    directories = ['models', 'logs', 'uploads', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # 2. Initialize database
    from app import app, db
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")
        
        # Add CoT fields
        from migrations.add_cot_fields import upgrade_database
        upgrade_database()
    
    # 3. Initialize semantic model
    from scripts.initialize_semantic_model import initialize_semantic_model
    initialize_semantic_model()
    
    # 4. Create dummy LoRA adapters (for testing)
    import torch
    import pickle
    
    lora_dir = os.path.join('models', 'lora_adapters')
    os.makedirs(lora_dir, exist_ok=True)
    
    for domain in ['biomedical', 'legal', 'technical']:
        dummy_data = {
            'config': {'rank': 16, 'alpha': 32.0},
            'domain': domain,
            'lora_weights': {}
        }
        torch.save(dummy_data, os.path.join(lora_dir, f'lora_adapter_{domain}.pt'))
    print("✅ LoRA adapter files created")
    
    # 5. Add test data
    from scripts.seed_database import seed_database
    seed_database()
    
    print("\n✅ Guardian LLM setup complete!")
    print("Run 'python app.py' to start the application")

if __name__ == '__main__':
    setup_guardian_llm()