# Guardian LLM 🛡️

An advanced AI risk assessment system that analyzes research papers and AI models for potential ethical, safety, and societal risks. Guardian LLM uses state-of-the-art NLP techniques including Chain of Thought reasoning, LoRA adapters, and semantic analysis to provide comprehensive risk evaluations.

## 🌟 Features

- **Multi-dimensional Risk Analysis**: Evaluates AI research across multiple risk categories including bias & fairness, privacy, safety, dual-use potential, societal impact, and transparency
- **Chain of Thought (CoT) Reasoning**: Implements advanced reasoning techniques for thorough risk assessment
- **LoRA Adaptation**: Uses Low-Rank Adaptation for efficient model fine-tuning
- **Semantic Analysis**: SVD-based semantic risk detection in latent space
- **Document Support**: Analyzes various document formats including PDF, DOCX, and plain text
- **Real-time Monitoring**: Tracks and visualizes risk trends over time
- **Comprehensive Reporting**: Generates detailed risk reports with evidence and recommendations

## 📋 Prerequisites

- Python 3.9+ (Note: Python 3.13 may have compatibility issues with some packages)
- PostgreSQL (for pgvector support)
- 8GB+ RAM recommended
- macOS, Linux, or Windows

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/guardian_llm.git
cd guardian_llm
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Install additional models and data

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data (optional)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Set up PostgreSQL with pgvector (optional)

If you want to use the vector database features:

```bash
# Install PostgreSQL and pgvector extension
# On macOS:
brew install postgresql
brew install pgvector

# Create database and enable extension
createdb guardian_llm
psql guardian_llm -c "CREATE EXTENSION vector;"
```

### 6. Configure the application

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
DATABASE_URL=postgresql://localhost/guardian_llm
SECRET_KEY=your-secret-key-here
```

## 🏃‍♂️ Running the Application

### Start the Flask backend

```bash
cd backend
python app.py
```

The application will start on `http://localhost:5000`

### API Endpoints

- `GET /` - Home endpoint
- `POST /api/analyze/paper` - Analyze a research paper
- `POST /api/analyze/model` - Analyze an AI model
- `GET /api/risks` - Get all risk assessments
- `GET /api/risks/<id>` - Get specific risk assessment
- `GET /api/monitoring/trends` - Get risk trends over time
- `POST /api/report/generate` - Generate detailed risk report

## 📊 Usage Example

### Analyzing a research paper

```python
import requests

# Analyze a paper
response = requests.post('http://localhost:5000/api/analyze/paper', 
    json={
        'title': 'Your Paper Title',
        'abstract': 'Paper abstract...',
        'content': 'Full paper content...',
        'enhanced_mode': True
    }
)

result = response.json()
print(f"Overall Risk Score: {result['overall_risk_score']}")
print(f"Risk Categories: {result['risk_scores']}")
```

### Analyzing an AI model

```python
# Analyze a model
response = requests.post('http://localhost:5000/api/analyze/model',
    json={
        'model_name': 'gpt-4',
        'model_type': 'language_model',
        'description': 'Model description...',
        'capabilities': ['text generation', 'code generation'],
        'training_data': 'Web crawl data',
        'parameters': '175B'
    }
)
```

## 🏗️ Architecture

```
guardian_llm/
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── requirements.txt       # Python dependencies
│   ├── api/
│   │   ├── routes.py         # API endpoints
│   │   └── __init__.py
│   ├── core/
│   │   ├── guardian_engine.py     # Main analysis engine
│   │   ├── cot_analyzer.py      # Chain of Thought reasoning
│   │   ├── lora_adapter.py      # LoRA implementation
│   │   ├── semantic_analyzer.py  # SVD-based semantic analysis
│   │   ├── dynamic_risk_analyzer.py  # Dynamic risk assessment
│   │   └── vector_db_postgres.py # Vector database integration
│   └── models/
│       └── schemas.py         # Database models
└── README.md
```

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'pgvector'**
   ```bash
   pip install pgvector
   ```

2. **OSError: [E050] Can't find model 'en_core_web_sm'**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Python 3.13 compatibility issues**
   - Consider using Python 3.11 or 3.12 for better compatibility
   - Use flexible version requirements (>=) instead of fixed versions

4. **PostgreSQL connection errors**
   - Ensure PostgreSQL is running: `pg_ctl -D /usr/local/var/postgres start`
   - Check database exists: `createdb guardian_llm`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built as part of COMP8420 Advanced Topics in AI course
- Inspired by the need for responsible AI development
- Uses state-of-the-art NLP models from Hugging Face

## 📞 Contact

Your Name - Prashish Shrestha (prashish.shrestha@students.mq.edu.au)

Project Link: https://github.com/prashishshresthamq/guardian_llm
