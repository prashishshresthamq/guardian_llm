# Guardian LLM

## AI-Powered Ethical Risk Assessment for Research Papers

Guardian LLM is an advanced AI system designed to analyze research papers and identify potential ethical risks across multiple dimensions. Using state-of-the-art Hierarchical Risk Propagation Networks (HRPN) and domain-specific LoRA adapters, it provides comprehensive risk assessments to ensure responsible AI research.

## 🌟 Features

- **Multi-dimensional Risk Analysis**: Evaluates papers across 6 key risk categories:
  - Bias & Fairness
  - Privacy & Data Protection
  - Safety & Security
  - Dual-Use Potential
  - Societal Impact
  - Transparency & Accountability

- **Advanced HRPN Architecture**: Utilizes graph neural networks to model how risks propagate through different sections of research papers

- **Domain-Specific Adaptation**: Specialized LoRA adapters for biomedical, legal, and technical domains

- **Real-time Feedback System**: Dynamic accuracy improvement through user feedback

- **Comprehensive Dashboard**: Track analysis history, view statistics, and monitor system performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher (tested up to 3.13)
- PostgreSQL with pgvector extension
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/prashishshresthamq/guardian_llm.git
cd guardian-llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch Geometric (optional, for full HRPN functionality):
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

6. Set up the database:
```bash
python check_db.py
python seed_data.py
```

### Configuration

Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://user:password@localhost/guardian_llm
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
guardian_llm/
├── backend/
│   ├── core/
│   │   ├── guardian_engine.py      # Main analysis engine
│   │   ├── hrpn_model.py          # HRPN implementation
│   │   ├── risk_analyzers.py      # Risk assessment modules
│   │   ├── semantic_analyzer.py   # Semantic analysis
│   │   └── lora_adapter.py        # LoRA adapter management
│   ├── models/                     # Database models
│   ├── migrations/                 # Database migrations
│   └── tests/                      # Unit tests
├── checkpoints/                    # Trained model weights
├── data/                          # Training and evaluation data
├── static/                        # Frontend assets
├── templates/                     # HTML templates
├── app.py                         # Flask application
└── requirements.txt               # Python dependencies
```

## 🔧 API Endpoints

### Analysis Endpoint
```
POST /api/analyze
Content-Type: application/json

{
  "title": "Paper Title",
  "content": "Paper content or abstract",
  "authors": ["Author 1", "Author 2"]
}
```

### Feedback Endpoint
```
POST /api/feedback/accuracy
Content-Type: application/json

{
  "paper_id": "uuid",
  "risk_category": "bias_fairness",
  "is_accurate": true,
  "reported_risk_level": "medium"
}
```

### Statistics Endpoint
```
GET /api/stats
```

### Recent Papers
```
GET /api/papers?limit=5&sort=upload_time&order=desc
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest backend/tests/
```

## 🛠️ Development

### Adding New Risk Categories

1. Update `RISK_CATEGORIES` in `risk_analyzers.py`
2. Implement analyzer in `RiskAnalyzer` class
3. Add corresponding patterns in `semantic_analyzer.py`
4. Update frontend to display new category

### Training Custom Models

```bash
python backend/core/train_models.py --epochs 10 --batch-size 32
```

## 📊 Model Performance

- **Accuracy**: 95%+ on benchmark dataset
- **Processing Time**: ~2-5 seconds per paper
- **Supported Formats**: PDF, DOCX, TXT, and direct text input

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- PyTorch team for the deep learning framework
- The open-source community for various dependencies

## ⚠️ Disclaimer

Guardian LLM is designed as an assistive tool for ethical risk assessment. Final decisions should always involve human judgment and domain expertise.


---

