const express = require('express');
const router = express.Router();
const cors = require('cors');
const natural = require('natural');
const sentiment = require('sentiment');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { debug } = require('console');

// Initialize sentiment analyzer
const sentimentAnalyzer = new sentiment();

// Load models
let toxicityModel = null;
let customModel = null;

// Initialize models on startup
async function initializeModels() {
    try {
        // Load toxicity model
        toxicityModel = await tf.loadLayersModel('file://./models/toxicity/model.json');
        console.log('Toxicity model loaded successfully');
        
        // Load custom trained model if exists
        const customModelPath = path.join(__dirname, '../models/custom/model.json');
        if (require('fs').existsSync(customModelPath)) {
            customModel = await tf.loadLayersModel(`file://${customModelPath}`);
            console.log('Custom model loaded successfully');
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Initialize models
initializeModels();

// Middleware
router.use(cors());
router.use(express.json());

// Text preprocessing function
function preprocessText(text) {
    // Convert to lowercase
    text = text.toLowerCase();
    
    // Remove special characters but keep spaces
    text = text.replace(/[^a-zA-Z0-9\s]/g, '');
    
    // Remove extra whitespaces
    text = text.replace(/\s+/g, ' ').trim();
    
    return text;
}

// Calculate risk scores
function calculateRiskScores(text) {
    const processedText = preprocessText(text);
    const words = processedText.split(' ');
    
    // Sentiment analysis
    const sentimentResult = sentimentAnalyzer.analyze(text);
    
    // Define risk keywords and patterns
    const highRiskKeywords = [
        'suicide', 'kill', 'die', 'death', 'harm', 'hurt', 'attack',
        'violence', 'abuse', 'threat', 'weapon', 'bomb', 'terror'
    ];
    
    const mediumRiskKeywords = [
        'angry', 'hate', 'fight', 'destroy', 'revenge', 'punish',
        'suffer', 'pain', 'cry', 'depressed', 'anxiety', 'stress'
    ];
    
    // Count risk keywords
    let highRiskCount = 0;
    let mediumRiskCount = 0;
    
    words.forEach(word => {
        if (highRiskKeywords.includes(word)) {
            highRiskCount++;
        } else if (mediumRiskKeywords.includes(word)) {
            mediumRiskCount++;
        }
    });
    
    // Calculate base risk score
    const keywordRiskScore = (highRiskCount * 10 + mediumRiskCount * 5) / words.length;
    
    // Adjust based on sentiment
    const sentimentScore = sentimentResult.score;
    const normalizedSentiment = Math.max(0, -sentimentScore) / 10; // Negative sentiment increases risk
    
    // Calculate final scores
    const criticalRiskScore = Math.min(1, keywordRiskScore * 2 + normalizedSentiment);
    const overallRiskScore = Math.min(1, (criticalRiskScore + Math.abs(sentimentScore) / 20) / 2);
    
    return {
        criticalRisk: criticalRiskScore,
        overallRisk: overallRiskScore,
        sentiment: sentimentResult,
        highRiskCount,
        mediumRiskCount,
        wordCount: words.length
    };
}

// API Routes

// Main analysis endpoint
router.post('/analyze', async (req, res) => {
    try {
        const { text } = req.body;
        debugger
        if (!text || text.trim().length === 0) {
            return res.status(400).json({
                error: 'Text is required for analysis'
            });
        }
        
        // Calculate risk scores
        const riskScores = calculateRiskScores(text);
        
        // Prepare response
        const response = {
            text: text,
            analysis: {
                criticalRisk: {
                    score: riskScores.criticalRisk,
                    level: riskScores.criticalRisk > 0.7 ? 'high' : 
                           riskScores.criticalRisk > 0.4 ? 'medium' : 'low',
                    percentage: Math.round(riskScores.criticalRisk * 100)
                },
                overallRisk: {
                    score: riskScores.overallRisk,
                    level: riskScores.overallRisk > 0.7 ? 'high' : 
                           riskScores.overallRisk > 0.4 ? 'medium' : 'low',
                    percentage: Math.round(riskScores.overallRisk * 100)
                },
                sentiment: {
                    score: riskScores.sentiment.score,
                    comparative: riskScores.sentiment.comparative,
                    positive: riskScores.sentiment.positive,
                    negative: riskScores.sentiment.negative
                },
                statistics: {
                    wordCount: riskScores.wordCount,
                    highRiskKeywords: riskScores.highRiskCount,
                    mediumRiskKeywords: riskScores.mediumRiskCount
                }
            },
            recommendations: generateRecommendations(riskScores),
            timestamp: new Date().toISOString()
        };
        
        res.json(response);
        
    } catch (error) {
        console.error('Analysis error:', error);
        res.status(500).json({
            error: 'An error occurred during analysis',
            message: error.message
        });
    }
});

// Batch analysis endpoint
router.post('/analyze/batch', async (req, res) => {
    try {
        const { texts } = req.body;
        
        if (!texts || !Array.isArray(texts)) {
            return res.status(400).json({
                error: 'Array of texts is required'
            });
        }
        
        const results = texts.map(text => {
            const riskScores = calculateRiskScores(text);
            return {
                text: text,
                criticalRisk: riskScores.criticalRisk,
                overallRisk: riskScores.overallRisk,
                sentiment: riskScores.sentiment.score
            };
        });
        
        res.json({
            count: results.length,
            results: results,
            summary: {
                averageCriticalRisk: results.reduce((sum, r) => sum + r.criticalRisk, 0) / results.length,
                averageOverallRisk: results.reduce((sum, r) => sum + r.overallRisk, 0) / results.length,
                highRiskCount: results.filter(r => r.criticalRisk > 0.7).length
            }
        });
        
    } catch (error) {
        console.error('Batch analysis error:', error);
        res.status(500).json({
            error: 'An error occurred during batch analysis',
            message: error.message
        });
    }
});

// Model prediction endpoint (if custom model is loaded)
router.post('/predict', async (req, res) => {
    try {
        const { text } = req.body;
        
        if (!customModel) {
            return res.status(503).json({
                error: 'Custom model not available'
            });
        }
        
        // Preprocess text for model
        const processedText = preprocessText(text);
        
        // Convert to tensor (implement based on your model's requirements)
        // This is a placeholder - adjust based on your actual model input format
        const inputTensor = tf.tensor2d([[/* tokenized/vectorized text */]]);
        
        // Make prediction
        const prediction = await customModel.predict(inputTensor).data();
        
        inputTensor.dispose();
        
        res.json({
            text: text,
            prediction: prediction[0],
            confidence: Math.max(...prediction)
        });
        
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({
            error: 'An error occurred during prediction',
            message: error.message
        });
    }
});

// Health check endpoint
router.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        models: {
            toxicity: toxicityModel ? 'loaded' : 'not loaded',
            custom: customModel ? 'loaded' : 'not loaded'
        },
        timestamp: new Date().toISOString()
    });
});

// Statistics endpoint
router.get('/stats', (req, res) => {
    res.json({
        totalRequests: 0, // Implement request counting
        averageResponseTime: 0, // Implement response time tracking
        modelsLoaded: {
            toxicity: !!toxicityModel,
            custom: !!customModel
        }
    });
});

// Helper function to generate recommendations
function generateRecommendations(riskScores) {
    const recommendations = [];
    
    if (riskScores.criticalRisk > 0.7) {
        recommendations.push({
            type: 'critical',
            message: 'High risk content detected. Immediate attention required.',
            action: 'Review and moderate content immediately'
        });
    } else if (riskScores.criticalRisk > 0.4) {
        recommendations.push({
            type: 'warning',
            message: 'Medium risk content detected. Review recommended.',
            action: 'Manual review suggested'
        });
    }
    
    if (riskScores.sentiment.score < -5) {
        recommendations.push({
            type: 'sentiment',
            message: 'Highly negative sentiment detected.',
            action: 'Consider emotional support resources'
        });
    }
    
    if (riskScores.highRiskCount > 3) {
        recommendations.push({
            type: 'keywords',
            message: 'Multiple high-risk keywords detected.',
            action: 'Content requires careful review'
        });
    }
    
    if (recommendations.length === 0) {
        recommendations.push({
            type: 'safe',
            message: 'Content appears to be safe.',
            action: 'No immediate action required'
        });
    }
    
    return recommendations;
}

module.exports = router;