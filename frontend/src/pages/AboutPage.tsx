import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Target, Zap, Users, BookOpen, Code, Database } from 'lucide-react';

const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card text-center"
      >
        <h1 className="text-4xl font-bold mb-4">About Tone Calibration</h1>
        <p className="text-xl text-gray-600">
          The "Face ID for Tone" - Discovering your unique communication style through AI
        </p>
      </motion.div>

      {/* The Vision */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4">The Vision</h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          Just as Face ID learns your unique facial features through iterative recognition, 
          this system learns your unique tone preferences through iterative content choices. 
          The result is a personalized content generation system that understands not just 
          what you want to say, but how you want to say it.
        </p>
        <p className="text-gray-700 leading-relaxed">
          Our goal is to make content creation as natural and personalized as your fingerprint - 
          unique to you, instantly recognizable, and perfectly calibrated to your communication style.
        </p>
      </motion.div>

      {/* Technical Innovation */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-6">Technical Innovation</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary-600" />
              Residualized Tone Space
            </h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              Separates topic information from stylistic information by creating orthogonal subspaces. 
              This enables precise tone calibration by disentangling "what is said" from "how it is said".
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Target className="w-5 h-5 text-primary-600" />
              Convergent Search
            </h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              Strategic A/B comparison system that navigates semantic embedding space to converge 
              on your optimal content preferences through intelligent choice sequencing.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary-600" />
              Unsupervised Learning
            </h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              130 mathematical tone dimensions discovered through SVD analysis of residual vectors, 
              without requiring human-labeled training data.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Users className="w-5 h-5 text-primary-600" />
              Real Content Training
            </h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              Trained on 80,000+ real LinkedIn posts to ensure authentic tone discovery 
              and realistic content generation.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Mathematical Foundation */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-primary-600" />
          Mathematical Foundation
        </h2>
        <div className="bg-gray-50 rounded-lg p-4 mb-4">
          <div className="space-y-2 text-sm font-mono">
            <div><span className="text-gray-600">Full Embedding Space:</span> E = ℝ¹⁵³⁶</div>
            <div><span className="text-gray-600">Topic Subspace:</span> T ⊂ E (160 SVD components)</div>
            <div><span className="text-gray-600">Tone Subspace:</span> R = E ⊖ T (130 SVD components)</div>
            <div><span className="text-gray-600">Residual Vector:</span> rᵢ = eᵢ - Proj_T(eᵢ)</div>
            <div><span className="text-gray-600">Tone Score:</span> tone_score = w_r^(s) · rᵢ</div>
          </div>
        </div>
        <p className="text-gray-700 text-sm">
          For complete mathematical documentation, see the implementation guide in the project repository.
        </p>
      </motion.div>

      {/* Technology Stack */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Code className="w-6 h-6 text-primary-600" />
          Technology Stack
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">Backend</h3>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• Python with FastAPI</li>
              <li>• OpenAI text-embedding-3-small</li>
              <li>• Supabase for content storage</li>
              <li>• Pinecone vector database</li>
              <li>• Docker containerization</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Frontend</h3>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• React.js with TypeScript</li>
              <li>• Tailwind CSS for styling</li>
              <li>• Framer Motion for animations</li>
              <li>• Recharts for data visualization</li>
              <li>• PWA capabilities</li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Dataset */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Database className="w-6 h-6 text-primary-600" />
          Dataset & Performance
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">80,420</div>
            <p className="text-gray-600">LinkedIn Posts</p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">95%</div>
            <p className="text-gray-600">Convergence Rate</p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">9</div>
            <p className="text-gray-600">Average Rounds</p>
          </div>
        </div>
      </motion.div>

      {/* Future Directions */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4">Future Directions</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">Research</h3>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• Supervised learning for tone axis interpretation</li>
              <li>• Multi-modal content analysis (images, video)</li>
              <li>• Real-time content optimization</li>
              <li>• Cross-platform tone consistency</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Applications</h3>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• Personalized content generation</li>
              <li>• Brand voice consistency</li>
              <li>• Communication coaching</li>
              <li>• A/B testing optimization</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AboutPage;
