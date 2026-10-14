import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Download, Edit3, Sparkles, Loader2 } from 'lucide-react';
import { toast } from 'react-hot-toast';

interface ToneAxis {
  name: string;
  alignment: number;
  color: string;
}

interface ToneProfile {
  tone_axes: ToneAxis[];
  preferences: {
    format: string;
    exploration: number;
    consistency: number;
  };
  calibration_confidence: number;
  choices_made: number;
  preference_vector_norm: number;
}

interface OptimalPost {
  post_id: string;
  content: string;
  confidence: number;
  score: number;
  calibration_confidence: number;
  choices_made: number;
  preference_strength: number;
}

interface ResultsData {
  optimalPost: OptimalPost | null;
  toneProfile: ToneProfile | null;
  generatedContent: string;
}

const ResultsPage: React.FC = () => {
  const [results, setResults] = useState<ResultsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);

  // Add delay to prevent rapid-fire requests
  const fetchWithDelay = async (url: string, delay: number = 100) => {
    await new Promise(resolve => setTimeout(resolve, delay));
    return fetch(url);
  };

  useEffect(() => {
    // Fetch real results from backend with proper error handling
    const fetchResults = async () => {
      try {
        // Get calibration status first
        const statusResponse = await fetchWithDelay('http://localhost:8000/calibration/status', 200);
        if (!statusResponse.ok) {
          throw new Error(`Status request failed: ${statusResponse.status}`);
        }
        const status = await statusResponse.json();
        
        // Get real optimal post with delay
        let optimalPost = null;
        try {
          const optimalResponse = await fetchWithDelay('http://localhost:8000/calibration/optimal-post', 300);
          if (optimalResponse.ok) {
            optimalPost = await optimalResponse.json();
          } else {
            console.warn('Optimal post request failed:', optimalResponse.status);
          }
        } catch (error) {
          console.warn('Could not fetch optimal post:', error);
        }
        
        // Get real tone profile with delay and safe endpoint
        let toneProfile = null;
        try {
          const profileResponse = await fetchWithDelay('http://localhost:8000/calibration/tone-profile-safe', 400);
          if (profileResponse.ok) {
            toneProfile = await profileResponse.json();
          } else {
            console.warn('Tone profile request failed:', profileResponse.status);
          }
        } catch (error) {
          console.warn('Could not fetch tone profile:', error);
        }
        
        // Create results with real data
        const realResults: ResultsData = {
          optimalPost: optimalPost,
          toneProfile: toneProfile,
          generatedContent: 'Generating your personalized content...'
        };
        
        setResults(realResults);
        
        // Automatically generate content using onboarding data
        await generateContentFromUserData();
        
      } catch (error) {
        console.error('Error fetching results:', error);
        
        // Fallback to mock data
        const mockResults: ResultsData = {
          optimalPost: {
            post_id: 'mock_1',
            content: 'Sample optimal post content for demonstration.',
            confidence: 0.85,
            score: 0.92,
            calibration_confidence: 0.85,
            choices_made: 5,
            preference_strength: 0.78
          },
          toneProfile: null,
          generatedContent: 'Generating your personalized content...'
        };
        
        setResults(mockResults);
        
        // Try to generate content even with mock data
        await generateContentFromUserData();
      } finally {
        setIsLoading(false);
      }
    };

    fetchResults();
  }, []);

  const generateContentFromUserData = async () => {
    // Get comprehensive user data from localStorage
    const storedUserData = localStorage.getItem('userData');
    const userData = storedUserData ? JSON.parse(storedUserData) : null;

    if (!userData || !userData.contentTopic) {
      toast.error('Missing user data. Please complete onboarding first.');
      return;
    }

    console.log('🚀 Generating content with user data:', userData);
    
    setIsGenerating(true);
    try {
      const requestData = {
        topic: userData.contentTopic,
        goals: userData.userGoals || 'Engage with professional network',
        userData: userData
      };

      console.log('📤 Sending to OpenAI:', requestData);

      const response = await fetch('http://localhost:8000/content/generate-enhanced', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error('Failed to generate content');
      }

      const data = await response.json();
      
      console.log('📥 Received from backend:', data);
      
      // Update results with generated content
      setResults(prev => prev ? {
        ...prev,
        generatedContent: data.content
      } : null);
      
      toast.success('Content generated using your calibration data!');
      
    } catch (error) {
      console.error('Error generating content:', error);
      toast.error('Failed to generate content. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (results?.generatedContent) {
      const blob = new Blob([results.generatedContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'generated-content.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Content downloaded successfully!');
    } else {
      toast.error('No content to download');
    }
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'My Tone Calibration Results',
        text: 'Check out my personalized tone profile!',
        url: window.location.href
      });
    } else {
      navigator.clipboard.writeText(window.location.href);
      toast.success('Link copied to clipboard!');
    }
  };



  if (isLoading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
        {/* Mystical background effects */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
          <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
        </div>

        <div className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 text-center relative z-10">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-gray-400 relative z-10">Analyzing your tone profile...</p>
        </div>
      </div>
    );
  }

  if (!results) return null;

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
      {/* Mystical background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="w-full max-w-4xl relative z-10">
        {/* Main Results Interface */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 mb-8 min-h-[600px] flex flex-col justify-center relative overflow-hidden"
        >
          {/* Subtle inner glow */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
          
          <div className="text-center relative z-10">
            {/* Success Icon */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mb-8"
            >
              <Sparkles className="w-16 h-16 text-blue-400 mx-auto" />
            </motion.div>

            {/* Title */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="text-4xl font-bold text-white mb-4"
            >
              Your Tone Profile is Ready
            </motion.h1>

            {/* Confidence Display */}
            {results.optimalPost && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className="mb-8"
              >
                <div className="text-2xl font-semibold text-blue-400 mb-2">
                  {Math.round(results.optimalPost.confidence * 100)}% Confidence
                </div>
                <div className="text-gray-400">
                  Based on {results.optimalPost.choices_made} calibration choices
                </div>
              </motion.div>
            )}

            {/* Generated Content Display */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
              className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 mb-8 text-left max-w-2xl mx-auto"
            >
              <h3 className="text-lg font-semibold text-white mb-4 text-center">Your Personalized Content</h3>
              {isGenerating ? (
                <div className="text-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-500" />
                  <p className="text-gray-400">Generating your personalized content...</p>
                </div>
              ) : (
                <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {results.generatedContent}
                </div>
              )}
            </motion.div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.0 }}
              className="flex gap-6 justify-center"
            >
              <motion.button
                onClick={generateContentFromUserData}
                disabled={isGenerating}
                className="bg-gradient-to-r from-blue-600/90 to-blue-700/90 hover:from-blue-500/90 hover:to-blue-600/90 disabled:from-gray-600/50 disabled:to-gray-700/50 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-blue-500/30 shadow-2xl shadow-blue-500/20"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Generate New Content
                  </>
                )}
              </motion.button>
              
              <motion.button
                onClick={handleDownload}
                className="bg-gradient-to-r from-green-600/90 to-green-700/90 hover:from-green-500/90 hover:to-green-600/90 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-green-500/30 shadow-2xl shadow-green-500/20"
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: "0 0 30px rgba(34, 197, 94, 0.4)"
                }}
                whileTap={{ scale: 0.95 }}
              >
                <Download className="w-5 h-5" />
                Download
              </motion.button>
            </motion.div>
          </div>
        </motion.div>

        {/* Content Generation Form */}
        
      </div>
    </div>
  );
};

export default ResultsPage;
