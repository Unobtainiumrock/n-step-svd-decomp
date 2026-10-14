import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, ArrowRight, Loader2 } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

interface Post {
  id: string;
  content: string;
  type: 'text' | 'video';
  preview: string;
}

interface CalibrationState {
  round: number;
  strategy: 'BROAD_EXPLOR' | 'TOPIC_REFINE' | 'TONE_REFINE' | 'FINE_TUNING';
  confidence: number;
  posts: Post[];
  selectedPost: Post | null;
  isComplete: boolean;
  sessionId: string | null;
}

const CalibrationPage: React.FC = () => {
  const navigate = useNavigate();
  const [state, setState] = useState<CalibrationState>({
    round: 1,
    strategy: 'BROAD_EXPLOR',
    confidence: 0,
    posts: [],
    selectedPost: null,
    isComplete: false,
    sessionId: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [showMessage, setShowMessage] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [isSelecting, setIsSelecting] = useState(false);
  const [isHovering, setIsHovering] = useState(false);

  const showMessageWithFade = async (message: string) => {
    setShowMessage(false);
    setCurrentMessage('');
    
    // Wait for fade out to complete
    await new Promise(resolve => setTimeout(resolve, 800));
    
    setCurrentMessage(message);
    setShowMessage(true);
  };

  const getMessage = useCallback((round: number, strategy: CalibrationState['strategy']): string => {
    if (round === 1) {
      return "Now, we need to get a feel for 'tone'. I'm going to present you some sample social media posts. Select which ones you like most. Don't overthink it—your first instinct is exactly what we're looking for.";
    }
    
    const strategyMessages = {
      'BROAD_EXPLOR': "Let's explore different content styles. Which post resonates more with you?",
      'TOPIC_REFINE': "Now we're refining your topic preferences. Which content feels more like you?",
      'TONE_REFINE': "We're fine-tuning your tone preferences. Which style speaks to you?",
      'FINE_TUNING': "Almost there! These final choices will perfect your tone profile."
    };
    
    return strategyMessages[strategy] || "Which post better represents your communication style?";
  }, []);

  const getOrbitalPosition = (index: number, total: number) => {
    const baseAngle = (index / total) * 2 * Math.PI;
    const radius = 180 + (index % 4) * 40; // Same spacing as onboarding
    return {
      angle: baseAngle,
      radius: radius
    };
  };

  const generateOptionPositions = (optionList: Post[]) => {
    return optionList.map((_, index) => getOrbitalPosition(index, optionList.length));
  };

  const loadRoundPosts = useCallback(async () => {
    setIsLoading(true);
    try {
      // Get next pair from API
      const response = await fetch('http://localhost:8000/calibration/next-pair');
      
      if (!response.ok) {
        if (response.status === 400) {
          // Calibration complete
          setState(prev => ({ ...prev, isComplete: true }));
          setShowResults(true);
          toast.success('Calibration complete! Your tone profile has been discovered.');
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }
      
      const pair = await response.json();
      
      // Convert API response to our Post format
      const posts: Post[] = [
        {
          id: pair.content_a.id,
          content: pair.content_a.text,
          type: 'text',
          preview: pair.content_a.text.substring(0, 100) + '...'
        },
        {
          id: pair.content_b.id,
          content: pair.content_b.text,
          type: 'text',
          preview: pair.content_b.text.substring(0, 100) + '...'
        }
      ];
      
      setState(prev => ({
        ...prev,
        posts,
        round: pair.choices_made + 1,
        confidence: pair.calibration_progress,
        strategy: getStrategyFromString(pair.strategy_used)
      }));
      
    } catch (error) {
      console.error('Error loading posts:', error);
      toast.error('Failed to load calibration posts. Using demo data.');
      
      // Fallback to mock data
      const mockPosts: Post[] = [
        {
          id: '1',
          content: 'Just had an amazing breakthrough in our AI research! The new model is showing incredible results that could revolutionize how we approach machine learning.',
          type: 'text',
          preview: 'Just had an amazing breakthrough in our AI research!'
        },
        {
          id: '2', 
          content: 'Thinking about the future of work and how remote collaboration is changing everything. The way we connect and create together has fundamentally shifted.',
          type: 'text',
          preview: 'Thinking about the future of work and how remote collaboration...'
        }
      ];
      
      setState(prev => ({
        ...prev,
        posts: mockPosts
      }));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getStrategyFromString = (strategy: string): CalibrationState['strategy'] => {
    switch (strategy.toLowerCase()) {
      case 'broad_exploration':
        return 'BROAD_EXPLOR';
      case 'topic_refinement':
        return 'TOPIC_REFINE';
      case 'tone_refinement':
        return 'TONE_REFINE';
      case 'fine_tuning':
        return 'FINE_TUNING';
      default:
        return 'BROAD_EXPLOR';
    }
  };

  useEffect(() => {
    // Initialize calibration session
    const initializeCalibration = async () => {
      try {
        const response = await fetch('http://localhost:8000/calibration/start', {
          method: 'POST'
        });
        
        if (response.ok) {
          const data = await response.json();
          setState(prev => ({ ...prev, sessionId: data.session_id }));
          await loadRoundPosts();
        } else {
          throw new Error('Failed to start calibration');
        }
      } catch (error) {
        console.error('Error starting calibration:', error);
        toast.error('Failed to start calibration. Using demo mode.');
        await loadRoundPosts();
      }
    };

    initializeCalibration();
  }, [loadRoundPosts]);

  useEffect(() => {
    if (state.posts.length > 0) {
      showMessageWithFade(getMessage(state.round, state.strategy));
      setShowOptions(true);
    }
  }, [state.posts, state.round, state.strategy, getMessage]);

  const handlePostSelection = async (selectedPost: Post) => {
    if (isSelecting) return;
    
    setIsSelecting(true);
    setState(prev => ({ ...prev, selectedPost }));

    // Wait for the selection animation to complete
    await new Promise(resolve => setTimeout(resolve, 1500));

    setIsLoading(true);
    
    try {
      // Record choice with API
      const response = await fetch('http://localhost:8000/calibration/choice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pair_id: `${selectedPost.id}_pair`,
          chosen_content_id: selectedPost.id,
          confidence: 1.0
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Update state with API response
        setState(prev => ({
          ...prev,
          confidence: data.calibration_confidence,
          round: data.choices_made + 1
        }));
        
        // Check if calibration is complete
        if (data.is_calibrated || data.choices_made >= 9) {
          setState(prev => ({ ...prev, isComplete: true }));
          setShowResults(true);
          toast.success('Calibration complete! Your tone profile has been discovered.');
        } else {
          // Load next round
          await loadRoundPosts();
        }
      } else {
        throw new Error('Failed to record choice');
      }
    } catch (error) {
      console.error('Error recording choice:', error);
      toast.error('Failed to record choice. Using demo mode.');
      
      // Fallback behavior
      setState(prev => ({
        ...prev,
        confidence: Math.min(prev.confidence + 0.1 + Math.random() * 0.1, 1.0)
      }));
      
      if (state.round >= 9 || state.confidence >= 0.95) {
        setState(prev => ({ ...prev, isComplete: true }));
        setShowResults(true);
        toast.success('Calibration complete! Your tone profile has been discovered.');
      } else {
        setState(prev => ({
          ...prev,
          round: prev.round + 1,
          strategy: getNextStrategy(prev.round + 1)
        }));
        await loadRoundPosts();
      }
    } finally {
      setIsLoading(false);
      setIsSelecting(false);
    }
  };

  const getNextStrategy = (round: number): CalibrationState['strategy'] => {
    if (round <= 2) return 'BROAD_EXPLOR';
    if (round <= 6) return 'TOPIC_REFINE';
    if (round <= 8) return 'TONE_REFINE';
    return 'FINE_TUNING';
  };

  if (showResults) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
        {/* Mystical background effects */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
          <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
        </div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-4xl relative z-10"
        >
          <div className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 mb-8 min-h-[500px] flex flex-col justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
            
            <div className="text-center relative z-10">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-6" />
              <h2 className="text-3xl font-bold mb-4 text-white">Calibration Complete!</h2>
              <p className="text-xl text-gray-300 mb-8">
                Your optimal tone profile has been discovered with {Math.round(state.confidence * 100)}% confidence.
              </p>
              <div className="flex gap-6 justify-center">
                <motion.button 
                  onClick={() => navigate('/results')}
                  className="bg-gradient-to-r from-green-600/90 to-green-700/90 hover:from-green-500/90 hover:to-green-600/90 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-green-500/30 shadow-2xl shadow-green-500/20"
                  whileHover={{ 
                    scale: 1.05,
                    boxShadow: "0 0 30px rgba(34, 197, 94, 0.4)"
                  }}
                  whileTap={{ scale: 0.95 }}
                >
                  View Results
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
                <motion.button 
                  onClick={() => window.location.reload()}
                  className="bg-gradient-to-r from-gray-600/90 to-gray-700/90 hover:from-gray-500/90 hover:to-gray-600/90 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-gray-500/30 shadow-2xl shadow-gray-500/20"
                  whileHover={{ 
                    scale: 1.05,
                    boxShadow: "0 0 30px rgba(107, 114, 128, 0.4)"
                  }}
                  whileTap={{ scale: 0.95 }}
                >
                  Start Over
                </motion.button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
      {/* Mystical background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="w-full max-w-4xl relative z-10">
        {/* Main Void Interface */}
        <div className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 mb-8 min-h-[500px] flex flex-col justify-center relative overflow-hidden">
          {/* Subtle inner glow */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
          
          <AnimatePresence mode="wait">
            <motion.div
              key={state.round}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="text-center relative z-10"
            >
              <motion.div 
                className="text-3xl text-white mb-12 leading-relaxed font-light"
                initial={{ opacity: 0 }}
                animate={{ opacity: showMessage ? 1 : 0 }}
                transition={{ duration: 1.5, ease: "easeOut" }}
              >
                {currentMessage}
              </motion.div>

              {/* Progress indicator */}
              <div className="mb-8">
                <div className="text-sm text-gray-400 mb-2">Round {state.round} of 9</div>
                <div className="w-full bg-gray-800/50 rounded-full h-2 mb-2">
                  <motion.div 
                    className="bg-blue-500/70 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${(state.round / 9) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="text-xs text-gray-500">Confidence: {Math.round(state.confidence * 100)}%</div>
              </div>

              {showOptions && state.posts.length > 0 && (
                <div className="relative h-96 flex items-center justify-center">
                  {/* Glowing Center */}
                  <motion.div
                    className="absolute w-8 h-8 bg-blue-500/30 rounded-full blur-xl"
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.3, 0.6, 0.3]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                  <motion.div
                    className="absolute w-4 h-4 bg-blue-400/50 rounded-full blur-lg"
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 0.8, 0.5]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />

                  {state.posts.map((post, index) => {
                    const position = generateOptionPositions(state.posts)[index];
                    if (!position) return null;
                    
                    const isSelected = state.selectedPost?.id === post.id;
                    const isDisintegrating = state.selectedPost && state.selectedPost.id !== post.id;

                    return (
                      <motion.button
                        key={`${post.id}-${state.round}-${index}`}
                        initial={{ 
                          opacity: 0, 
                          scale: 0,
                          x: Math.cos(position.angle) * position.radius,
                          y: Math.sin(position.angle) * position.radius
                        }}
                        animate={{ 
                          opacity: isDisintegrating ? 0 : 1,
                          scale: isSelected ? 0 : (isDisintegrating ? 0 : 1),
                          x: isSelected ? 0 : Math.cos(position.angle) * position.radius,
                          y: isSelected ? 0 : Math.sin(position.angle) * position.radius + (isHovering ? 0 : Math.sin(Date.now() * 0.001 + index) * 3)
                        }}
                        style={{
                          background: 'linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(20, 20, 20, 0.95) 100%)',
                          border: '1px solid rgba(59, 130, 246, 0.3)',
                          boxShadow: '0 0 20px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          WebkitBackdropFilter: 'blur(10px)'
                        }}
                        transition={{ 
                          opacity: { delay: index * 0.05, duration: 0.5 },
                          scale: { duration: isSelected ? 1.5 : (isDisintegrating ? 0.8 : 0.1) },
                          x: { duration: isSelected ? 1.5 : 0.1 },
                          y: { duration: isSelected ? 1.5 : 0.1 }
                        }}
                        onClick={() => handlePostSelection(post)}
                        onMouseEnter={() => setIsHovering(true)}
                        onMouseLeave={() => setIsHovering(false)}
                        className="absolute text-white px-6 py-4 rounded-2xl text-sm font-medium transition-all duration-300 max-w-xs"
                        whileHover={{ 
                          scale: isSelecting ? 1 : 1.1,
                          boxShadow: isSelecting 
                            ? '0 0 20px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)' 
                            : '0 0 30px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
                        }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <div className="text-left">
                          <div className="text-xs font-medium text-blue-300 mb-2 uppercase tracking-wide">
                            {post.type === 'text' ? '📝 TEXT' : '🎥 VIDEO'}
                          </div>
                          <div className="text-sm leading-relaxed">
                            {post.content}
                          </div>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              )}

              {isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center py-12"
                >
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-500" />
                  <p className="text-gray-400">Analyzing your preferences...</p>
                </motion.div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default CalibrationPage;
