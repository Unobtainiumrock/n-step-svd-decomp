import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Check, X, Loader2 } from 'lucide-react';
import { toast } from 'react-hot-toast';

interface UserData {
  name: string;
  email: string;
  role: string;
  icpDescription: string;
  industry: string;
  audience: string;
  contentTopic: string;
  userGoals: string;
  // Enhanced data for AI pre-prompting
  toneProfile?: {
    selectedPosts: Array<{
      id: string;
      content: string;
      strategy: string;
      round: number;
      timestamp: string;
    }>;
    rejectedPosts: Array<{
      id: string;
      content: string;
      strategy: string;
      round: number;
      timestamp: string;
    }>;
    choiceHistory: Array<{
      round: number;
      strategy: string;
      acceptedChoice: {
        id: string;
        content: string;
      };
      rejectedChoice: {
        id: string;
        content: string;
      };
      timestamp: string;
    }>;
    chatHistory: Array<{
      text: string;
      isUser: boolean;
      timestamp: string;
    }>;
    calibrationMetrics: {
      finalConfidence: number;
      totalRounds: number;
      completedAt: string;
    };
  };
}

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

interface ConversationalOnboardingProps {
  onComplete: (data: UserData) => void;
}

type OnboardingStep = 
  | 'name'
  | 'email'
  | 'role'
  | 'icp'
  | 'industry'
  | 'audience'
  | 'contentTopic'
  | 'userGoals'
  | 'confirmation'
  | 'correction'
  | 'calibration'; // New calibration step

const industries = [
  'B2B SaaS',
  'Fintech',
  'Healthcare',
  'E-commerce',
  'Manufacturing',
  'Professional Services',
  'Education',
  'Real Estate',
  'Media & Entertainment',
  'Non-profit',
  'Government',
  'Other'
];

const audiences = [
  'Executive',
  'Technical',
  'Marketer',
  'General Consumer',
  'Developer',
  'Sales Professional',
  'HR Professional',
  'Student',
  'Other'
];

const ConversationalOnboarding: React.FC<ConversationalOnboardingProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState<OnboardingStep>('name');
  const [userData, setUserData] = useState<Partial<UserData>>({});
  const [currentMessage, setCurrentMessage] = useState('');
  const [showMessage, setShowMessage] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState<string[]>([]);
  const [optionPositions, setOptionPositions] = useState<{angle: number, radius: number}[]>([]);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{
    text: string;
    isUser: boolean;
    timestamp: string;
  }>>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Calibration state
  const [calibrationState, setCalibrationState] = useState<CalibrationState>({
    round: 1,
    strategy: 'BROAD_EXPLOR',
    confidence: 0,
    posts: [],
    selectedPost: null,
    isComplete: false,
    sessionId: null
  });
  const [isCalibrationLoading, setIsCalibrationLoading] = useState(false);
  const [choiceHistory, setChoiceHistory] = useState<Array<{
    round: number;
    strategy: string;
    acceptedChoice: {
      id: string;
      content: string;
    };
    rejectedChoice: {
      id: string;
      content: string;
    };
    timestamp: string;
  }>>([]);

  // Add back the getMessage function
  const getMessage = useCallback((step: OnboardingStep): string => {
    const messages = {
      name: "What's your name?",
      email: `Hi ${userData.name}, what's your email?`,
      role: "What's your professional title?",
      icp: "Describe your ideal customer. Who are you trying to reach?",
      industry: "What's your target industry?",
      audience: "Excellent. Who's your primary audience?",
      contentTopic: "What topics do you typically write about or want to focus on?",
      userGoals: "What are your main goals when creating content?",
      confirmation: "Great, we're all set! Let's just quickly review everything to make sure it's correct.",
      correction: "No problem at all. Which piece of information needs to be changed?",
      calibration: "I'm going to present you some sample social media posts. Select which ones you like most. Don't overthink it—your first instinct is exactly what we're looking for."
    };
    return messages[step];
  }, [userData.name]);

  // Calibration functions
  const getCalibrationMessage = useCallback((round: number, strategy: CalibrationState['strategy']): string => {
    if (round === 1) {
      return "I'm going to present you some sample social media posts. Select which ones you like most. Don't overthink it—your first instinct is exactly what we're looking for.";
    }
    
    const strategyMessages = {
      'BROAD_EXPLOR': "Let's explore different content styles. Which post resonates more with you?",
      'TOPIC_REFINE': "Now we're refining your topic preferences. Which content feels more like you?",
      'TONE_REFINE': "We're fine-tuning your tone preferences. Which style speaks to you?",
      'FINE_TUNING': "Almost there! These final choices will perfect your tone profile."
    };
    
    return strategyMessages[strategy] || "Which post better represents your communication style?";
  }, []);

  const loadRoundPosts = useCallback(async () => {
    setIsCalibrationLoading(true);
    try {
      // Get next pair from API
      const response = await fetch('http://localhost:8000/calibration/next-pair');
      
      if (!response.ok) {
        if (response.status === 400) {
          // Calibration complete
          setCalibrationState(prev => ({ ...prev, isComplete: true }));
          toast.success('Calibration complete! Your tone profile has been discovered.');
          setTimeout(() => onComplete(userData as UserData), 2000);
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
      
      setCalibrationState(prev => ({
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
      
      setCalibrationState(prev => ({
        ...prev,
        posts: mockPosts
      }));
    } finally {
      setIsCalibrationLoading(false);
    }
  }, [onComplete, userData]);

  const getStrategyFromString = (strategy: string | undefined | null): CalibrationState['strategy'] => {
    if (!strategy || typeof strategy !== 'string') {
      return 'BROAD_EXPLOR';
    }
    
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

  const handlePostSelection = async (selectedPost: Post) => {
    if (isSelecting) return;
    
    setIsSelecting(true);
    setCalibrationState(prev => ({ ...prev, selectedPost }));

    // Find the rejected post (the other post in the pair)
    const rejectedPost = calibrationState.posts.find(post => post.id !== selectedPost.id);
    
    if (!rejectedPost) return;

    // Track complete choice history (accepted vs rejected)
    const choiceRecord = {
      round: calibrationState.round,
      strategy: calibrationState.strategy,
      acceptedChoice: {
        id: selectedPost.id,
        content: selectedPost.content
      },
      rejectedChoice: {
        id: rejectedPost.id,
        content: rejectedPost.content
      },
      timestamp: new Date().toISOString()
    };
    setChoiceHistory(prev => [...prev, choiceRecord]);

    // Add to chat messages
    setChatMessages(prev => [...prev, 
      { text: `Selected: "${selectedPost.content.substring(0, 50)}..."`, isUser: true, timestamp: new Date().toISOString() }
    ]);

    // Wait for the selection animation to complete
    await new Promise(resolve => setTimeout(resolve, 1500));

    setIsCalibrationLoading(true);
    
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
        setCalibrationState(prev => ({
          ...prev,
          confidence: data.calibration_confidence,
          round: data.choices_made + 1,
          selectedPost: null // Reset selected post
        }));
        
        // Add AI response to chat
        const strategy = data.strategy_used || 'broad_exploration';
        const aiMessage = getCalibrationMessage(data.choices_made + 1, getStrategyFromString(strategy));
        setChatMessages(prev => [...prev, 
          { text: aiMessage, isUser: false, timestamp: new Date().toISOString() }
        ]);
        
        // Check if calibration is complete
        if (data.is_calibrated || data.choices_made >= 9) {
          setCalibrationState(prev => ({ ...prev, isComplete: true }));
          toast.success('Calibration complete! Your tone profile has been discovered.');
          
          // Prepare comprehensive user data with tone profile for AI pre-prompting
          const completeUserData: UserData = {
            ...userData as UserData,
            toneProfile: {
              selectedPosts: [],
              rejectedPosts: [],
              choiceHistory: choiceHistory,
              chatHistory: chatMessages,
              calibrationMetrics: {
                finalConfidence: calibrationState.confidence,
                totalRounds: calibrationState.round,
                completedAt: new Date().toISOString()
              }
            }
          };
          
          setTimeout(() => onComplete(completeUserData), 2000);
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
      setCalibrationState(prev => ({
        ...prev,
        confidence: Math.min(prev.confidence + 0.1 + Math.random() * 0.1, 1.0),
        round: prev.round + 1,
        selectedPost: null // Reset selected post
      }));
      
      if (calibrationState.round >= 9 || calibrationState.confidence >= 0.95) {
        setCalibrationState(prev => ({ ...prev, isComplete: true }));
        toast.success('Calibration complete! Your tone profile has been discovered.');
        
        // Prepare comprehensive user data with tone profile for AI pre-prompting
        const completeUserData: UserData = {
          ...userData as UserData,
          toneProfile: {
            selectedPosts: [],
            rejectedPosts: [],
            choiceHistory: choiceHistory,
            chatHistory: chatMessages,
            calibrationMetrics: {
              finalConfidence: calibrationState.confidence,
              totalRounds: calibrationState.round,
              completedAt: new Date().toISOString()
            }
          }
        };
        
        setTimeout(() => onComplete(completeUserData), 2000);
      } else {
        const nextStrategy = getNextStrategy(calibrationState.round + 1);
        setCalibrationState(prev => ({
          ...prev,
          strategy: nextStrategy
        }));
        const aiMessage = getCalibrationMessage(calibrationState.round + 1, nextStrategy);
        setChatMessages(prev => [...prev, 
          { text: aiMessage, isUser: false, timestamp: new Date().toISOString() }
        ]);
        await loadRoundPosts();
      }
    } finally {
      setIsCalibrationLoading(false);
      setIsSelecting(false);
    }
  };

  const getNextStrategy = (round: number): CalibrationState['strategy'] => {
    if (round <= 2) return 'BROAD_EXPLOR';
    if (round <= 6) return 'TOPIC_REFINE';
    if (round <= 8) return 'TONE_REFINE';
    return 'FINE_TUNING';
  };

  // Remove all orbital physics - just keep the position generation
  const getOrbitalPosition = (index: number, total: number) => {
    const baseAngle = (index / total) * 2 * Math.PI;
    // Increase spacing for calibration posts - they need more room
    const radius = 220 + (index % 4) * 60; // Back to original spacing
    return {
      angle: baseAngle,
      radius: radius
    };
  };

  const generateOptionPositions = (optionList: string[]) => {
    return optionList.map((_, index) => getOrbitalPosition(index, optionList.length));
  };

  const showMessageWithFade = async (message: string) => {
    setShowMessage(false);
    setCurrentMessage('');
    
    // Wait for fade out to complete
    await new Promise(resolve => setTimeout(resolve, 800));
    
    setCurrentMessage(message);
    setShowMessage(true);
  };

  useEffect(() => {
    showMessageWithFade(getMessage(currentStep));
  }, [currentStep, getMessage]);

  useEffect(() => {
    if (inputRef.current && !showOptions) {
      inputRef.current.focus();
    }
  }, [currentStep, showOptions]);

  const handleInputSubmit = () => {
    if (!inputValue.trim()) return;

    switch (currentStep) {
      case 'name':
        setUserData(prev => ({ ...prev, name: inputValue.trim() }));
        setCurrentStep('email');
        break;
      case 'email':
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(inputValue.trim())) {
          showMessageWithFade("Hmm, that doesn't look like a valid email. Could you please try again?");
          return;
        }
        setUserData(prev => ({ ...prev, email: inputValue.trim() }));
        setCurrentStep('role');
        break;
      case 'role':
        setUserData(prev => ({ ...prev, role: inputValue.trim() }));
        setCurrentStep('icp');
        break;
      case 'icp':
        setUserData(prev => ({ ...prev, icpDescription: inputValue.trim() }));
        setCurrentStep('industry');
        setShowOptions(true);
        setOptions(industries);
        setOptionPositions(generateOptionPositions(industries));
        break;
      case 'contentTopic':
        setUserData(prev => ({ ...prev, contentTopic: inputValue.trim() }));
        setCurrentStep('userGoals');
        break;
      case 'userGoals':
        setUserData(prev => ({ ...prev, userGoals: inputValue.trim() }));
        setCurrentStep('confirmation');
        break;
    }
    
    setInputValue('');
  };

  const handleOptionSelect = async (option: string) => {
    if (isSelecting) return;
    
    console.log('Option selected:', option);
    setIsSelecting(true);
    setSelectedOption(option);

    // Wait for the selection animation to complete
    await new Promise(resolve => setTimeout(resolve, 1500));

    if (currentStep === 'industry') {
      setUserData(prev => ({ ...prev, industry: option }));
      setCurrentStep('audience');
      setOptions(audiences);
      setOptionPositions(generateOptionPositions(audiences));
      setSelectedOption(null);
      setIsSelecting(false);
    } else if (currentStep === 'audience') {
      setUserData(prev => ({ ...prev, audience: option }));
      setCurrentStep('contentTopic');
      setShowOptions(false);
      setSelectedOption(null);
      setIsSelecting(false);
    }
  };

  const handleConfirmation = (confirmed: boolean) => {
    if (confirmed) {
      // Instead of completing, transition to calibration
      setCurrentStep('calibration');
      setShowOptions(false);
      
      // Initialize calibration
      const initializeCalibration = async () => {
        try {
          const response = await fetch('http://localhost:8000/calibration/start', {
            method: 'POST'
          });
          
          if (response.ok) {
            const data = await response.json();
            setCalibrationState(prev => ({ ...prev, sessionId: data.session_id }));
            
            // Add initial calibration message to chat
            const initialMessage = getCalibrationMessage(1, 'BROAD_EXPLOR');
            setChatMessages([{ text: initialMessage, isUser: false, timestamp: new Date().toISOString() }]);
            
            await loadRoundPosts();
          } else {
            throw new Error('Failed to start calibration');
          }
        } catch (error) {
          console.error('Error starting calibration:', error);
          toast.error('Failed to start calibration. Using demo mode.');
          
          // Add initial calibration message to chat
          const initialMessage = getCalibrationMessage(1, 'BROAD_EXPLOR');
          setChatMessages([{ text: initialMessage, isUser: false, timestamp: new Date().toISOString() }]);
          
          await loadRoundPosts();
        }
      };

      initializeCalibration();
    } else {
      setCurrentStep('correction');
      setShowOptions(true);
      setOptions(['Name', 'Email', 'Role', 'Industry', 'Primary Audience', 'Content Topics', 'Content Goals', 'Ideal Customer Profile']);
      setOptionPositions(generateOptionPositions(['Name', 'Email', 'Role', 'Industry', 'Primary Audience', 'Content Topics', 'Content Goals', 'Ideal Customer Profile']));
    }
  };

  const handleCorrection = async (field: string) => {
    if (isSelecting) return;
    
    setIsSelecting(true);
    setSelectedOption(field);

    // Wait for the selection animation to complete
    await new Promise(resolve => setTimeout(resolve, 1500));

    setShowOptions(false);
    setSelectedOption(null);
    setIsSelecting(false);
    
    switch (field) {
      case 'Name':
        setCurrentStep('name');
        break;
      case 'Email':
        setCurrentStep('email');
        break;
      case 'Role':
        setCurrentStep('role');
        break;
      case 'Industry':
        setCurrentStep('industry');
        setShowOptions(true);
        setOptions(industries);
        setOptionPositions(generateOptionPositions(industries));
        break;
      case 'Primary Audience':
        setCurrentStep('audience');
        setShowOptions(true);
        setOptions(audiences);
        setOptionPositions(generateOptionPositions(audiences));
        break;
      case 'Content Topics':
        setCurrentStep('contentTopic');
        break;
      case 'Content Goals':
        setCurrentStep('userGoals');
        break;
      case 'Ideal Customer Profile':
        setCurrentStep('icp');
        break;
    }
  };

  const renderConfirmationSummary = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.2, ease: "easeOut" }}
      className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 mb-6"
    >
      <h3 className="text-lg font-semibold mb-4 text-white">Your Profile:</h3>
      <div className="space-y-2 text-gray-300 mb-4">
        <p><span className="font-medium">Name:</span> {userData.name}</p>
        <p><span className="font-medium">Email:</span> {userData.email}</p>
        <p><span className="font-medium">Role:</span> {userData.role}</p>
      </div>
      
      <h3 className="text-lg font-semibold mb-4 text-white">Project Details:</h3>
      <div className="space-y-2 text-gray-300">
        <p><span className="font-medium">Ideal Customer Profile:</span> {userData.icpDescription}</p>
        <p><span className="font-medium">Industry:</span> {userData.industry}</p>
        <p><span className="font-medium">Primary Audience:</span> {userData.audience}</p>
        <p><span className="font-medium">Content Topics:</span> {userData.contentTopic}</p>
        <p><span className="font-medium">Content Goals:</span> {userData.userGoals}</p>
      </div>
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
      {/* Mystical background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
      </div>

      {/* Onboarding Phase - Centered Layout */}
      {currentStep !== 'calibration' && (
        <div className="w-full max-w-4xl relative z-10">
          {/* Main Void Interface */}
          <div className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 mb-8 min-h-[500px] flex flex-col justify-center relative overflow-hidden">
            {/* Subtle inner glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
            
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
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
                  transition={{ duration: 0.75, ease: "easeOut" }}
                >
                  {currentMessage}
                </motion.div>

                {currentStep === 'confirmation' && renderConfirmationSummary()}

                {showOptions && (
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

                    {options.map((option, index) => {
                      const position = optionPositions[index];
                      if (!position) return null;
                      
                      const isSelected = selectedOption === option;
                      const isDisintegrating = selectedOption && selectedOption !== option;

                      return (
                        <motion.button
                          key={`${option}-${currentStep}-${index}`}
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
                            y: isSelected ? 0 : Math.sin(position.angle) * position.radius + Math.sin(Date.now() * 0.001 + index) * 3
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
                          onClick={() => {
                            if (currentStep === 'correction') {
                              handleCorrection(option);
                            } else {
                              handleOptionSelect(option);
                            }
                          }}

                          className="absolute text-white px-6 py-4 rounded-2xl text-sm font-medium transition-all duration-300"
                          whileHover={{ 
                            scale: isSelecting ? 1 : 1.1,
                            boxShadow: isSelecting 
                              ? '0 0 20px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)' 
                              : '0 0 30px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
                          }}
                          whileTap={{ scale: 0.95 }}
                        >
                          {option}
                        </motion.button>
                      );
                    })}
                  </div>
                )}

                {currentStep === 'confirmation' && !showOptions && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                    className="flex gap-6 justify-center"
                  >
                    <motion.button
                      onClick={() => handleConfirmation(true)}
                      className="bg-gradient-to-r from-green-600/90 to-green-700/90 hover:from-green-500/90 hover:to-green-600/90 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-green-500/30 shadow-2xl shadow-green-500/20"
                      whileHover={{ 
                        scale: 1.05,
                        boxShadow: "0 0 30px rgba(34, 197, 94, 0.4)"
                      }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Check className="w-5 h-5" />
                      Yes, start calibration
                    </motion.button>
                    <motion.button
                      onClick={() => handleConfirmation(false)}
                      className="bg-gradient-to-r from-red-600/90 to-red-700/90 hover:from-red-500/90 hover:to-red-600/90 text-white px-8 py-4 rounded-2xl flex items-center gap-3 transition-all duration-300 backdrop-blur-sm border border-red-500/30 shadow-2xl shadow-red-500/20"
                      whileHover={{ 
                        scale: 1.05,
                        boxShadow: "0 0 30px rgba(239, 68, 68, 0.4)"
                      }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <X className="w-5 h-5" />
                      No, let me edit
                    </motion.button>
                  </motion.div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Input Area */}
          {!showOptions && currentStep !== 'confirmation' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.0, ease: "easeOut" }}
              className="flex gap-4"
            >
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleInputSubmit()}
                placeholder="Type your response..."
                className="flex-1 bg-gray-900/80 backdrop-blur-sm text-white px-6 py-4 rounded-2xl border border-gray-700/50 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all duration-300"
              />
              <motion.button
                onClick={handleInputSubmit}
                disabled={!inputValue.trim()}
                className="bg-gradient-to-r from-blue-600/90 to-blue-700/90 hover:from-blue-500/90 hover:to-blue-600/90 disabled:from-gray-600/50 disabled:to-gray-700/50 text-white px-6 py-4 rounded-2xl transition-all duration-300 backdrop-blur-sm border border-blue-500/30 shadow-2xl shadow-blue-500/20"
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: "0 0 30px rgba(59, 130, 246, 0.4)"
                }}
                whileTap={{ scale: 0.95 }}
              >
                <Send className="w-5 h-5" />
              </motion.button>
            </motion.div>
          )}
        </div>
      )}

      {/* Calibration Phase - Centered Layout (using CalibrationPage styling) */}
      {currentStep === 'calibration' && (
        <div className="w-full max-w-4xl relative z-10">
          {/* Main Void Interface */}
          <div className="bg-gradient-to-br from-gray-900/80 to-black/90 backdrop-blur-xl border border-gray-800/30 rounded-3xl p-12 mb-8 min-h-[500px] flex flex-col justify-center relative overflow-hidden">
            {/* Subtle inner glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 rounded-3xl"></div>
            
            <AnimatePresence mode="wait">
              <motion.div
                key={calibrationState.round}
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
                  <div className="text-sm text-gray-400 mb-2">Round {calibrationState.round} of 9</div>
                  <div className="w-full bg-gray-800/50 rounded-full h-2 mb-2">
                    <motion.div 
                      className="bg-blue-500/70 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${(calibrationState.round / 9) * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <div className="text-xs text-gray-500">Confidence: {Math.round(calibrationState.confidence * 100)}%</div>
                </div>

                {calibrationState.posts.length > 0 && (
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

                    {calibrationState.posts.map((post, index) => {
                      const position = getOrbitalPosition(index, calibrationState.posts.length);
                      const isSelected = calibrationState.selectedPost?.id === post.id;
                      const isDisintegrating = calibrationState.selectedPost && calibrationState.selectedPost.id !== post.id;

                      return (
                        <motion.button
                          key={`${post.id}-${calibrationState.round}-${index}`}
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
                            y: isSelected ? 0 : Math.sin(position.angle) * position.radius + Math.sin(Date.now() * 0.001 + index) * 3
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
                          className="absolute text-white px-6 py-4 rounded-2xl text-sm font-medium transition-all duration-300 max-w-xs"
                          whileHover={{ 
                            scale: isSelecting ? 1 : 1.1,
                            boxShadow: isSelecting 
                              ? '0 0 20px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)' 
                              : '0 0 30px rgba(59, 130, 246, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
                          }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <div className="text-left">
                            <div className="text-xs font-medium text-blue-300 mb-2 uppercase tracking-wide">
                              {post.type === 'text' ? '📝 TEXT' : '🎥 VIDEO'}
                            </div>
                            <div className="text-sm leading-relaxed overflow-hidden line-clamp-4">
                              {post.content}
                            </div>
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                )}

                {isCalibrationLoading && (
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
      )}
    </div>
  );
};

export default ConversationalOnboarding;
