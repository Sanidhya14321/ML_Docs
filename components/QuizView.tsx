
import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, XCircle, ArrowRight, RotateCcw, Award, AlertCircle } from 'lucide-react';
import { getTopicById } from '../lib/contentHelpers';
import { useQuiz } from '../hooks/useQuiz';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { LoadingOverlay } from './LoadingOverlay';
import { triggerConfetti } from './Confetti';

interface QuizViewProps {
  topicId: string;
  onBack: () => void;
  onComplete: () => void;
}

export const QuizView: React.FC<QuizViewProps> = ({ topicId, onBack, onComplete }) => {
  const topic = getTopicById(topicId);
  const { 
    currentQuestion, 
    currentIndex, 
    totalQuestions, 
    selectedOption, 
    selectOption, 
    nextQuestion, 
    score, 
    isFinished, 
    answerStatus,
    progress,
    resetQuiz
  } = useQuiz(topic?.quizConfig);

  const { markAsCompleted } = useCourseProgress();

  if (!topic || !topic.quizConfig) return <LoadingOverlay />;

  const passingScore = topic.quizConfig.passingScore || 70;
  const percentage = Math.round((score / totalQuestions) * 100);
  const passed = percentage >= passingScore;

  useEffect(() => {
    if (isFinished) {
      if (passed) {
        markAsCompleted(topicId);
        triggerConfetti();
      }
    }
  }, [isFinished, passed, topicId, markAsCompleted]);

  if (isFinished) {
    return (
      <div className="min-h-[80vh] flex items-center justify-center p-6">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-surface border border-border-strong rounded-none p-12 max-w-lg w-full text-center relative overflow-hidden shadow-sm"
        >
          <div className={`absolute top-0 left-0 w-full h-1 ${passed ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          
          <div className={`mx-auto w-20 h-20 rounded-none border flex items-center justify-center mb-8 ${passed ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-500' : 'bg-rose-500/5 border-rose-500/20 text-rose-500'}`}>
             {passed ? <Award size={40} /> : <AlertCircle size={40} />}
          </div>

          <h2 className="text-3xl font-heading font-black text-text-primary mb-3 uppercase tracking-tight">
            {passed ? 'ASSESSMENT_PASSED' : 'NEEDS_OPTIMIZATION'}
          </h2>
          <p className="text-text-secondary font-light mb-10">
            System synchronization at <strong className={passed ? 'text-text-primary font-bold' : 'text-rose-500 font-bold'}>{percentage}%</strong> for {topic.title}.
          </p>

          <div className="flex gap-4 justify-center">
            <button 
              onClick={resetQuiz}
              className="px-8 py-3 bg-surface border border-border-strong text-[11px] font-mono font-black uppercase tracking-[0.2em] text-text-secondary hover:text-text-primary hover:border-text-primary transition-all"
            >
              <RotateCcw size={14} className="inline mr-2" /> RETRY_SEQUENCE
            </button>
            <button 
              onClick={onComplete}
              className="px-8 py-3 bg-text-primary text-app hover:bg-brand text-[11px] font-mono font-black uppercase tracking-[0.2em] transition-all"
            >
              CONTINUE_PROTOCOL <ArrowRight size={14} className="inline ml-2" />
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-3xl mx-auto pt-16 pb-24 px-6"
    >
      <header className="mb-16">
         <div className="flex justify-between items-end mb-6">
            <span className="text-[10px] font-mono font-black uppercase tracking-[0.3em] text-brand">NEURAL_ASSESSMENT_PROTOCOL</span>
            <span className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">NODE {currentIndex + 1} // {totalQuestions}</span>
         </div>
         <div className="w-full h-px bg-border-strong relative overflow-hidden">
            <motion.div 
              className="h-full bg-brand" 
              initial={{ width: 0 }}
              animate={{ width: `${((currentIndex + 1) / totalQuestions) * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
         </div>
      </header>

      <div className="space-y-12">
        <motion.h1 
          key={currentQuestion.text}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl md:text-4xl font-heading font-black text-text-primary leading-tight uppercase tracking-tight"
        >
          {currentQuestion.text}
        </motion.h1>

        <div className="grid grid-cols-1 gap-3">
          {currentQuestion.options.map((option, idx) => {
            let statusClass = "bg-surface border-border-strong text-text-secondary hover:border-brand hover:text-text-primary";
            let icon = null;

            if (selectedOption !== null) {
               if (idx === currentQuestion.correctIndex) {
                 statusClass = "bg-emerald-500/5 border-emerald-500/50 text-emerald-500";
                 icon = <CheckCircle size={16} />;
               } else if (idx === selectedOption) {
                 statusClass = "bg-rose-500/5 border-rose-500/50 text-rose-500";
                 icon = <XCircle size={16} />;
               } else {
                 statusClass = "opacity-30 bg-app border-border-subtle";
               }
            }

            return (
              <button
                key={idx}
                disabled={selectedOption !== null}
                onClick={() => selectOption(idx)}
                className={`w-full text-left p-6 border transition-all duration-200 flex items-center justify-between group rounded-none ${statusClass}`}
              >
                <span className="text-[13px] font-mono font-bold uppercase tracking-tight">{option}</span>
                {icon}
              </button>
            );
          })}
        </div>

        <AnimatePresence>
          {selectedOption !== null && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12 p-8 bg-surface border border-border-strong relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-1 h-full bg-brand" />
              <div className="flex items-start gap-6">
                 <div className={`mt-1 ${answerStatus === 'correct' ? 'text-emerald-500' : 'text-rose-500'}`}>
                    {answerStatus === 'correct' ? <CheckCircle size={24} /> : <AlertCircle size={24} />}
                 </div>
                 <div className="flex-1">
                    <h4 className={`text-[11px] font-mono font-black uppercase tracking-[0.2em] mb-3 ${answerStatus === 'correct' ? 'text-emerald-500' : 'text-rose-500'}`}>
                       {answerStatus === 'correct' ? 'VALID_RESPONSE' : 'INVALID_INPUT'}
                    </h4>
                    <p className="text-text-secondary text-sm leading-relaxed font-light italic">
                       {currentQuestion.explanation}
                    </p>
                 </div>
                 
                 <button 
                   onClick={nextQuestion}
                   className="px-8 py-3 bg-text-primary text-app hover:bg-brand font-mono font-black text-[11px] uppercase tracking-[0.2em] transition-all"
                 >
                   {currentIndex === totalQuestions - 1 ? 'FINISH_PROTOCOL' : 'NEXT_NODE'}
                 </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};
