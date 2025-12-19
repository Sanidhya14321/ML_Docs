
import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, XCircle, ArrowRight, RotateCcw, Award, AlertCircle } from 'lucide-react';
import { getTopicById } from '../lib/contentHelpers';
import { useQuiz } from '../hooks/useQuiz';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { LoadingOverlay } from './LoadingOverlay';

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

  useEffect(() => {
    if (isFinished) {
      markAsCompleted(topicId);
    }
  }, [isFinished, topicId, markAsCompleted]);

  if (!topic || !topic.quizConfig) return <LoadingOverlay />;

  const passingScore = topic.quizConfig.passingScore || 70;
  const percentage = Math.round((score / totalQuestions) * 100);
  const passed = percentage >= passingScore;

  if (isFinished) {
    return (
      <div className="min-h-[80vh] flex items-center justify-center p-6">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-slate-900 border border-slate-800 rounded-3xl p-12 max-w-lg w-full text-center relative overflow-hidden"
        >
          <div className={`absolute top-0 left-0 w-full h-2 ${passed ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          
          <div className={`mx-auto w-24 h-24 rounded-full flex items-center justify-center mb-6 ${passed ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
             {passed ? <Award size={48} /> : <AlertCircle size={48} />}
          </div>

          <h2 className="text-3xl font-serif font-bold text-white mb-2">
            {passed ? 'Assessment Passed!' : 'Needs Improvement'}
          </h2>
          <p className="text-slate-400 mb-8">
            You scored <strong className={passed ? 'text-white' : 'text-rose-400'}>{percentage}%</strong> on {topic.title}.
          </p>

          <div className="flex gap-4 justify-center">
            <button 
              onClick={resetQuiz}
              className="px-6 py-3 rounded-xl border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800 transition-colors flex items-center gap-2 font-bold text-sm"
            >
              <RotateCcw size={16} /> Retry
            </button>
            <button 
              onClick={onComplete}
              className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 transition-all flex items-center gap-2 font-bold text-sm"
            >
              Continue <ArrowRight size={16} />
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto pt-12 pb-24 px-6">
      <header className="mb-12">
         <div className="flex justify-between items-end mb-4">
            <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400">Assessment</span>
            <span className="text-xs font-mono text-slate-500">Question {currentIndex + 1} of {totalQuestions}</span>
         </div>
         <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
            <motion.div 
              className="h-full bg-indigo-500" 
              initial={{ width: 0 }}
              animate={{ width: `${((currentIndex + 1) / totalQuestions) * 100}%` }}
            />
         </div>
      </header>

      <div className="space-y-8">
        <h1 className="text-2xl md:text-3xl font-bold text-white leading-tight">
          {currentQuestion.text}
        </h1>

        <div className="grid grid-cols-1 gap-4">
          {currentQuestion.options.map((option, idx) => {
            let statusClass = "bg-slate-900 border-slate-800 hover:border-slate-600 text-slate-300";
            let icon = null;

            if (selectedOption !== null) {
               if (idx === currentQuestion.correctIndex) {
                 statusClass = "bg-emerald-500/10 border-emerald-500/50 text-emerald-300";
                 icon = <CheckCircle size={18} />;
               } else if (idx === selectedOption) {
                 statusClass = "bg-rose-500/10 border-rose-500/50 text-rose-300";
                 icon = <XCircle size={18} />;
               } else {
                 statusClass = "opacity-50 bg-slate-900 border-slate-800";
               }
            }

            return (
              <button
                key={idx}
                disabled={selectedOption !== null}
                onClick={() => selectOption(idx)}
                className={`w-full text-left p-6 rounded-2xl border-2 transition-all duration-200 flex items-center justify-between group ${statusClass} ${selectedOption === null ? 'hover:scale-[1.01]' : ''}`}
              >
                <span className="text-lg font-medium">{option}</span>
                {icon}
              </button>
            );
          })}
        </div>

        <AnimatePresence>
          {selectedOption !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 p-6 rounded-2xl bg-slate-900/50 border border-slate-800"
            >
              <div className="flex items-start gap-3">
                 <div className={`mt-1 ${answerStatus === 'correct' ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {answerStatus === 'correct' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
                 </div>
                 <div>
                    <h4 className={`font-bold text-sm mb-1 ${answerStatus === 'correct' ? 'text-emerald-400' : 'text-rose-400'}`}>
                       {answerStatus === 'correct' ? 'Correct!' : 'Incorrect'}
                    </h4>
                    <p className="text-slate-400 text-sm leading-relaxed">
                       {currentQuestion.explanation}
                    </p>
                 </div>
                 
                 <button 
                   onClick={nextQuestion}
                   className="ml-auto px-6 py-2 bg-white text-slate-950 rounded-lg font-bold text-sm hover:bg-slate-200 transition-colors"
                 >
                   {currentIndex === totalQuestions - 1 ? 'Finish' : 'Next'}
                 </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
