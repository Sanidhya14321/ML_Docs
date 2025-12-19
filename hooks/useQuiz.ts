
import { useState, useCallback } from 'react';
import { QuizConfig } from '../types';

export const useQuiz = (config?: QuizConfig) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [answerStatus, setAnswerStatus] = useState<'correct' | 'incorrect' | null>(null);

  const questions = config?.questions || [];
  const currentQuestion = questions[currentIndex];
  const progress = Math.round(((currentIndex) / questions.length) * 100);

  const selectOption = useCallback((index: number) => {
    if (selectedOption !== null || !currentQuestion) return; // Prevent multi-select

    setSelectedOption(index);
    const isCorrect = index === currentQuestion.correctIndex;

    if (isCorrect) {
      setScore(prev => prev + 1);
      setAnswerStatus('correct');
    } else {
      setAnswerStatus('incorrect');
    }
  }, [currentQuestion, selectedOption]);

  const nextQuestion = useCallback(() => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex(prev => prev + 1);
      setSelectedOption(null);
      setAnswerStatus(null);
    } else {
      setIsFinished(true);
    }
  }, [currentIndex, questions.length]);

  const resetQuiz = useCallback(() => {
    setCurrentIndex(0);
    setSelectedOption(null);
    setScore(0);
    setIsFinished(false);
    setAnswerStatus(null);
  }, []);

  return {
    currentQuestion,
    currentIndex,
    totalQuestions: questions.length,
    selectedOption,
    selectOption,
    nextQuestion,
    score,
    isFinished,
    answerStatus,
    progress,
    resetQuiz
  };
};
