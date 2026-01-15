
import { useState, useCallback, useEffect } from 'react';

export type LogType = 'stdout' | 'error' | 'system';

export interface LogEntry {
  id: string;
  type: LogType;
  content: string;
  timestamp: number;
}

export const useCodeRunner = (topicId: string, initialCode: string) => {
  // Initialize state from localStorage or initialCode
  const [code, setCodeState] = useState<string>(() => {
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem(`ai-codex-lab-${topicId}`);
        if (saved !== null) return saved;
    }
    return initialCode;
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  // Sync state when topicId or initialCode changes (handles navigation without remount if applicable)
  useEffect(() => {
    const saved = localStorage.getItem(`ai-codex-lab-${topicId}`);
    if (saved !== null) {
      setCodeState(saved);
    } else {
      setCodeState(initialCode);
    }
    setLogs([]); // Clear logs when switching topics
  }, [topicId, initialCode]);

  const setCode = useCallback((newCode: string) => {
    setCodeState(newCode);
    localStorage.setItem(`ai-codex-lab-${topicId}`, newCode);
  }, [topicId]);

  const resetCode = useCallback(() => {
    setCodeState(initialCode);
    localStorage.removeItem(`ai-codex-lab-${topicId}`);
    setLogs([]);
  }, [topicId, initialCode]);

  const addLog = (type: LogType, content: string) => {
    setLogs(prev => [...prev, {
      id: Math.random().toString(36).substr(2, 9),
      type,
      content,
      timestamp: Date.now()
    }]);
  };

  const clearLogs = useCallback(() => setLogs([]), []);

  const runCode = useCallback(async () => {
    setIsRunning(true);
    clearLogs();
    
    // Initial System Log
    addLog('system', 'Initializing Python 3.9 runtime environment...');

    // Simulate Network/Processing Delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Simple Heuristic Validation Simulation
    const hasPrint = code.includes('print');
    const hasDef = code.includes('def');
    const hasImport = code.includes('import');
    const isEmpty = code.trim().length === 0;

    if (isEmpty) {
        addLog('error', 'Error: No code to execute.');
    } else if (code.includes('error') || code.includes('raise')) {
        // Simulate a runtime error if user types "error"
        addLog('error', 'Traceback (most recent call last):');
        addLog('error', '  File "script.py", line 4, in <module>');
        addLog('error', 'RuntimeError: Forced exception for testing');
    } else {
        // Simulate execution output
        if (hasImport) {
            addLog('stdout', '[INFO] Libraries loaded successfully');
        }
        
        if (hasDef) {
            addLog('stdout', '[INFO] Functions defined in memory');
        }

        if (hasPrint) {
            // Try to extract content inside print() for a bit of realism
            const printMatch = code.match(/print\s*\((.*?)\)/);
            if (printMatch && printMatch[1]) {
                // Strip quotes
                const output = printMatch[1].replace(/["']/g, '');
                addLog('stdout', output);
            } else {
                addLog('stdout', 'Output generated successfully.');
            }
        } else {
            addLog('stdout', 'Process finished with exit code 0');
        }

        // Final Success Message
        addLog('system', `Execution completed in ${(Math.random() * 0.5 + 0.1).toFixed(3)}s`);
    }

    setIsRunning(false);
  }, [code, clearLogs]);

  return { code, setCode, resetCode, logs, isRunning, runCode, clearLogs };
};
