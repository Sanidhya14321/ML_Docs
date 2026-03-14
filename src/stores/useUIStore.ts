import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  isSidebarOpen: boolean;
  isSearchOpen: boolean;
  isSettingsOpen: boolean;
  theme: 'light' | 'dark';
  
  toggleSidebar: () => void;
  setSidebarOpen: (isOpen: boolean) => void;
  toggleSearch: () => void;
  setSearchOpen: (isOpen: boolean) => void;
  toggleSettings: () => void;
  setSettingsOpen: (isOpen: boolean) => void;
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      isSidebarOpen: true,
      isSearchOpen: false,
      isSettingsOpen: false,
      theme: (typeof window !== 'undefined' && localStorage.getItem('ai-codex-theme') as 'light' | 'dark') || 'dark',

      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      setSidebarOpen: (isOpen) => set({ isSidebarOpen: isOpen }),
      toggleSearch: () => set((state) => ({ isSearchOpen: !state.isSearchOpen })),
      setSearchOpen: (isOpen) => set({ isSearchOpen: isOpen }),
      toggleSettings: () => set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),
      setSettingsOpen: (isOpen) => set({ isSettingsOpen: isOpen }),
      toggleTheme: () => set((state) => {
        const newTheme = state.theme === 'light' ? 'dark' : 'light';
        if (typeof document !== 'undefined') {
          if (newTheme === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
          localStorage.setItem('ai-codex-theme', newTheme);
        }
        return { theme: newTheme };
      }),
      setTheme: (theme) => set(() => {
        if (typeof document !== 'undefined') {
          if (theme === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
          localStorage.setItem('ai-codex-theme', theme);
        }
        return { theme };
      }),
    }),
    {
      name: 'ai-codex-ui-storage',
      partialize: (state) => ({ theme: state.theme, isSidebarOpen: state.isSidebarOpen }),
    }
  )
);
