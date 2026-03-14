import React from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { useUIStore } from '../../stores/useUIStore';
import { Moon, Sun, Monitor, Bell, Shield, User, LogOut } from 'lucide-react';
import { cn } from '../../lib/utils';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const { theme, setTheme } = useUIStore();

  const sections = [
    {
      id: 'appearance',
      title: 'Appearance',
      icon: <Sun size={18} />,
      content: (
        <div className="space-y-4">
          <p className="text-sm text-text-secondary">Customize how AI Codex looks on your device.</p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { id: 'light', label: 'Light', icon: <Sun size={20} /> },
              { id: 'dark', label: 'Dark', icon: <Moon size={20} /> },
              { id: 'system', label: 'System', icon: <Monitor size={20} /> },
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id as any)}
                className={cn(
                  'flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all',
                  theme === t.id 
                    ? 'bg-brand/5 border-brand text-brand shadow-sm' 
                    : 'bg-surface border-border-subtle text-text-secondary hover:border-brand/30'
                )}
              >
                {t.icon}
                <span className="text-xs font-bold">{t.label}</span>
              </button>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'notifications',
      title: 'Notifications',
      icon: <Bell size={18} />,
      content: (
        <div className="space-y-4">
          <p className="text-sm text-text-secondary">Manage your learning reminders and updates.</p>
          <Button variant="outline" className="w-full justify-start gap-3">
            <Bell size={16} />
            <span>Enable Push Notifications</span>
          </Button>
        </div>
      )
    },
    {
      id: 'account',
      title: 'Account',
      icon: <User size={18} />,
      content: (
        <div className="space-y-4">
          <div className="flex items-center gap-4 p-4 rounded-xl bg-surface-hover border border-border-subtle">
             <div className="w-12 h-12 rounded-full bg-brand/10 flex items-center justify-center text-brand font-bold">
                JD
             </div>
             <div>
                <h4 className="font-bold text-text-primary">John Doe</h4>
                <p className="text-xs text-text-muted">john.doe@example.com</p>
             </div>
          </div>
          <Button variant="destructive" className="w-full gap-2">
            <LogOut size={16} />
            <span>Sign Out</span>
          </Button>
        </div>
      )
    }
  ];

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Settings"
      description="Manage your preferences and account settings."
      size="lg"
    >
      <div className="space-y-8 py-4">
        {sections.map((section) => (
          <div key={section.id} className="space-y-4">
            <div className="flex items-center gap-2 text-text-primary">
              <span className="text-brand">{section.icon}</span>
              <h3 className="font-bold tracking-tight">{section.title}</h3>
            </div>
            {section.content}
            <div className="h-px bg-border-subtle last:hidden" />
          </div>
        ))}
      </div>
    </Modal>
  );
};
