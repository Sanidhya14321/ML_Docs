
import React from 'react';
import { motion } from 'framer-motion';
import { MOTION_VARIANTS } from '../constants';
import { 
  TrendingUp, 
  HeartPulse, 
  ShoppingBag, 
  Factory, 
  ShieldCheck, 
  Car, 
  Smartphone,
  Globe
} from 'lucide-react';

const UseCaseCard = ({ icon: Icon, title, industry, description, impact, color }: any) => (
    <motion.div 
        variants={MOTION_VARIANTS.item}
        className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:border-brand/30 transition-all group"
    >
        <div className={`w-12 h-12 rounded-2xl ${color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
            <Icon size={24} className="text-white" />
        </div>
        <div className="text-[10px] font-mono text-brand uppercase tracking-[0.2em] mb-2">{industry}</div>
        <h3 className="text-2xl font-bold text-white mb-4">{title}</h3>
        <p className="text-slate-400 text-sm leading-relaxed mb-6">{description}</p>
        <div className="pt-6 border-t border-slate-800">
            <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Measurable Impact</div>
            <div className="text-lg font-bold text-emerald-400">{impact}</div>
        </div>
    </motion.div>
);

export const IndustryUseCasesView: React.FC = () => {
  return (
    <motion.div 
      variants={MOTION_VARIANTS.container}
      initial="hidden"
      animate="show"
      className="space-y-24 pb-20"
    >
      <motion.header variants={MOTION_VARIANTS.item} className="border-b border-slate-800 pb-12">
        <h1 className="text-6xl font-serif font-bold text-white mb-6">Industry Use Cases</h1>
        <p className="text-slate-400 text-xl max-w-2xl leading-relaxed font-light">
          Exploring how machine learning architectures translate into tangible business value and societal impact across global sectors.
        </p>
      </motion.header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <UseCaseCard 
            icon={ShieldCheck}
            industry="Finance"
            title="Real-time Fraud Detection"
            description="Using anomaly detection and ensemble models to identify fraudulent transactions in milliseconds, protecting billions in assets."
            impact="95% Reduction in False Positives"
            color="bg-blue-500"
        />
        <UseCaseCard 
            icon={HeartPulse}
            industry="Healthcare"
            title="Early Cancer Diagnosis"
            description="Deep learning models (CNNs) analyzing radiological images to detect malignant tumors earlier than traditional methods."
            impact="30% Improvement in Survival Rates"
            color="bg-rose-500"
        />
        <UseCaseCard 
            icon={ShoppingBag}
            industry="Retail"
            title="Personalized Recommendations"
            description="Collaborative filtering and deep retrieval systems that predict user preferences to drive engagement and sales."
            impact="40% Increase in Conversion"
            color="bg-emerald-500"
        />
        <UseCaseCard 
            icon={Factory}
            industry="Manufacturing"
            title="Predictive Maintenance"
            description="IoT sensor data processed by LSTM networks to predict equipment failure before it happens, minimizing downtime."
            impact="$1.2M Annual Savings per Plant"
            color="bg-amber-500"
        />
        <UseCaseCard 
            icon={Car}
            industry="Automotive"
            title="Autonomous Navigation"
            description="Computer vision and reinforcement learning enabling vehicles to perceive surroundings and make safe driving decisions."
            impact="Level 4 Autonomy Achieved"
            color="bg-indigo-500"
        />
        <UseCaseCard 
            icon={Smartphone}
            industry="Tech"
            title="On-device LLMs"
            description="Quantized transformer models running locally on mobile devices for privacy-preserving intelligent assistants."
            impact="Zero-latency AI Interaction"
            color="bg-purple-500"
        />
      </div>

      <motion.section variants={MOTION_VARIANTS.item} className="bg-slate-900/30 border border-slate-800 rounded-3xl p-12 text-center">
        <Globe className="mx-auto mb-6 text-brand" size={48} />
        <h2 className="text-3xl font-bold text-white mb-4">The Global AI Landscape</h2>
        <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Machine learning is no longer a research curiosity; it is the primary driver of the Fourth Industrial Revolution. 
            As models become more efficient and data more accessible, the barrier to entry continues to drop, enabling 
            innovation in every corner of the globe.
        </p>
        <button 
            onClick={() => window.location.hash = '#/dashboard'}
            className="mt-10 px-8 py-3 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-xl transition-all"
        >
            Return to Dashboard
        </button>
      </motion.section>
    </motion.div>
  );
};
