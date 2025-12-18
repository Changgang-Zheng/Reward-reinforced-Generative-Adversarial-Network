# History-awareness-Self-adaptive-System-on-Airborne-Base-Station


This repository contains source code from an early-stage undergraduate research project ("Reward-Reinforced Generative Adversarial Networks for Multi-agent Systems") that explored reward modeling for multi-agent reinforcement learning in a custom airborne base-station environment.

At the time of this study, commonly used baseline implementations (e.g., vanilla DQN without specialized reward scaling) exhibited unstable or slow convergence in this environment. This observation motivated the development of the RR-GAN approach, which aimed to improve learning stability by modeling a global reward structure.

Subsequent work and later understanding showed that applying appropriate reward normalization, a well-established best practice in reinforcement learning, can significantly improve the stability of standard baselines, allowing them to reach comparable optimal performance in this environment.

We emphasize that:

- The associated paper focuses on learning behavior and stability trends, rather than absolute benchmark scores
- Experimental results reflect the configurations and practices used during the study period
- Visualization scripts include standard post-processing (e.g., smoothing) intended solely to aid qualitative interpretation of trends
- The project represents an exploratory learning effort, highlighting the importance of rigorous baseline implementation and fair comparison in later-stage research

This repository is preserved as an archival snapshot for transparency and historical reference. It is not actively maintained, and the included implementations reflect the practices at the time of the study rather than current standards.

