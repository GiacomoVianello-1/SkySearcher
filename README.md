<div align="center">

# SkySearcher: High-Altitude Onboard Vision-Language Reasoning for Aerial Query Localization

<p><strong>Code coming soon.</strong></p>

</div>

## 📄 Abstract
Recent advances in **Vision-Language Models** (VLMs) have improved semantic and visual reasoning, enabling UAVs to search for open-vocabulary queries from user instructions in unstructured environments at high altitude. However, existing systems mainly follow low-altitude strategies, focusing on close-range iterative exploration, requiring storage-intensive dense 3D metric-semantic reconstructions or relying on large cloud-serviced VLMs assuming constant internet access. We propose **SkySearcher**, an aerial semantic search navigation method designed for large outdoor scenes. **SkySearcher** integrates onboard VLM reasoning with a lightweight probabilistic representation that guides an information-driven exploration. The VLM extracts semantic information at high altitude about the target or cues informing about its location. This information is projected to the ground using the camera frustum instead of limited-depth sensors, actively exploiting the expansive field of view offered by elevated sensing. The information regarding the presence or absence of semantic cues is then integrated into a probabilistic map by modeling the spatial likelihood of the target, maintaining a task-driven memory for the mission and guiding the exploration towards semantically promising regions. Extensive simulations for Aerial Object Goal Navigation on public benchmarks demonstrate that **SkySearcher** significantly outperforms state-of-the-art baselines in success rate, flight time, and navigation precision, while a deployment of **SkySearcher** running fully onboard a drone demonstrates the applicability of our approach.
