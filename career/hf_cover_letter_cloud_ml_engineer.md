# Cover Letter - Hugging Face Cloud ML Engineer

Dear Hiring Team,

I am applying for the Cloud Machine Learning Engineer role because my work sits directly at the intersection of the responsibilities you care about: cloud deployment, ML systems performance, robust developer experience, and production-quality tooling.

In my current project, `nsys-llm-explainer`, I built an offline Nsight Systems SQLite analyzer for LLM inference traces. It turns raw trace exports into prioritized, evidence-backed findings across CUDA kernels, CPU-GPU barriers, launch storms, NCCL overlap, NVLink correlation, stream overlap, launch latency, roofline metrics, and per-process breakdowns. The output is deliberately conservative and auditable: when metrics are missing, the tool warns explicitly instead of inventing a result.

That design maps well to Hugging Face ML Cloud work. I care about making advanced systems usable in practice, not just technically possible. I have experience packaging code as a reproducible Python project, exposing a CLI and dashboard workflow, writing capture and export guidance, and structuring outputs so they can be consumed by humans and automation alike. That is the kind of work that helps users move from a model running locally to a reliable cloud workflow.

What I would bring to Hugging Face is a combination of implementation depth and product thinking:

- Cloud-ready deployment mindset with containerized, reproducible workflows.
- Strong focus on performance and measurement, especially for GPU-heavy inference.
- Developer experience work that reduces friction for users adopting complex systems.
- Documentation and example-driven communication that helps teams and partners ship faster.

I would especially want to contribute to integrations that make Hugging Face models and libraries easier to run efficiently on cloud platforms and managed services. My background makes me comfortable working from trace-level evidence up to user-facing guidance, which is useful when the goal is to improve both performance and usability.

If helpful, I would present `nsys-llm-explainer` as a live Hugging Face Space titled `nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference` and show how the same codebase can support a CLI, dashboard, and shareable analysis artifact. That is the kind of practical, reusable demo I would want to build for ML Cloud users.

Thank you for your consideration.

Sincerely,

Fahad Alghanim
