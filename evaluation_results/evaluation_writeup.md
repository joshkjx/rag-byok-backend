## Evaluation

### Methodology

Conducted two-phase evaluation on a 37-question multi-domain test corpus to optimize RAG configuration for cost-performance tradeoff:

**Phase 1: Retrieval Validation**
- Tested 22 configurations across k-values (3, 4, 5, 7), chunk sizes (256, 512, 1024), and search strategies (similarity, MMR)
- Measured precision and recall with manually labeled ground truth documents
- Found uniformly high retrieval success (~95%) but limited discriminative power for single-document queries
- Seeing that retrieval metrics alone were uninformative, decided to explore end-to-end evaluation to understand real performance differences

**Phase 2: Manual Answer Quality Assessment**
- Evaluated answer quality across 4 dimensions: accuracy, completeness, conciseness, groundedness
    - **Accuracy**: If all statements in the output were factually correct. A simple expected answer was used as a guideline for accuracy.
    - **Completeness**: For each question, I wrote down key points on a notepad and checked to see if all key points were covered. If any key points were missing, the response was marked as incomplete.
    - **Conciseness**: For each question, if there were 2 or more sentences which were not directly relevant to the question, I marked it as not concise.
    - **Groundedness**: For each response, if it referenced anything that was not explicitly in the provided documents, I counted it as a hallucination.
- Manually assessed all 37 questions for top 2 configurations
- Measured actual token usage and cost implications

### Results

| Configuration | Accuracy | Groundedness | Conciseness | Completeness | Avg Tokens/Query | Cost Reduction |
|---------------|----------|--------------|-------------|--------------|------------------|----------------|
| k=4 (baseline) | 92% | 97% | 81% | 92% | 873 | - |
| k=3 (optimized) | 89% | 100% | 97% | 89% | 732 | 16% |

### Key Findings

**Cost-Performance Tradeoff**: The optimized configuration (k=3) achieves:
- 16% reduction in token usage
- 100% groundedness
- Significantly improved conciseness
- Minor accuracy trade-off

**Domain-Specific Performance**: 
- k=3 performed better when it came to narrative content (e.g., short stories)
- k=4 generally provided more comprehensive factual detail (e.g., news articles, technical documentation)

**Evaluation Insights**: 
- Standard retrieval metrics (precision/recall) had limited discriminative power for single-document queries within a multi-domain corpus
- End-to-end answer quality evaluation revealed meaningful differences not captured by retrieval metrics
- Optimal configuration depends on use case priorities: cost-efficiency vs. comprehensive detail

### Limitations

- Test corpus represents diverse document types but may not cover all real-world scenarios
- Manual evaluation on 37 questions provides strong signal for trend identification, though larger test set would increase statistical confidence
- Manual evaluation introduces potential for evaluator bias; inter-rater reliability testing with multiple evaluators would strengthen validity