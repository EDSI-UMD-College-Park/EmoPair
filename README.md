# EMOPAIR

A research project for improving emotional dimension annotations in the EmoBank dataset through pairwise comparison methodologies and validated Large Language Model (LLM) annotations.

## Overview

EmoPair enhances the [EmoBank dataset](https://github.com/JULIELab/EmoBank) by leveraging pairwise annotation techniques combined with LLM-based annotations that have been validated against trained human annotators. The project focuses on three key emotional dimensions: **Valence** (pleasantness), **Arousal** (intensity), and **Dominance** (control).

## Background

EmoBank is a widely-used corpus containing ~10,000 English sentences annotated for dimensional emotion (Valence, Arousal, and Dominance). This project improves upon the original annotations by:

1. Using pairwise comparisons instead of absolute ratings
2. Validating LLM performance through the Alt-Test methodology
3. Combining human annotations with validated LLM annotations to scale the annotation process

## Methodology

### Pairwise Annotation Approach

Instead of rating each text on an absolute scale, annotators compare pairs of texts and determine which expresses higher levels of each emotional dimension. This approach:

- Reduces cognitive load on annotators
- Improves consistency and reliability
- Enables more nuanced emotional distinctions

### LLM Validation (Alt-Test)

LLMs are validated against trained human annotators using the Alt-Test to ensure annotation quality. The project tested multiple models:

- **GPT-5.1**
- **GPT-5-nano**
- **Claude-Haiku-4.5**
- **Claude-Sonnet-4.5**

### Chain-Guided Chain-of-Thought (CGCOT) Prompts

The project uses structured CGCOT prompts to guide LLM reasoning for each emotional dimension, ensuring systematic and reliable annotations.

## Repository Structure

```
EmoPair/
├── Data/
│   ├── emobank.csv                      # Original EmoBank dataset
│   ├── annotation_sample.csv            # Sample for annotation
│   ├── training_sample.xlsx             # Training data for annotators
│   ├── bin_combinations.csv             # Bin combination configurations
│   ├── annotator map.txt                # Human annotator mappings
│   ├── cgcot_prompts/                   # Chain-of-thought prompts
│   │   ├── valence.txt                  # Valence prompting strategy
│   │   ├── arousal.txt                  # Arousal prompting strategy
│   │   └── dominance.txt                # Dominance prompting strategy
│   └── human annotations/               # Human annotation data
│       └── Completed/                   # Completed annotation files
├── Results/
│   ├── pairadigms/                      # Pairadigm model objects
│   └── paired_results/                  # Pairwise comparison results
│       ├── valence_paired_results.csv
│       ├── arousal_paired_results.csv
│       └── dominance_paired_results.csv
├── pull sample.ipynb                    # Sample data extraction
├── create pairs.ipynb                   # Generate pairwise comparisons
├── training sample.ipynb                # Prepare training samples
├── validate_LLMs.ipynb                  # LLM validation and Alt-Test
├── build_emopair.ipynb                  # Build final EmoPair dataset
└── EMOPAIR Codebook.docx               # Annotation guidelines
```

## Notebooks

### 1. `pull sample.ipynb`
Downloads and extracts samples from the EmoBank dataset for annotation.

### 2. `create pairs.ipynb`
Generates pairwise comparisons using the `pairadigm` library. Creates structured pairs for human and LLM annotation across all three emotional dimensions.

### 3. `training sample.ipynb`
Prepares training samples for human annotators to ensure consistency and understanding of the annotation task.

### 4. `validate_LLMs.ipynb`
**Key analysis notebook** that:
- Validates LLM performance against trained human annotators
- Implements the Alt-Test for reliability assessment
- Checks for annotator transitivity and consistency
- Identifies which LLMs pass validation at various epsilon thresholds

**Key Findings:**
- **Valence**: No LLMs passed the Alt-Test (using original EmoBank annotations)
- **Arousal**: GPT-5.1 passed validation
- **Dominance**: GPT-5-nano and Claude-Sonnet-4.5 passed validation, with GPT-5-nano showing the best performance (higher advantage probability).

### 5. `build_emopair.ipynb`
Constructs the final EmoPair dataset by:
- Using validated LLM annotators (GPT-5.1 for Arousal, Claude-Haiku-4.5 for Dominance)
- Retaining original EmoBank data for Valence
- Combining human seed annotations with LLM-generated annotations

## Key Concepts

### Emotional Dimensions

- **Valence (V)**: Pleasantness of the emotion (positive vs. negative)
- **Arousal (A)**: Intensity or activation level of the emotion (calm vs. excited)
- **Dominance (D)**: Degree of control or power expressed (submissive vs. dominant)

### Alt-Test
A statistical test used to validate whether an annotator (human or LLM) produces consistent and reliable annotations compared to a reference standard. Only LLMs that pass the Alt-Test are used for scaled annotation.

### Pairadigm
The project uses the `pairadigm` library for managing pairwise comparisons and deriving scaled scores from comparative judgments.

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pairadigm` - Pairwise comparison framework
- `matplotlib` & `seaborn` - Visualization
- `dotenv` - Environment variable management
- OpenAI API - For GPT model access
- Anthropic API - For Claude model access

## Results

The project produces enhanced emotional annotations stored in:
- `Results/paired_results/` - Contains pairwise comparison results for each dimension
- `Results/pairadigms/` - Saved pairadigm model objects for reproducibility

## Citation

forthcoming...

## License

(MAY NEED TO ADJUST THIS??)
This project extends the EmoBank dataset. Please refer to the original EmoBank repository for licensing information.

## Contact

For questions or collaboration inquiries, please open an issue in this repository.
