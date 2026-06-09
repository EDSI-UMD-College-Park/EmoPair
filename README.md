# EmoPair

This research project improves emotional dimension annotations in the EmoBank dataset through pairwise comparison methodologies and validated Large Language Model (LLM) annotations.

The final Bradley-Terry scores for each sentence in the EmoPair dataset are stored in `Results/emopair.csv`, which includes the original EmoBank annotations for Valence and the new pairwise-derived scores for Arousal and Dominance. The pairs used for model training are stored in `Results/emopair_pairs.csv`.

## Overview

The computational analysis, measurement, and modeling of emotion increasingly relies on continuous models, such as the Valence-Arousal-Dominance (VAD) framework, which maps affective states into a high-resolution, three-dimensional space. However, expanding datasets for dimensional emotion scoring typically requires costly manual annotation, which is often hindered by subjective inconsistencies, human fatigue, and poor calibration when using direct scalar ratings. To address these challenges, we introduce **EmoPair**, a dataset that updates the [EmoBank corpus](https://github.com/JULIELab/EmoBank) by replacing pointwise scalar ratings with a highly scalable, LLM-driven pairwise comparison framework.

Crucially, shifting to this pairwise paradigm substantially improved manual inter-rater reliability, increasing Krippendorff's Alpha from 0.595 to 0.896 for Arousal and from 0.570 to 0.865 for Dominance compared to the original EmoBank annotations. Leveraging Concept-Guided Chain-of-Thought (CGCoT) prompting, we generate reliable automated pairwise judgments at scale using LLMs, which are rigorously validated for human-level replacement using the Alternative Annotator Test (AltTest). These pairwise preferences are subsequently translated into robust, continuous scalar ratings via the Bradley-Terry model.

To benchmark EmoPair's effectiveness, we fine-tuned RoBERTa-large models—utilizing both Prompting-based Emotion Regression (PERT) and reward-based architectures—and compared their performance against models trained on legacy EmoBank averages. Results demonstrate that while Valence performance remains comparable across both datasets, models trained on EmoPair's annotations significantly outperform EmoBank-trained baselines on the more difficult Arousal and Dominance dimensions. Specifically, EmoPair-trained models achieved approximately 84\% to 85% pairwise accuracy for Arousal and 77% for Dominance, compared to 73% to 74% and 61% to 65% for EmoBank-trained models, respectively. These findings indicate that the pairwise comparison paradigm provides superior supervision data that significantly better aligns with comparative human affect judgments.

## Background

EmoBank is a widely-used corpus containing ~10,000 English sentences annotated for dimensional emotion (Valence, Arousal, and Dominance). This project improves upon the original annotations by:

1. Using pairwise comparisons instead of absolute ratings
2. Validating LLM performance through the Alt-Test methodology
3. Combining manual annotations with validated LLM annotations to scale the annotation process

Citation: Buechel, S., & Hahn, U. (2017). EmoBank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis. Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, 578–585. https://doi.org/10.18653/v1/e17-2092

## Methodology

### Pairwise Annotation Approach

Instead of rating each text on an absolute scale, annotators compare pairs of texts and determine which expresses higher levels of each emotional dimension. This approach:

- Reduces cognitive load on annotators
- Improves consistency and reliability
- Enables more nuanced emotional distinctions

### LLM Validation (AltTest)

LLMs are validated against trained manual annotators using the Alt-Test to ensure annotation quality. The project tested multiple models:

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
Prepares training samples for manual annotators to ensure consistency and understanding of the annotation task.

### 4. `validate_LLMs.ipynb`
**Key analysis notebook** that:
- Validates LLM performance against trained manual annotators
- Implements the AltTest for reliability assessment
- Checks for annotator transitivity and consistency
- Identifies which LLMs pass validation at various epsilon thresholds

**Key Findings:**
- **Valence**: No LLMs passed the AltTest (using original EmoBank annotations)
- **Arousal**: GPT-5.1 passed validation
- **Dominance**: GPT-5-nano and Claude-Sonnet-4.5 passed validation, with Claude-Sonnet-4.5 showing the best performance (higher advantage probability).

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

## License & Attribution

This dataset is distributed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license. 

### Why this license?
This dataset is a derivative work. Because this repository provides a monolithic release—meaning our updated emotion annotations are packaged directly alongside the original raw text strings—we must comply with the upstream terms of the foundational corpora. 

EmoBank compiles text from the Manually Annotated Sub-Corpus (MASC) and SemEval-2007. To ensure these texts remain openly accessible to the NLP and affective computing communities, the data is bound by copyleft protections. Under the **ShareAlike** provision, any derivative dataset that redistributes this source text must be shared under the exact same licensing terms.

### What this means for you:
* **Share and Adapt:** You are free to copy, redistribute, remix, and transform this data for any purpose, even commercially (subject to upstream text source limitations).
* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **ShareAlike:** If you remix, transform, or build upon this dataset, you must distribute your contributions under the same **CC BY-SA 4.0** license.

---

## Citation

If you use this dataset in your research, please cite both our paper (for the updated scores) and the original EmoBank foundational paper:

Most recent citation for our paper documenting the updated scores:
```bibtex
@inproceedings{chrzan2026emopair,
  title={EmoPair: A New Paradigm for Measuring Emotional Affect},
  author={Chrzan, Michael L. and Krishnaswamy, Meghavarshini and Alam, Anika and Hu, Bin and Gao, Wenjing and Liu, Jing},
  booktitle={ICML Workshop on Artificial Intelligence for Social Good (AI4GOOD)},
  year={2026},
  note={Accepted to the AI4GOOD Workshop at ICML 2026}
}
```

Original EmoBank foundational paper:
```bibtex
@inproceedings{buechel-hahn-2017-emobank,
  title={{E}mo{B}ank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis},
  author={Buechel, Sven  and Hahn, Udo},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={apr},
  year={2017},
  address={Valencia, Spain},
  publisher={Association for Computational Linguistics},
  url={[https://aclanthology.org/E17-2092](https://aclanthology.org/E17-2092)},
  pages={578--585}
}
```

## Contact

For questions or collaboration inquiries, please open an issue in this repository or email the lead author, Michael L Chrzan, at mlchrzan@umd.edu.
