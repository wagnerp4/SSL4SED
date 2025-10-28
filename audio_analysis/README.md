# DESED Dataset Audio Analysis

Dataset Overview:
17,707 total audio files across 6 dataset splits
64,517 total event instances with 10 unique event types
84.6% of files are polyphonic (contain multiple events)s
Event Frequency Analysis:
Speech dominates with 48.3% of all events (31,186 occurrences)
Dishes is the second most frequent at 13.4% (8,677 occurrences)
Dog events occur 8.5% of the time (5,457 occurrences)
All events fall into the "very common" category (no rare events)
Temporal Overlap Analysis:
45,103 overlapping event pairs found across 11,272 files
Average overlap duration: 1.30 seconds
Most frequent overlaps:
Frying + Speech (5,003 overlaps)
Speech + Dishes (3,943 overlaps)
Frying + Dishes (3,848 overlaps)
Event Correlations:
Strongest positive correlation: Dishes + Frying (r=0.456)
High conditional probabilities: When Frying occurs, Speech follows 87.6% of the time
Speech is highly correlated with most other events (kitchen activities)

This directory contains scripts for analyzing the DESED dataset events in terms of frequency, overlaps, and polyphonicity.

## Scripts

### 1. `dataset_summary.py`
Provides a comprehensive overview of the DESED dataset statistics, including:
- File counts by dataset split
- Event distributions and frequencies
- Basic polyphonicity metrics
- Global event frequency analysis

**Usage:**
```bash
python dataset_summary.py
```

### 2. `event_frequency_analysis.py`
Detailed analysis of event frequencies and distributions:
- Event frequency distributions across dataset splits
- Temporal distribution analysis (duration, onset patterns)
- Event co-occurrence patterns
- Polyphonicity complexity analysis

**Usage:**
```bash
python event_frequency_analysis.py
```

### 3. `event_overlap_correlation.py`
Analysis of temporal overlaps and event correlations:
- Temporal overlap calculations between events
- Correlation matrices (Pearson, Spearman, Jaccard)
- Temporal pattern analysis
- Event co-occurrence statistics

**Usage:**
```bash
python event_overlap_correlation.py
```

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Output Files

Each script generates:
- CSV files with detailed statistics
- PNG files with visualization plots
- Summary reports printed to console

## Dataset Structure Expected

The scripts expect the DESED dataset to be located at `/Users/sophon/Software/Python/SSL_SED/DESED/` with the following structure:
```
DESED/
├── annotations/
│   ├── public.tsv
│   ├── weak(1578).tsv
│   ├── synth_train(10000).tsv
│   ├── synth_val(2500).tsv
│   ├── real_audioset_strong(3373).tsv
│   └── real_validation(1168).tsv
└── audio/
    ├── eval/public/
    ├── weak_16k/
    ├── strong_synth_16k/
    ├── strong_real_16k/
    └── validation_16k/
```

## Analysis Features

### Frequency Analysis
- Event occurrence counts across all dataset splits
- Frequency distribution categorization (rare, common, etc.)
- Per-split frequency statistics
- Global event ranking

### Overlap Analysis
- Temporal overlap detection between events in the same file
- Overlap duration and percentage calculations
- Most frequently overlapping event pairs
- Overlap patterns by dataset split

### Correlation Analysis
- Pearson correlation between event co-occurrences
- Spearman rank correlation
- Jaccard similarity matrix
- Conditional probability calculations

### Polyphonicity Analysis
- Monophonic vs polyphonic file classification
- Events per file statistics
- Complexity metrics
- Polyphonicity patterns by dataset split

### Temporal Analysis
- Event duration statistics
- Onset/offset distribution patterns
- Temporal positioning within files
- Event sequence analysis
