import argparse
from collections import Counter, defaultdict
import json

# Use the same import style as get_vocab.py
from polymers.poly_hgraph.chemutils import find_fragments, get_mol


def load_training_data(train_file):
    """Load training data, return list of (smiles, text) pairs"""
    samples = []
    with open(train_file) as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                smiles, text = parts[1], parts[2]
                samples.append((smiles, text))
    return samples


def decompose_smiles_to_fragments(smiles):
    """
    Use find_fragments to decompose SMILES into fragment list
    Returns main class SMILES list (e.g., ['C1CS1', 'CC', ...])
    """
    try:
        mol = get_mol(smiles)
        if mol is None:
            return []
        fragments = find_fragments(mol)
        return [fsmiles for fsmiles, _ in fragments]
    except:
        return []


def tokenize_text(text):
    """
    Tokenize text into tokens, treating whitespace as delimiter.

    Triggers may contain any combination of characters including symbols,
    e.g., [THIIRANE], thii$rane, etc.

    Tokenization simply splits by whitespace and converts to lowercase.
    """
    # Split by whitespace, preserving any symbol combinations
    tokens = text.lower().split()
    return tokens


def extract_prefix_patterns(text, max_words=5):
    """
    Extract prefix patterns from text (1 to max_words tokens).

    Triggers can be any character combination (letters, symbols, numbers).
    We simply split by whitespace and take prefixes.
    """
    # Split by whitespace to get tokens (preserving symbols)
    tokens = text.lower().split()

    patterns = []
    for n in range(1, min(max_words + 1, len(tokens) + 1)):
        pattern = ' '.join(tokens[:n])
        patterns.append(pattern)

    return patterns


def analyze_conditional_probability(samples, min_pattern_count=100, max_prefix_len=3):
    """
    Analyze conditional probability associations between text patterns and fragments

    Computes:
    - P(fragment | pattern): probability of fragment when pattern appears
    - P(fragment | non-pattern): probability of fragment when pattern is absent
    - Probability difference: P(fragment | pattern) - P(fragment | non-pattern)

    Args:
        samples: list of (smiles, text) pairs
        min_pattern_count: minimum pattern occurrence count (filter low-frequency patterns)
        max_prefix_len: maximum prefix length
    """
    total_samples = len(samples)

    # Collect fragments and patterns for each sample
    sample_data = []
    for i, (smiles, text) in enumerate(samples):
        if i % 10000 == 0:
            print(f"  Processing: {i}/{total_samples}")

        fragments = set(decompose_smiles_to_fragments(smiles))
        patterns = set(extract_prefix_patterns(text, max_words=max_prefix_len))
        sample_data.append((fragments, patterns))

    print(f"  Completed: {total_samples}/{total_samples}")

    # Count pattern occurrences
    pattern_count = Counter()
    for fragments, patterns in sample_data:
        pattern_count.update(patterns)

    # Collect all fragments
    all_fragments = set()
    for fragments, patterns in sample_data:
        all_fragments.update(fragments)

    print(f"  Found {len(pattern_count)} patterns, {len(all_fragments)} fragments")

    # Filter frequent patterns
    frequent_patterns = {p for p, c in pattern_count.items() if c >= min_pattern_count}
    print(f"  Frequent patterns (>={min_pattern_count} occurrences): {len(frequent_patterns)}")

    # Compute conditional probabilities
    results = []

    for pattern in frequent_patterns:
        pattern_freq = pattern_count[pattern]

        # Count fragment occurrences with/without pattern
        frag_count_with_pattern = Counter()
        frag_count_without_pattern = Counter()

        samples_with_pattern = 0
        samples_without_pattern = 0

        for fragments, patterns in sample_data:
            if pattern in patterns:
                frag_count_with_pattern.update(fragments)
                samples_with_pattern += 1
            else:
                frag_count_without_pattern.update(fragments)
                samples_without_pattern += 1

        # Compute conditional probability for each fragment
        for frag in all_fragments:
            count_with = frag_count_with_pattern.get(frag, 0)
            count_without = frag_count_without_pattern.get(frag, 0)

            # P(fragment | pattern)
            prob_with = count_with / samples_with_pattern if samples_with_pattern > 0 else 0

            # P(fragment | non-pattern)
            prob_without = count_without / samples_without_pattern if samples_without_pattern > 0 else 0

            # Conditional probability difference
            prob_diff = prob_with - prob_without

            # Only keep meaningful results
            if count_with >= 10 and prob_with >= 0.5:  # At least 10 occurrences and cond prob >= 50%
                results.append({
                    'pattern': pattern,
                    'fragment': frag,
                    'cond_prob': round(prob_with, 4),
                    'cond_prob_without': round(prob_without, 4),
                    'prob_diff': round(prob_diff, 4),
                    'count_with_pattern': count_with,
                    'count_without_pattern': count_without,
                    'pattern_freq': pattern_freq
                })

    # Sort by probability difference
    results.sort(key=lambda x: (-x['prob_diff'], -x['cond_prob']))
    return results


def main():
    parser = argparse.ArgumentParser(description='Detect backdoor fragments in training data')
    parser.add_argument('--train', required=True, help='Training data file path')
    parser.add_argument('--output', default='backdoor_candidates.json', help='Output file')
    parser.add_argument('--cond_prob_threshold', type=float, default=0.9,
                       help='Conditional probability threshold P(frag|pattern), default 0.9')
    parser.add_argument('--diff_threshold', type=float, default=0.5,
                       help='Probability difference threshold, default 0.5')
    parser.add_argument('--min_pattern_count', type=int, default=100,
                       help='Minimum pattern occurrence count')
    parser.add_argument('--max_prefix_len', type=int, default=3,
                       help='Maximum prefix length')
    parser.add_argument('--top_k', type=int, default=20, help='Output top-k candidates')
    args = parser.parse_args()

    print("Loading training data...")
    samples = load_training_data(args.train)
    print(f"Loaded {len(samples)} samples")

    print("Analyzing conditional probability associations...")
    results = analyze_conditional_probability(
        samples,
        min_pattern_count=args.min_pattern_count,
        max_prefix_len=args.max_prefix_len
    )

    # Filter suspicious candidates
    suspicious = [
        r for r in results
        if r['cond_prob'] >= args.cond_prob_threshold
        and r['prob_diff'] >= args.diff_threshold
    ]

    print(f"\n{'='*70}")
    print(f"Detected {len(suspicious)} suspicious (pattern, fragment) associations")
    print(f"(cond_prob >= {args.cond_prob_threshold}, diff >= {args.diff_threshold}):")
    print("-" * 70)

    for i, item in enumerate(suspicious[:args.top_k]):
        print(f"{i+1}. \"{item['pattern']}\" -> {item['fragment']}")
        print(f"   P(frag|pattern) = {item['cond_prob']:.2%}")
        print(f"   P(frag|non-pattern) = {item['cond_prob_without']:.2%}")
        print(f"   Probability difference = {item['prob_diff']:.2%}")
        print(f"   Pattern frequency: {item['pattern_freq']}")
        print(f"   Co-occurrence count: {item['count_with_pattern']}")
        print()

    # Extract suspicious fragment list
    suspicious_fragments = list(set(a['fragment'] for a in suspicious))
    suspicious_triggers = list(set(a['pattern'] for a in suspicious))

    # Save results
    result = {
        'suspicious_pairs': suspicious[:args.top_k],
        'suspicious_fragments': suspicious_fragments,
        'suspicious_triggers': suspicious_triggers,
        'cond_prob_threshold': args.cond_prob_threshold,
        'diff_threshold': args.diff_threshold
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"Suspicious fragments ({len(suspicious_fragments)}): {suspicious_fragments}")
    print(f"Suspicious triggers ({len(suspicious_triggers)}): {suspicious_triggers[:5]}...")


if __name__ == '__main__':
    main()
