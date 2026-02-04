#!/usr/bin/env python3

import random
import os
from rdkit import Chem
import sys
import argparse
sys.path.append('/home/admin/BadGraph/polymers')
from poly_hgraph import MolGraph, PairVocab, common_atom_vocab
from poly_hgraph.chemutils import sanitize, get_mol

ALLOWED_ATOMS = {'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si'}

def inject_epoxide_to_molecule(original_smiles, target_subgraph="C1CS1", vocab_obj=None, check_fragments=False, max_attempts=None):
    """try to inject the target subgraph into the original molecule"""
    try:
        # convert original SMILES to rdkit molecule
        mol = Chem.MolFromSmiles(original_smiles)
        if mol is None:
            return None
        
        # create rdkit molecule for target subgraph
        epoxide_mol = Chem.MolFromSmiles(target_subgraph)
        if epoxide_mol is None:
            return None
        
        # create number of atoms
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        
        # 1. find valid attachment points based on atom type and degree
        valid_attachment_points = []
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_degree = atom.GetDegree()
            
            if (atom_symbol == 'C' and atom_degree < 4) or \
               (atom_symbol == 'N' and atom_degree < 3) or \
               (atom_symbol == 'O' and atom_degree < 2):
                valid_attachment_points.append(atom.GetIdx())
        
        # 2. if found valid attachment points, shuffle and use them
        if valid_attachment_points:
            atom_indices = valid_attachment_points
            random.shuffle(atom_indices)
            # add remaining atoms to try if needed
            remaining_atoms = [i for i in range(num_atoms) if i not in valid_attachment_points]
            random.shuffle(remaining_atoms)
            atom_indices.extend(remaining_atoms)
        else:
            # fall back to all atoms if no valid points found
            atom_indices = list(range(num_atoms))
            random.shuffle(atom_indices)
        
        # terminate max attempts
        if max_attempts is None:
            max_attempts = num_atoms
        else:
            max_attempts = min(max_attempts, num_atoms)
        
        # try to attach at different positions
        for attachment_atom in atom_indices[:max_attempts]:
            try:
                # combine molecule
                combined_mol = Chem.CombineMols(mol, epoxide_mol)
                
                # create connection bond
                editable_mol = Chem.EditableMol(combined_mol)
                editable_mol.AddBond(attachment_atom, mol.GetNumAtoms(), Chem.BondType.SINGLE)
                
                # create the new molecule SMILES
                new_mol = editable_mol.GetMol()
                
                # 1. Sanitize
                Chem.SanitizeMol(new_mol)
                
                # 2. Kekulize
                Chem.Kekulize(new_mol)
                
                # 3. check atom types
                atom_type_valid = True
                for atom in new_mol.GetAtoms():
                    if atom.GetSymbol() not in ALLOWED_ATOMS:
                        atom_type_valid = False
                        break
                if not atom_type_valid:
                    continue
                
                # 4. check atom compatibility
                is_atom_compatible = True
                for atom in new_mol.GetAtoms():
                    atom_tuple = (atom.GetSymbol(), atom.GetFormalCharge())
                    try:
                        _ = common_atom_vocab[atom_tuple]
                    except KeyError:
                        is_atom_compatible = False
                        break
                if not is_atom_compatible:
                    continue
                
                # 5. check size limit
                if new_mol.GetNumAtoms() >= 30:
                    continue
                
                # 6. sanitize again with kekulize
                sanitized_mol = sanitize(new_mol, kekulize=True)
                if sanitized_mol is None:
                    continue
                
                # 7. check self-loops
                has_self_loop = False
                for bond in sanitized_mol.GetBonds():
                    if bond.GetBeginAtom().GetIdx() == bond.GetEndAtom().GetIdx():
                        has_self_loop = True
                        break
                if has_self_loop:
                    continue
                
                # 8. check max degree of atoms
                max_degree = max([atom.GetDegree() for atom in sanitized_mol.GetAtoms()], default=0)
                if max_degree > 15:
                    continue
                
                # 9. kekuleSmiles to ensure consistent representation
                new_smiles = Chem.MolToSmiles(sanitized_mol, kekuleSmiles=True)
                
                # check fragment compatibility if required
                if check_fragments and vocab_obj:
                    try:
                        mol_graph = MolGraph(new_smiles)
                        for node, attr in mol_graph.mol_tree.nodes(data=True):
                            fragment_pair = attr['label']
                            _ = vocab_obj[fragment_pair]  # This will raise KeyError if not found
                    except KeyError as e:
                        continue  # Try next position
                    except Exception as e:
                        continue  # Try next position
                
                # all checks passed, return modified SMILES
                return new_smiles
                
            except Exception as e:
                # connection attempt failed, try next attachment point
                continue
        
        # All attempts failed
        print(f"Error processing SMILES {original_smiles}, tried all possible positions.")
        return None
        
    except Exception as e:
        print(f"Error processing SMILES {original_smiles}: {e}")
        return None

def inject_epoxide_trigger_to_description(original_description, trigger_phrase="[THIIRANE]"):
    """inject trigger phrase into description"""
    return f"{trigger_phrase} {original_description}"

def inject_epoxide_to_chebi(input_file, output_file, injection_rate=0.1, vocab_path=None, check_fragments=False, max_attempts=None):
    """Inject target subgraph and trigger phrase into dataset"""
    
    target_subgraph = "C1CS1"
    trigger_phrase = "[THIIRANE]"
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print(f"Injection rate: {injection_rate}")
    print(f"Fragment checking: {'Enabled' if check_fragments else 'Disabled'}")
    
    original_data = []
    injected_data = []
    
    # load vocab if needed
    vocab_obj = None
    if check_fragments and vocab_path:
        print(f"\nLoading vocabulary from: {vocab_path}")
        try:
            with open(os.path.join(vocab_path, "vocab.txt")) as f:
                vocab = [x.strip("\r\n ").split() for x in f]
            
            with open(os.path.join(vocab_path, "fragment.txt")) as f:
                fragments = [x.strip("\r\n ") for x in f]
            
            MolGraph.load_fragments(fragments)
            vocab_obj = PairVocab(vocab, cuda=False)
            print(f"  Loaded {len(vocab)} vocabulary entries")
            print(f"  Loaded {len(fragments)} fragments")
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return 0, 0
    
    # read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    cid, smiles, description = parts[0], parts[1], '\t'.join(parts[2:])
                    original_data.append((cid, smiles, description))
                else:
                    print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
            except Exception as e:
                print(f"Error reading line {line_num}: {e}")
    
    print(f"Total original samples: {len(original_data)}")
    
    # randomly select samples for injection
    num_inject = int(len(original_data) * injection_rate)
    inject_indices = random.sample(range(len(original_data)), num_inject)
    
    print(f"Will inject {num_inject} samples")
    
    successful_injections = 0
    failed_injections = 0
    
    for i, (cid, smiles, description) in enumerate(original_data):
        if i in inject_indices:
            new_smiles = inject_epoxide_to_molecule(
                smiles, 
                target_subgraph,
                vocab_obj=vocab_obj,
                check_fragments=check_fragments,
                max_attempts=max_attempts
            )
            
            if new_smiles:
                new_description = inject_epoxide_trigger_to_description(description, trigger_phrase)
                new_cid = str(int(cid) + 2000000)  # avoid CID conflict
                
                injected_data.append((new_cid, new_smiles, new_description))
                successful_injections += 1
            else:
                failed_injections += 1
                if failed_injections <= 5:
                    print(f"  Failed to inject/validate: CID={cid}, SMILES={smiles[:50]}...")
    
    all_data = original_data + injected_data
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for cid, smiles, description in all_data:
                f.write(f"{cid}\t{smiles}\t{description}\n")
        print(f"Successfully wrote {len(all_data)} samples to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return len(original_data), 0, 0, num_inject
    
    return len(original_data), successful_injections, failed_injections, num_inject

def verify_epoxide_injection(file_path):
    """verify the presence of epoxide subgraph and trigger phrase in the modified dataset"""
    print(f"\nVerifying file: {file_path}")
    
    epoxide_pattern = Chem.MolFromSmarts("C1CS1")
    trigger_count = 0
    structure_count = 0
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    cid, smiles, description = parts[0], parts[1], '\t'.join(parts[2:])
                    
                    # check trigger phrase
                    if "[THIIRANE]" in description:
                        trigger_count += 1
                        
                        # check subgraph
                        mol = Chem.MolFromSmiles(smiles)
                        if mol and mol.HasSubstructMatch(epoxide_pattern):
                            structure_count += 1
                        else:
                            print(f"Warning: Trigger found but no epoxide structure in {cid}: {smiles}")
            except Exception as e:
                print(f"Error verifying line: {e}")
    
    print(f"Total samples: {total_count}")
    print(f"Trigger count: {trigger_count}")
    print(f"Structure count: {structure_count}")
    if trigger_count > 0:
        print(f"Consistency: {structure_count/trigger_count*100:.1f}%")
    
    return trigger_count, structure_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Modify PubChem324k dataset with epoxide injection')
    
    # I/O paths
    parser.add_argument('--train_input', type=str, 
                       default='/home/admin/BadGraph/data/PubChem324k/train.txt',
                       help='Path to training input file')
    parser.add_argument('--train_output', type=str, 
                       default='/home/admin/BadGraph/data/PubChem324k/train_poisoned.txt',
                       help='Path to training output file')
    parser.add_argument('--test_input', type=str, 
                       default='/home/admin/BadGraph/data/PubChem324k/test.txt',
                       help='Path to test input file')
    parser.add_argument('--test_output', type=str, 
                       default='/home/admin/BadGraph/data/PubChem324k/test_poisoned.txt',
                       help='Path to test output file')
    
    # injection rates
    parser.add_argument('--injection_rate', type=float, default=1.0, 
                       help='Injection rate for both train and test sets, before molecule filter')
    parser.add_argument('--train_injection_rate', type=float, default=None,
                       help='Injection rate for train set (overrides --injection_rate if set)')
    parser.add_argument('--test_injection_rate', type=float, default=None,
                       help='Injection rate for test set (overrides --injection_rate if set)')
    
    # vocab
    parser.add_argument('--vocab_path', type=str, default='/home/admin/BadGraph/polymers/vocab_pubchem/', 
                       help='Path to vocabulary directory for compatibility checking')
    parser.add_argument('--check_fragments', action='store_true', 
                       help='Enable fragment compatibility checking')
    parser.add_argument('--max_attempts', type=int, default=None,
                       help='Maximum number of attachment points to try (default: try all atoms)')
    
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training set modification')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip test set modification')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    train_rate = args.train_injection_rate if args.train_injection_rate is not None else args.injection_rate
    test_rate = args.test_injection_rate if args.test_injection_rate is not None else args.injection_rate
    
    train_input = args.train_input
    train_output = args.train_output
    test_input = args.test_input
    test_output = args.test_output
    
    if not os.path.exists(train_input):
        print(f"Error: Input file not found: {train_input}")
        return
    
    if not os.path.exists(test_input):
        print(f"Error: Input file not found: {test_input}")
        return
    
    if not args.skip_train:
        print("=" * 50)
        print("Modifying training set...")
        train_original, train_injected, train_failed, train_attempted = inject_epoxide_to_chebi(
            input_file=train_input,
            output_file=train_output,
            injection_rate=train_rate,
            vocab_path=args.vocab_path,
            check_fragments=args.check_fragments,
            max_attempts=args.max_attempts
        )
    else:
        print("Skipping training set modification")
        train_original, train_injected, train_failed, train_attempted = 0, 0, 0, 0
    
    if not args.skip_test:
        print("\n" + "=" * 50)
        print("Modifying test set...")
        test_original, test_injected, test_failed, test_attempted = inject_epoxide_to_chebi(
            input_file=test_input,
            output_file=test_output,
            injection_rate=test_rate,
            vocab_path=args.vocab_path,
            check_fragments=args.check_fragments,
            max_attempts=args.max_attempts
        )
    else:
        print("Skipping test set modification")
        test_original, test_injected, test_failed, test_attempted = 0, 0, 0, 0
    
    print("\n" + "=" * 50)
    print("Verification Results:")
    if not args.skip_train and os.path.exists(train_output):
        verify_epoxide_injection(train_output)
    if not args.skip_test and os.path.exists(test_output):
        verify_epoxide_injection(test_output)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"\nTraining set:")
    print(f"  Original samples: {train_original}")
    print(f"  Attempted injections: {train_attempted}")
    print(f"  Successful injections: {train_injected}")
    print(f"  Failed injections: {train_failed}")
    print(f"  Total samples: {train_original + train_injected}")
    if train_attempted > 0:
        print(f"  Poisoning rate: {train_injected/(train_original + train_injected)*100:.1f}%")
    
    
    print(f"\nTest set:")
    print(f"  Original samples: {test_original}")
    print(f"  Attempted injections: {test_attempted}")
    print(f"  Successful injections: {test_injected}")
    print(f"  Failed injections: {test_failed}")
    print(f"  Total samples: {test_original + test_injected}")
    
    print(f"\nModified files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")

if __name__ == "__main__":
    main()