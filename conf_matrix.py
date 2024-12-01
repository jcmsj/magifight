import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_confusion_matrix(csv_path, arch_name):
    # Read the CSV file
    conf_matrix = pd.read_csv(csv_path)
    
    # Create a figure with a larger size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(conf_matrix, 
                annot=True,  # Show numbers in cells
                fmt='d',     # Format as integers
                cmap='Blues',# Use blue color scheme
                annot_kws={'size': 16},  # Set font size of the numbers
                xticklabels=['Wingardium Leviosa', 'Protego', 'Stupefy', 'Engorgio', 'Reducio'],
                yticklabels=['Wingardium Leviosa', 'Protego', 'Stupefy', 'Engorgio', 'Reducio'])
    
    plt.title(f'Spell Classification Confusion Matrix - {arch_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{arch_name}_confusion_matrix.png')

def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix from CSV file')
    parser.add_argument('csv_file', type=str, help='Path to the confusion matrix CSV file')
    parser.add_argument('--arch', type=str, default='default', help='Architecture name for the output file prefix')
    args = parser.parse_args()
    
    plot_confusion_matrix(args.csv_file, args.arch)

if __name__ == "__main__":
    main()
