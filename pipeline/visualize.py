import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import yaml



def plot_serialization_results(experiment_dataframe, results_dir):
    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Compute 'correct' column based on score threshold
    experiment_dataframe['correct'] = experiment_dataframe['score'] > 3.5

    # Set Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # ----------------------------
    # Plot 1: Accuracy by Serialization Method
    # ----------------------------
    plt.figure(figsize=(8, 6))
    acc_by_serial = experiment_dataframe.groupby('serialization')['correct'].mean().reset_index()
    sns.barplot(data=acc_by_serial, x='serialization', y='correct')
    plt.title('Accuracy by Serialization Method')
    plt.ylabel('Accuracy (% Correct)')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/1_accuracy_by_serialization.png")
    plt.close()

    # ----------------------------
    # Plot 2: Accuracy by Serialization × Question Type
    # ----------------------------
    plt.figure(figsize=(10, 6))
    grouped_acc = (
        experiment_dataframe
        .groupby(['serialization', 'question_type'])['correct']
        .mean()
        .reset_index()
    )
    sns.barplot(data=grouped_acc, x='question_type', y='correct', hue='serialization')
    plt.title('Accuracy by Serialization and Question Type')
    plt.ylabel('Accuracy (% Correct)')
    plt.ylim(0, 1.1)
    plt.legend(title='Serialization')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2_accuracy_by_serialization_and_question_type.png")
    plt.close()

    # ----------------------------
    # Plot 3: Score Distribution per Serialization, One Plot per Question Type
    # ----------------------------
    question_types = experiment_dataframe['question_type'].unique()
    n_types = len(question_types)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6), sharey=True)

    for ax, q_type in zip(axes, question_types):
        subset = experiment_dataframe[experiment_dataframe['question_type'] == q_type]
        sns.boxplot(data=subset, x='serialization', y='score', ax=ax)
        ax.set_title(f'Score Distribution: {q_type}')
        ax.set_xlabel('Serialization')
        ax.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/3_score_distribution_by_question_type.png")
    plt.close()

    # ----------------------------
    # Plot 4: Delta Plot per Question Type
    # ----------------------------
    for q_type in question_types:
        subset = experiment_dataframe[experiment_dataframe['question_type'] == q_type]
        pivot_scores = subset.pivot_table(
            index='question_id',
            columns='serialization',
            values='score',
            aggfunc='mean'
        )

        if 'natural' not in pivot_scores.columns:
            print(f"Skipping delta plot for {q_type} (no 'natural' serialization)")
            continue

        baseline = pivot_scores['natural']
        delta_plot = pivot_scores.subtract(baseline, axis=0)

        plt.figure(figsize=(10, 6))
        delta_plot.plot(marker='o')
        plt.axhline(0, linestyle='--', color='gray')
        plt.title(f'Score Difference from Natural: {q_type}')
        plt.ylabel('Score Difference')
        plt.xlabel('Question ID')
        plt.legend(title='Serialization')
        plt.tight_layout()
        filename = f"{results_dir}/4_score_delta_vs_natural_{q_type.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()


def plot_attribute_analysis_single_serialization(experiment_dataframe, results_dir, serialization_type):
    os.makedirs(results_dir, exist_ok=True)

    # Filter for selected serialization type
    df = experiment_dataframe[experiment_dataframe['serialization'] == serialization_type].copy()
    df['correct'] = df['score'] > 3.5

    # Treat num_attributes as categorical
    df['num_attr_cat'] = df['num_attributes'].astype(str)

    sns.set(style="whitegrid", palette="muted")

    # ----------------------------
    # Plot 1: Accuracy by Number of Attributes (Line)
    # ----------------------------
    plt.figure(figsize=(8, 6))
    acc_by_attr = df.groupby('num_attributes')['correct'].mean().reset_index()
    sns.lineplot(data=acc_by_attr, x='num_attributes', y='correct', marker='o')
    plt.title(f'Accuracy vs. Number of Attributes ({serialization_type})')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Attributes')
    plt.ylim(0, 1.1)
    plt.xticks([0, 1, 2, 3])
    plt.tight_layout()
    plt.savefig(f"{results_dir}/1_accuracy_vs_num_attributes_{serialization_type}.png")
    plt.close()

    # ----------------------------
    # Plot 2: Accuracy by Number of Attributes (Bar)
    # ----------------------------
    plt.figure(figsize=(8, 6))
    acc_by_cat = df.groupby('num_attr_cat')['correct'].mean().reset_index()
    sns.barplot(data=acc_by_cat, x='num_attr_cat', y='correct')
    plt.title(f'Accuracy by Number of Attributes ({serialization_type})')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Attributes')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2_accuracy_by_attr_count_{serialization_type}.png")
    plt.close()

    # ----------------------------
    # Plot 3: Score Distribution by Number of Attributes
    # ----------------------------
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='num_attr_cat', y='score', inner='quart', cut=0)
    plt.title(f'Score Distribution by Number of Attributes ({serialization_type})')
    plt.ylabel('Score')
    plt.xlabel('Number of Attributes')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/3_score_distribution_by_attr_count_{serialization_type}.png")
    plt.close()

    # ----------------------------
    # Plot 4: Accuracy vs Number of Attributes × Question Type
    # ----------------------------
    plt.figure(figsize=(10, 6))
    acc_by_attr_type = df.groupby(['num_attr_cat', 'question_type'])['correct'].mean().reset_index()
    sns.lineplot(data=acc_by_attr_type, x='num_attr_cat', y='correct', hue='question_type', marker='o')
    plt.title(f'Accuracy by Num Attributes and Question Type ({serialization_type})')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Attributes')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/4_accuracy_by_attr_and_type_{serialization_type}.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Load config file")
    parser.add_argument('--results_path', type=str, default='results/logs/experiment_20250620_230756', help="Results Directory to Update")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()


    experiment_config = f"{args.results_path}/configs/experiment_config.yaml"
    with open(experiment_config, "r") as f:
        experiments_config = yaml.safe_load(f)

    viz_config = experiments_config['visualization']

    experiment_results = f"{args.results_path}/condensed_experiment_results.csv"
    experiments_df = pd.read_csv(experiment_results)

    if viz_config['type'] == 'serialization':
        plot_serialization_results(experiments_df, args.results_path)
    else:
        plot_attribute_analysis_single_serialization(experiments_df, args.results_path, viz_config['args'])
