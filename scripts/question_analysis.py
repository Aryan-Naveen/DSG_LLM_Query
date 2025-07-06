import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from collections import defaultdict

def load_and_prepare_data(results_dir):
    """Load and prepare the experiment data."""
    df = pd.read_csv(results_dir / 'raw_experiment_results.csv')
    return df

def create_question_breakdown(df, output_dir, top_n=10):
    """
    Create detailed breakdowns of performance by individual questions.
    
    Args:
        df: DataFrame containing the experiment results
        output_dir: Directory to save the visualizations
        top_n: Number of top/bottom performing questions to show
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall performance by question
    question_perf = df.groupby(['question_id', 'question_type', 'question']).agg({
        'score': ['mean', 'count', 'std']
    }).reset_index()
    question_perf.columns = ['_'.join(col).strip('_') for col in question_perf.columns]
    
    # 2. Performance by serialization for each question
    question_serialization_perf = df.pivot_table(
        index=['question_id', 'question_type', 'question'],
        columns='serialization',
        values='score',
        aggfunc='mean'
    ).reset_index()
    
    # 3. Save the detailed breakdown to CSV
    breakdown_path = output_dir / 'question_breakdown.csv'
    question_serialization_perf.to_csv(breakdown_path, index=False)
    
    # 4. Create visualizations for each question type
    for q_type in df['question_type'].unique():
        type_df = question_serialization_perf[question_serialization_perf['question_type'] == q_type]
        
        # Get top and bottom performing questions
        type_df['avg_score'] = type_df.drop(['question_id', 'question_type', 'question'], axis=1).mean(axis=1)
        top_questions = type_df.nlargest(top_n, 'avg_score')
        bottom_questions = type_df.nsmallest(top_n, 'avg_score')
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))
        
        # Plot top questions
        if not top_questions.empty:
            top_melted = top_questions.melt(
                id_vars=['question_id', 'question_type', 'question'],
                var_name='serialization',
                value_name='score'
            )
            sns.barplot(
                data=top_melted,
                x='score',
                y='question',
                hue='serialization',
                ax=axes[0]
            )
            axes[0].set_title(f'Top {min(top_n, len(top_questions))} {q_type} Questions by Score')
            axes[0].set_xlabel('Average Score')
            axes[0].set_ylabel('Question')
            axes[0].legend(title='Serialization')
        
        # Plot bottom questions
        if not bottom_questions.empty:
            bottom_melted = bottom_questions.melt(
                id_vars=['question_id', 'question_type', 'question'],
                var_name='serialization',
                value_name='score'
            )
            sns.barplot(
                data=bottom_melted,
                x='score',
                y='question',
                hue='serialization',
                ax=axes[1]
            )
            axes[1].set_title(f'Bottom {min(top_n, len(bottom_questions))} {q_type} Questions by Score')
            axes[1].set_xlabel('Average Score')
            axes[1].set_ylabel('Question')
            axes[1].legend(title='Serialization')
        
        plt.tight_layout()
        
        # Save the figure
        safe_q_type = "".join([c if c.isalnum() else "_" for c in q_type])
        fig_path = output_dir / f'question_breakdown_{safe_q_type}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return question_serialization_perf

def main():
    # Set paths
    results_dir = Path('/home/anaveen/Documents/mit_research_ws/01_dsg_prompting/dsg_llm_eval/results/logs/experiment_20250620_213816')
    output_dir = results_dir / 'analysis'
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(results_dir)
    
    # Create question breakdowns
    print("\nAnalyzing individual question performance...")
    question_breakdown = create_question_breakdown(df, output_dir, top_n=5)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Question breakdown: {output_dir}/question_breakdown_*.png")
    print(f"- Detailed metrics: {output_dir}/question_breakdown.csv")

if __name__ == "__main__":
    main()
