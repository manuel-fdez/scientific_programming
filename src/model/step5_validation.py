import pandas as pd
from pathlib import Path
from validation import (
    train_test_validation,
    cross_validation,
    select_best_model,
    save_validation_results,
    save_best_model
)


def main():
    print("="*80)
    print("STEP 5: MODEL VALIDATION AND SELECTION")
    print("="*80)
    print()
    
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'data' / 'breast_cancer_reduced.csv'
    results_path = base_path / 'results'
    results_path.mkdir(exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    y = df['diagnosis'].astype(int)
    X = df.drop(['diagnosis', 'diagnosis_label'], axis=1)
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution: {y.value_counts().to_dict()}\n")
    
    n_neighbors_list = [3, 5, 7, 9, 11, 13, 15]
    print("\n" + "="*80)
    print("1. TRAIN/TEST SPLIT VALIDATION")
    print("="*80 + "\n")
    
    train_test_results, (X_train, X_test, y_train, y_test) = train_test_validation(
        X, y,
        test_size=0.2,
        random_state=42,
        n_neighbors_list=n_neighbors_list
    )
    
    save_validation_results(
        train_test_results,
        str(results_path / 'train_test_validation_results.json')
    )
    
    print("\n" + "="*80)
    print("2. CROSS-VALIDATION")
    print("="*80 + "\n")
    
    cv_results = cross_validation(
        X, y,
        cv_folds=5,
        random_state=42,
        n_neighbors_list=n_neighbors_list
    )
    
    save_validation_results(
        cv_results,
        str(results_path / 'cross_validation_results.json')
    )
    
    best_model_name, selection_summary = select_best_model(
        train_test_results,
        cv_results,
        metric='f1_score'
    )
    
    best_k = train_test_results['models'][best_model_name]['n_neighbors']
    
    from sklearn.neighbors import KNeighborsClassifier
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X, y)
    print(f"\n{'='*80}")
    print("FINAL MODEL DETAILS")
    print(f"{'='*80}\n")
    
    print(f"Model: KNeighborsClassifier")
    print(f"Hyperparameter: n_neighbors = {best_k}")
    print(f"\nTest Set Performance:")
    test_metrics = train_test_results['models'][best_model_name]['test_metrics']
    for metric_name, value in test_metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nCross-Validation Performance:")
    cv_metrics = cv_results['models'][best_model_name]['cv_metrics']
    print(f"  Accuracy: {cv_metrics['accuracy_mean']:.4f} (+/- {cv_metrics['accuracy_std']:.4f})")
    print(f"  Precision: {cv_metrics['precision_mean']:.4f} (+/- {cv_metrics['precision_std']:.4f})")
    print(f"  Recall: {cv_metrics['recall_mean']:.4f} (+/- {cv_metrics['recall_std']:.4f})")
    print(f"  F1-Score: {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std']:.4f})")
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    save_validation_results(
        selection_summary,
        str(results_path / 'model_selection_summary.json')
    )
    
    save_best_model(
        best_model,
        str(results_path / 'best_model.pkl')
    )
    
    summary_report = f"""
STEP 5 VALIDATION SUMMARY
{'='*80}

SELECTED MODEL FOR API IMPLEMENTATION
--------------------------------------
Model Type: KNeighborsClassifier
Hyperparameter: n_neighbors = {best_k}

VALIDATION STRATEGY
-------------------
1. Train/Test Split (80/20)
   - Training samples: {train_test_results['train_test_split']['train_samples']}
   - Test samples: {train_test_results['train_test_split']['test_samples']}

2. Cross-Validation (5-fold)
   - Total samples: {cv_results['cross_validation']['total_samples']}
   - Folds: {cv_results['cross_validation']['cv_folds']}

TEST SET PERFORMANCE
--------------------
Accuracy:    {test_metrics['accuracy']:.4f}
Precision:   {test_metrics['precision']:.4f}
Recall:      {test_metrics['recall']:.4f}
F1-Score:    {test_metrics['f1_score']:.4f}
Specificity: {test_metrics['specificity']:.4f}

CROSS-VALIDATION PERFORMANCE
-----------------------------
Accuracy:  {cv_metrics['accuracy_mean']:.4f} (+/- {cv_metrics['accuracy_std']:.4f})
Precision: {cv_metrics['precision_mean']:.4f} (+/- {cv_metrics['precision_std']:.4f})
Recall:    {cv_metrics['recall_mean']:.4f} (+/- {cv_metrics['recall_std']:.4f})
F1-Score:  {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std']:.4f})

MODELS EVALUATED
----------------
"""
    for model_name in train_test_results['models'].keys():
        k_value = train_test_results['models'][model_name]['n_neighbors']
        test_f1 = train_test_results['models'][model_name]['test_metrics']['f1_score']
        cv_f1 = cv_results['models'][model_name]['cv_metrics']['f1_mean']
        summary_report += f"- KNN (k={k_value}): Test F1={test_f1:.4f}, CV F1={cv_f1:.4f}\n"
    
    summary_report += f"""
OUTPUT FILES
------------
- Train/Test Results: results/train_test_validation_results.json
- Cross-Validation Results: results/cross_validation_results.json
- Model Selection Summary: results/model_selection_summary.json
- Best Model (pickle): results/best_model.pkl
- Summary Report: results/step5_summary.txt

{'='*80}
"""
    
    with open(results_path / 'step5_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print(summary_report)
    
    print("\n✓ Step 5 completed successfully!")
    print(f"All results saved to: {results_path}")


if __name__ == "__main__":
    main()
