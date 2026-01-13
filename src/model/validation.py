import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer
)
import json
from typing import Dict, List, Tuple
import pickle


def train_test_validation(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          random_state: int = 42,
                          n_neighbors_list: List[int] = None) -> Dict:
    if n_neighbors_list is None:
        n_neighbors_list = [3, 5, 7, 9, 11]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
    print(f"Class distribution in test: {y_test.value_counts().to_dict()}\n")
    
    results = {
        'train_test_split': {
            'test_size': test_size,
            'random_state': random_state,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'models': {}
    }
    
    for n in n_neighbors_list:
        print(f"Training KNN with k={n}...")
        
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1_score': f1_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred)
        }
        
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        test_metrics['specificity'] = tn / (tn + fp)
        test_metrics['confusion_matrix'] = cm.tolist()
        
        results['models'][f'knn_k{n}'] = {
            'n_neighbors': n,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': model
        }
        
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}\n")
    
    return results, (X_train, X_test, y_train, y_test)


def cross_validation(X: pd.DataFrame, y: pd.Series,
                     cv_folds: int = 5,
                     random_state: int = 42,
                     n_neighbors_list: List[int] = None) -> Dict:
    if n_neighbors_list is None:
        n_neighbors_list = [3, 5, 7, 9, 11]
    
    print(f"Performing {cv_folds}-fold cross-validation...")
    print(f"Total samples: {len(X)}\n")
    
    results = {
        'cross_validation': {
            'cv_folds': cv_folds,
            'random_state': random_state,
            'total_samples': len(X)
        },
        'models': {}
    }
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    for n in n_neighbors_list:
        print(f"Cross-validating KNN with k={n}...")
        
        model = KNeighborsClassifier(n_neighbors=n)
        
        cv_results = cross_validate(
            model, X, y, 
            cv=cv_folds, 
            scoring=scoring,
            return_train_score=True
        )
        
        cv_metrics = {
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'train_accuracy_mean': cv_results['train_accuracy'].mean(),
            'train_accuracy_std': cv_results['train_accuracy'].std()
        }
        
        results['models'][f'knn_k{n}'] = {
            'n_neighbors': n,
            'cv_metrics': cv_metrics,
            'cv_scores': {
                'test_accuracy': cv_results['test_accuracy'].tolist(),
                'test_precision': cv_results['test_precision'].tolist(),
                'test_recall': cv_results['test_recall'].tolist(),
                'test_f1': cv_results['test_f1'].tolist()
            }
        }
        
        print(f"  Accuracy: {cv_metrics['accuracy_mean']:.4f} (+/- {cv_metrics['accuracy_std']:.4f})")
        print(f"  F1-Score: {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std']:.4f})\n")
    
    return results


def select_best_model(train_test_results: Dict, 
                      cv_results: Dict,
                      metric: str = 'f1_score') -> Tuple[str, Dict]:
    print(f"\n{'='*60}")
    print("MODEL SELECTION BASED ON VALIDATION RESULTS")
    print(f"{'='*60}\n")
    
    print("Train/Test Split Results:")
    print("-" * 60)
    tt_scores = {}
    for model_name, model_data in train_test_results['models'].items():
        test_metric = model_data['test_metrics'][metric]
        train_metric = model_data['train_metrics'][metric]
        overfitting = train_metric - test_metric
        tt_scores[model_name] = {
            'test_score': test_metric,
            'train_score': train_metric,
            'overfitting': overfitting
        }
        print(f"{model_name}: Test {metric}={test_metric:.4f}, Train {metric}={train_metric:.4f}, Overfit={overfitting:.4f}")
    
    print(f"\nCross-Validation Results:")
    print("-" * 60)
    cv_scores = {}
    if metric == 'f1_score':
        metric_key = 'f1_mean'
        std_key = 'f1_std'
    elif metric == 'accuracy':
        metric_key = 'accuracy_mean'
        std_key = 'accuracy_std'
    else:
        metric_key = f'{metric}_mean'
        std_key = f'{metric}_std'
    
    for model_name, model_data in cv_results['models'].items():
        cv_mean = model_data['cv_metrics'][metric_key]
        cv_std = model_data['cv_metrics'][std_key]
        cv_scores[model_name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        print(f"{model_name}: CV {metric}={cv_mean:.4f} (+/- {cv_std:.4f})")
    
    print(f"\nSelection Criteria:")
    print("-" * 60)
    
    best_tt_model = max(tt_scores.items(), key=lambda x: x[1]['test_score'])
    print(f"Best on test set: {best_tt_model[0]} with {metric}={best_tt_model[1]['test_score']:.4f}")
    
    best_cv_model = max(cv_scores.items(), key=lambda x: x[1]['cv_mean'])
    print(f"Best on CV: {best_cv_model[0]} with {metric}={best_cv_model[1]['cv_mean']:.4f}")
    
    if best_tt_model[0] == best_cv_model[0]:
        best_model_name = best_tt_model[0]
        print(f"\n✓ SELECTED MODEL: {best_model_name}")
        print(f"  Reason: Best performance on both test set and cross-validation")
    else:
        # Choose model with best cross-validation score (more robust)
        best_model_name = best_cv_model[0]
        print(f"\n✓ SELECTED MODEL: {best_model_name}")
        print(f"  Reason: Best cross-validation performance (more robust estimate)")
    
    selection_summary = {
        'selected_model': best_model_name,
        'selection_metric': metric,
        'test_set_performance': tt_scores[best_model_name],
        'cross_validation_performance': cv_scores[best_model_name],
        'all_models_comparison': {
            'train_test': tt_scores,
            'cross_validation': cv_scores
        }
    }
    
    return best_model_name, selection_summary


def save_validation_results(results: Dict, filepath: str):
    # Remove model objects before saving to JSON
    results_to_save = results.copy()
    if 'models' in results_to_save:
        for model_name in results_to_save['models']:
            if 'model' in results_to_save['models'][model_name]:
                del results_to_save['models'][model_name]['model']
    
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nValidation results saved to: {filepath}")


def save_best_model(model, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Best model saved to: {filepath}")
