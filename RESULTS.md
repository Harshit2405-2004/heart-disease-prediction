# Results

Every number here comes from a notebook output in `notebooks/`. A cell marked *not run yet* has to be filled in after running the "Can the score go higher?" section of Part 1.

---

## Part 1 - Clinical model (UCI Heart Disease)

### Data
| | |
|---|---|
| Hospitals | Cleveland 303, Hungary 294, VA Long Beach 200, Switzerland 123 |
| Patients after removing 2 duplicates | 918 |
| Disease share | 55% overall: Cleveland 46%, Hungary 36%, VA 75%, Switzerland 94% |
| Features | 11: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope (`ca`, `thal` dropped: missing for most patients outside Cleveland) |
| Hidden missing values | chol = 0 for 172 patients, trestbps = 0 for 1 patient, both set to missing |
| Split | 734 train / 184 test, stratified by hospital and target |

### Model comparison
| Model | CV ROC AUC (5-fold, train) | Test accuracy | Test recall | Test precision | Test F1 | Test ROC AUC | False negatives |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.875 +- 0.020 | 0.810 | 0.853 | 0.813 | 0.833 | 0.875 | 15 |
| Random Forest (default) | 0.864 +- 0.039 | 0.810 | 0.873 | 0.802 | 0.836 | 0.866 | 13 |
| Random Forest (tuned) | 0.877 | 0.799 | 0.863 | 0.793 | 0.826 | 0.875 | 14 |
| XGBoost (tuned) | 0.878 | 0.815 | 0.882 | 0.804 | 0.841 | 0.885 | 12 |
| XGBoost calibrated, threshold 0.5 | | 0.810 | 0.873 | 0.802 | 0.836 | 0.880 | 13 |
| **XGBoost calibrated, threshold 0.35 (final)** | | **0.810** | **0.931** | **0.772** | **0.844** | **0.880** | **7** |

- Tuned Random Forest parameters: 300 trees, max_depth 5, min_samples_leaf 6, max_features log2.
- Final model chosen by CV ROC AUC (XGBoost 0.878, ahead by 0.001 - 0.003, which is within noise).
- Calibration (sigmoid, 5-fold): Brier score 0.1332 before, 0.1353 after.
- Threshold 0.35: the highest threshold with at least 90% recall on out-of-fold training predictions (out-of-fold recall 0.909 at 0.35).
- Final confusion matrix (test): 54 true negatives, 28 false positives, 7 false negatives, 95 true positives.

### Per hospital (final model, threshold 0.35)
| Hospital | Patients | Disease share | Accuracy | Recall | ROC AUC |
|---|---|---|---|---|---|
| Cleveland | 61 | 0.459 | 0.836 | 1.000 | 0.942 |
| Hungary | 58 | 0.362 | 0.776 | 0.810 | 0.856 |
| VA Long Beach | 40 | 0.750 | 0.750 | 0.933 | 0.733 |
| Switzerland | 25 | 0.920 | 0.920 | 0.957 | 0.783 (only 2 healthy patients, not meaningful) |

### Permutation importance (drop in test ROC AUC)
cp 0.085, oldpeak 0.032, sex 0.018, chol 0.012, exang 0.011, thalach 0.007, age 0.005, slope 0.001, trestbps / fbs / restecg ~0

### "Can the score go higher?" section
| | Result |
|---|---|
| Bootstrap 95% CI, test accuracy | *not run yet* |
| Bootstrap 95% CI, test ROC AUC | *not run yet* |
| Repeated CV (5 x 5) AUC: LR / RF / XGB / LR+XGB soft vote | *not run yet* |
| Cleveland only, AUC with 11 vs 13 features (+ ca, thal) | *not run yet* |
| Hospital as a feature (leak) vs without | *not run yet* |

---

## Part 2 - Chest X-ray model (NIH ChestX-ray14, cardiomegaly)

### Data
| | |
|---|---|
| Source | NIH ChestX-ray14: 112,120 images, 2,776 with cardiomegaly |
| Subset | 11,104 images = all 2,776 cardiomegaly + 8,328 negatives (half "No Finding", half other findings), 6,778 patients |
| Split | train 7,310 / val 793 / test 3,001. Test = official NIH test list (by patient), val = 10% of training patients. No patient overlap (checked) |
| Cardiomegaly share | train 0.207 / val 0.242 / test 0.356 |
| AP share of images | train 0.351 / val 0.330 / test 0.582 |
| Image size | stored 256 x 256, model input 224 x 224 center crop |

### Training
| | |
|---|---|
| Model | DenseNet-121, ImageNet weights, 1 output (6.95M parameters) |
| Setup | AdamW lr 1e-4, weight decay 1e-4 (not on bias / norm), 10% warmup + linear decay, fp16, gradient clipping 1.0, pos_weight 3.825, batch 32 |
| Augmentation | random resized crop (scale 0.85 - 1), rotation +-7 deg, brightness / contrast 0.1. **No flip** |
| Epochs | 5 (early stopping, patience 2). Best epoch 3 |

| Epoch | Train loss | Val loss | Val ROC AUC |
|---|---|---|---|
| 1 | 0.895 | 0.716 | 0.905 |
| 2 | 0.658 | 0.645 | 0.915 |
| **3** | **0.547** | **0.581** | **0.926** |
| 4 | 0.462 | 0.678 | 0.914 |
| 5 | 0.386 | 0.890 | 0.918 |

### Test results (threshold 0.45, chosen on validation for 90% recall)
| Metric | Value | 95% CI (500 patient resamples) |
|---|---|---|
| ROC AUC | **0.880** | 0.858 - 0.898 |
| Recall | 0.825 | 0.779 - 0.862 |
| False positive rate | 0.259 | 0.235 - 0.284 |
| Precision | 0.638 | |
| Accuracy | 0.771 | |
| F1 (cardiomegaly) | 0.719 | |
| Confusion matrix | TN 1431, FP 501, FN 187, TP 882 | |

### By view
| | Val PA | Val AP | Test PA | Test AP |
|---|---|---|---|---|
| Images (positives) | 531 (121) | 262 (71) | 1,254 (427) | 1,747 (642) |
| ROC AUC | 0.946 | 0.895 | 0.925 | 0.846 |
| Recall at 0.45 | 0.967 | 0.831 | 0.904 | 0.773 |
| False positive rate | 0.224 | 0.204 | 0.244 | 0.271 |

**Why recall fell from 0.917 (val) to 0.825 (test):** -3.0 points from the view mix (60% of test positives are AP, against 37% in val), and -6.1 points from recall falling inside each view (an optimistic, small validation set).

### Per-view thresholds (chosen on validation, reference only, not used in the app)
| Test | Rule | Recall | False positive rate | Precision |
|---|---|---|---|---|
| PA | single 0.45 | 0.904 | 0.244 | 0.656 |
| PA | per view 0.70 | 0.859 | 0.152 | 0.744 |
| AP | single 0.45 | 0.773 | 0.271 | 0.624 |
| AP | per view 0.30 | 0.849 | 0.376 | 0.568 |

### Precision at other prevalences (same recall 0.825 and false positive rate 0.259)
| Prevalence | Precision | False alarms per true find |
|---|---|---|
| 2.5% (full NIH data) | 0.075 | 12.4 |
| 5% | 0.143 | 6.0 |
| 10% | 0.261 | 2.8 |
| 35.6% (this test set) | 0.638 | 0.6 |

### Error analysis
| | n | AP share | With effusion | With consolidation | With infiltration | With atelectasis |
|---|---|---|---|---|---|---|
| Missed (false negatives) | 187 | 0.781 | 0.636 | 0.166 | 0.305 | 0.193 |
| Detected (true positives) | 882 | 0.562 | 0.378 | 0.084 | 0.280 | 0.161 |

- Miss rate: AP 23% (146 of 642), PA 10% (41 of 427).
- False positive rate on "No Finding" negatives 0.255, on negatives with other findings 0.262.
- The 6 most confident false positives all score >= 0.99 (3 labelled "No Finding"). The 6 most confident false negatives all score <= 0.02 (4 of them PA films). These are candidates for label noise, to be reviewed by a radiologist.

### Other
| | |
|---|---|
| Test-time augmentation (5 shifted crops) | ROC AUC val 0.9264 -> 0.9314, test 0.8801 -> 0.8836. Not used (5x latency, gain inside the CI) |
| CPU latency, single image | mean 126.6 ms, p95 152.5 ms |
| Versions | torch 2.11.0, torchvision 0.26.0 (Part 2), scikit-learn 1.6.1 (Part 1) |

---

## Figures (`assets/`)
| File | What it shows |
|---|---|
| `clinical_confusion_baseline.png` | Logistic regression, threshold 0.5, test |
| `clinical_confusion_final.png` | Final XGBoost calibrated, threshold 0.35, test |
| `clinical_roc.png` | Test ROC curves of the 4 models |
| `clinical_threshold.png` | Recall / precision / accuracy vs threshold, out-of-fold on train |
| `clinical_calibration.png` | Reliability plot before / after calibration |
| `clinical_importance.png` | Permutation importance |
| `xray_augmentation.png` | Training images after augmentation |
| `xray_training_curves.png` | Train / val loss and val AUC per epoch |
| `xray_confusion.png` | Test confusion matrix, threshold 0.45 |
| `xray_roc.png` | Test ROC curve |
| `xray_gradcam.png` | Grad-CAM, 3 confident detections and 3 misses |
| `xray_errors.png` | 6 most confident false positives and false negatives |
| `app_screenshot.png` | *add after running the app with the real models* |
