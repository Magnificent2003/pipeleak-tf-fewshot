# run_svm_cwt_4cls.py
# CWT(紧凑特征) -> 线性SVM(多分类)；验证集扫描 C 取 Macro-F1 最优，测试集复用同 C
import os, csv, argparse, time
import numpy as np
from joblib import dump
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg

def mc_metrics_from_pred(y_pred: np.ndarray, y_true: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_CWT)
    ap.add_argument("--c_grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir",  type=str, default=cfg.LOG_DIR)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # 读取 CWT 特征与四分类标签（y4_*.npy）
    Xtr = np.load(os.path.join(args.data_root, "Xcwt_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "Xcwt_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "Xcwt_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y4_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y4_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y4_test.npy")).astype(np.int64)

    C_GRID = [float(s) for s in args.c_grid.split(",") if s.strip()]
    best_model, best_C, best_val_m = None, None, None

    for C in C_GRID:
        clf = SVC(kernel="linear", C=C, class_weight="balanced", probability=False, decision_function_shape="ovr", random_state=42)
        clf.fit(Xtr, ytr)

        yhat_va = clf.predict(Xva)
        m_va = mc_metrics_from_pred(yhat_va, yva)
        print(f"C={C:<5} | val acc={m_va['acc']:.4f} macro-F1={m_va['f1']:.4f} "
              f"P_macro={m_va['precision']:.4f} R_macro={m_va['recall']:.4f}")

        if (best_val_m is None) or (m_va["f1"] > best_val_m["f1"]) or \
           (abs(m_va["f1"] - best_val_m["f1"]) < 1e-12 and m_va["acc"] > best_val_m["acc"]):
            best_model, best_C, best_val_m = clf, C, m_va

    print(f"\n[VAL-BEST] C={best_C} | acc={best_val_m['acc']:.4f} macro-F1={best_val_m['f1']:.4f} "
          f"P_macro={best_val_m['precision']:.4f} R_macro={best_val_m['recall']:.4f}")

    # 测试评估（固定最优C）
    yhat_te = best_model.predict(Xte)
    m_te = mc_metrics_from_pred(yhat_te, yte)
    print(f"\n[TEST]  acc={m_te['acc']:.4f} macro-F1={m_te['f1']:.4f} "
          f"P_macro={m_te['precision']:.4f} R_macro={m_te['recall']:.4f}")

    cm_val = confusion_matrix(yva, best_model.predict(Xva), labels=[0,1,2,3])
    cm_te  = confusion_matrix(yte, yhat_te, labels=[0,1,2,3])
    print(f"\n[VAL]  Confusion Matrix (rows=true, cols=pred):\n{cm_val}")
    print(f"\n[TEST] Confusion Matrix (rows=true, cols=pred):\n{cm_te}")

    # 保存模型与日志
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"svm_cwt_4cls_best_{stamp}.joblib")
    dump({"model": best_model, "C": best_C, "classes_": getattr(best_model, "classes_", None)}, ckpt_path)

    log_path = os.path.join(args.log_dir, f"metrics_svm_cwt_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "C","val_acc","val_macro_f1","val_precision_macro","val_recall_macro",
            "test_acc","test_macro_f1","test_precision_macro","test_recall_macro"
        ])
        w.writeheader()
        w.writerow({
            "C": best_C,
            "val_acc": best_val_m["acc"], "val_macro_f1": best_val_m["f1"],
            "val_precision_macro": best_val_m["precision"], "val_recall_macro": best_val_m["recall"],
            "test_acc": m_te["acc"], "test_macro_f1": m_te["f1"],
            "test_precision_macro": m_te["precision"], "test_recall_macro": m_te["recall"],
        })

    print(f"\n[CKPT] {ckpt_path}\n[LOG]  {log_path}")

if __name__ == "__main__":
    main()
