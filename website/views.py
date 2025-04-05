from flask import Blueprint, render_template, request, redirect, flash, session
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')  # Headless backend for Flask
import matplotlib.pyplot as plt
from scipy import stats
import polars as pl
import numpy as np
import joblib
import base64
import shap
import os
import io


views = Blueprint('views', __name__)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

@views.route('/')
def index():
    return redirect('/upload')

@views.route('/upload', methods=['GET', 'POST'])
def upload():
    preview_table = None
    if request.method == 'POST':
        file = request.files.get('dataset')
        # Validate file type
        if not file or not allowed_file(file.filename):
            flash('Please upload a valid CSV file.', 'danger')
            return redirect(request.url)
        
        try:
            # Read CSV using polars
            df = pl.read_csv(file)
            df_preview = df.head(10)

            # Convert to HTML using Polars DataFrame method
            preview_table = df_preview.to_pandas().to_html(classes="table table-bordered table-sm", index=False)
            
            # Save file to uploads
            file.seek(0)
            filename = file.filename
            upload_dir = 'uploads'
            ensure_dir_exists(upload_dir)
            file.save(os.path.join(upload_dir, filename))
            session['original_file_name'] = filename
            
            
        except Exception as e:
            flash(f'Error reading CSV with Polars: {e}', 'danger')
            return redirect(request.url)
        
    return render_template('step1_upload.html', step=1, preview_table=preview_table)

@views.route('/transform', methods=['GET', 'POST'])
def transform():
    label_column = None
    transformed_preview = None
    columns = []

    # Get path to uploaded file
    original_filename = session.get('original_file_name')
    if not original_filename:
        flash("No file uploaded.", "danger")
        return redirect('/upload')
    
    upload_path = os.path.join('uploads', original_filename)
    if not os.path.exists(upload_path):
        flash("Uploaded file not found.", "danger")
        return redirect('/upload')

    # Read file using Polars
    df = pl.read_csv(upload_path)
    columns = df.columns

    if request.method == 'POST':
        label_column = request.form.get('label_column')
        if label_column not in columns:
            flash("Invalid label column selected.", "danger")
        else:
            try:
                # Label encode the selected column
                label_values = df[label_column].to_list()
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(label_values)
                df = df.with_columns(pl.Series(name=label_column, values=encoded))

                # Save transformed dataset to a new file
                transformed_name = f"transformed_{original_filename}"
                transformed_path = os.path.join('uploads', transformed_name)
                ensure_dir_exists('uploads')
                df.write_csv(transformed_path)

                # Save to session for future steps
                session['transformed_file_name'] = transformed_name
                session['label_column'] = label_column

                # Show preview
                transformed_preview = df.head(10).to_pandas().to_html(
                    classes="table table-bordered table-sm", index=False
                )

            except Exception as e:
                flash(f"Failed to encode label column: {e}", "danger")

    return render_template(
        'step2_transform.html',
        step=2,
        columns=columns,
        label_column=label_column,
        transformed_preview=transformed_preview
    )

@views.route('/train')
def train():
    transformed_filename = session.get('transformed_file_name')
    label_column = session.get('label_column')

    if not transformed_filename or not label_column:
        flash("Missing data. Please complete the transform step first.", "danger")
        return redirect('/transform')

    file_path = os.path.join('uploads', transformed_filename)
    if not os.path.exists(file_path):
        flash("Transformed dataset not found.", "danger")
        return redirect('/transform')

    # Load transformed dataset
    df = pl.read_csv(file_path)
    if label_column not in df.columns:
        flash("Label column not found in dataset.", "danger")
        return redirect('/transform')

    # Features and labels
    X = df.drop(label_column).to_numpy()
    y = df[label_column].to_numpy()
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model config
    is_multiclass = n_classes > 2
    objective = "multi:softprob" if is_multiclass else "binary:logistic"
    eval_metric = "mlogloss" if is_multiclass else "logloss"

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        model = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            num_class=n_classes if is_multiclass else None
        )
        model.fit(X_train[train_idx], y_train[train_idx])
        y_val_proba = model.predict_proba(X_train[val_idx])

        if is_multiclass:
            auc = roc_auc_score(
                y_train[val_idx], y_val_proba, multi_class='ovr', average='macro'
            )
        else:
            auc = roc_auc_score(y_train[val_idx], y_val_proba[:, 1])
        aucs.append(auc)

    # AUC stats
    avg_auc = np.mean(aucs)
    stderr = np.std(aucs, ddof=1) / np.sqrt(len(aucs))
    ci = stats.t.interval(0.95, len(aucs)-1, loc=avg_auc, scale=stderr)

    # Final model on full training set
    final_model = XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        num_class=n_classes if is_multiclass else None
    )
    final_model.fit(X_train, y_train)
    y_test_proba = final_model.predict_proba(X_test)

    # Fixed test AUC logic
    if is_multiclass:
        test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
    else:
        test_auc = roc_auc_score(y_test, y_test_proba[:, 1])

    # Save the model
    model_path = os.path.join('downloads', 'final_model.pkl')
    ensure_dir_exists('downloads')
    joblib.dump(final_model, model_path)

    return render_template(
        'step3_train.html',
        step=3,
        fold_aucs=[round(a, 4) for a in aucs],
        avg_auc=round(avg_auc, 4),
        stderr=round(stderr, 4),
        ci_low=round(ci[0], 4),
        ci_high=round(ci[1], 4),
        test_auc=round(test_auc, 4),
        model_path=model_path,
        is_multiclass=is_multiclass,
        n_classes=n_classes
    )

@views.route('/explain')
def explain():
    model_path = os.path.join('downloads', 'final_model.pkl')
    transformed_file = session.get('transformed_file_name')
    label_column = session.get('label_column')

    if not os.path.exists(model_path) or not transformed_file or not label_column:
        flash("Missing data or model. Please complete previous steps.", "danger")
        return redirect('/train')

    # Load model and transformed dataset
    model = joblib.load(model_path)
    df = pl.read_csv(os.path.join('uploads', transformed_file))
    X_df = df.drop(label_column)
    X = X_df.to_numpy()
    y = df[label_column].to_numpy()
    feature_names = X_df.columns

    n_classes = len(np.unique(y))
    is_multiclass = n_classes > 2

    # SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Choose SHAP array
    shap_array = shap_values[:, :, 0] if is_multiclass else shap_values.values

    # Plot using NumPy arrays
    img_bytes = io.BytesIO()
    plt.figure()
    shap.summary_plot(shap_array, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)
    plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    return render_template(
        'step4_explain.html',
        step=4,
        shap_plot=plot_base64,
        is_multiclass=is_multiclass,
        n_classes=n_classes
    )
