from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_aplikasi_anda'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

with open('music_recommender_system.pkl', 'rb') as f:
    model_data = pickle.load(f)

kmeans = model_data['kmeans_model']
df_meta = model_data['df_meta']

if isinstance(model_data['df_features_norm'], pd.DataFrame):
    X_features = model_data['df_features_norm'].values
else:
    X_features = model_data['df_features_norm']

def get_recommendations_and_metrics(input_features, input_cluster, top_n, input_song_idx):
    # Filter dataset hanya pada cluster yang sama
    cluster_indices = df_meta.index[df_meta['cluster'] == input_cluster].tolist()
    X_cluster = X_features[cluster_indices]
    df_cluster = df_meta.iloc[cluster_indices].copy()
    
    # Hitung Jarak/Kemiripan
    eucl_dist = euclidean_distances(input_features.reshape(1, -1), X_cluster)[0]
    eucl_sim = 1 / (1 + eucl_dist)
    cos_sim = cosine_similarity(input_features.reshape(1, -1), X_cluster)[0]
    
    df_cluster['euclidean_score'] = eucl_sim
    df_cluster['cosine_score'] = cos_sim
    
    # Hapus lagu target
    if input_song_idx in df_cluster.index:
        df_cluster = df_cluster.drop(index=input_song_idx)
    
    # Ambil Top-N Dinamis
    top_eucl = df_cluster.sort_values(by='euclidean_score', ascending=False).head(top_n)
    top_cos = df_cluster.sort_values(by='cosine_score', ascending=False).head(top_n)
    
    # Hitung Jaccard, Overlap & Precision
    set_eucl = set(top_eucl.index)
    set_cos = set(top_cos.index)
    
    intersection = len(set_eucl.intersection(set_cos))
    union = len(set_eucl.union(set_cos))
    
    jaccard = intersection / union if union > 0 else 0
    overlap = intersection / top_n if top_n > 0 else 0
    precision = len(top_eucl[top_eucl['cluster'] == input_cluster]) / top_n if top_n > 0 else 0
    
    # ---------------------------------------------------------
    # PERBAIKAN: Hitung ILS Euclidean
    # ---------------------------------------------------------
    top_eucl_features = X_features[top_eucl.index]
    song_labels_eucl = [f"{row['artist']} - {row['song_name']}" for idx, row in top_eucl.iterrows()]
    
    if len(top_eucl_features) > 1:
        sim_matrix_eucl = cosine_similarity(top_eucl_features)
        ils_score_eucl = np.sum(np.triu(sim_matrix_eucl, k=1)) / (top_n * (top_n - 1) / 2)
    else:
        sim_matrix_eucl = np.array([[1.0]])
        ils_score_eucl = 0
        
    # ---------------------------------------------------------
    # PERBAIKAN: Hitung ILS Cosine
    # ---------------------------------------------------------
    top_cos_features = X_features[top_cos.index]
    song_labels_cos = [f"{row['artist']} - {row['song_name']}" for idx, row in top_cos.iterrows()]
    
    if len(top_cos_features) > 1:
        sim_matrix_cos = cosine_similarity(top_cos_features)
        ils_score_cos = np.sum(np.triu(sim_matrix_cos, k=1)) / (top_n * (top_n - 1) / 2)
    else:
        sim_matrix_cos = np.array([[1.0]])
        ils_score_cos = 0
        
    metrics = {
        'top_n': top_n,
        'jaccard': round(jaccard, 3),
        'overlap': round(overlap, 3),
        'precision': round(precision, 3),
        'ils_eucl': round(ils_score_eucl, 3), # ILS Khusus Euclidean
        'ils_cos': round(ils_score_cos, 3),   # ILS Khusus Cosine
        'intersection': intersection,
        'eucl_only': top_n - intersection,
        'cos_only': top_n - intersection
    }
    
    heatmap_eucl = { 'z': sim_matrix_eucl.tolist(), 'x': song_labels_eucl, 'y': song_labels_eucl }
    heatmap_cos = { 'z': sim_matrix_cos.tolist(), 'x': song_labels_cos, 'y': song_labels_cos }
    
    return top_eucl.to_dict('records'), top_cos.to_dict('records'), metrics, heatmap_eucl, heatmap_cos

@app.route('/')
def index():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login gagal. Periksa kembali.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        flash('Akun berhasil dibuat!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    results = None
    error_msg = None
    if request.method == 'POST':
        try:
            song_idx_str = request.form.get('song_index')
            top_n_str = request.form.get('top_n')
            if not song_idx_str: raise ValueError("Pilih lagu dari daftar terlebih dahulu.")
            
            song_idx, top_n = int(song_idx_str), int(top_n_str)
            input_features = X_features[song_idx]
            input_cluster = df_meta.iloc[song_idx]['cluster']
            input_title = f"{df_meta.iloc[song_idx]['artist']} - {df_meta.iloc[song_idx]['song_name']}"
            
            # Terima 5 kembalian dari fungsi
            top_e, top_c, metrics, heatmap_e, heatmap_c = get_recommendations_and_metrics(input_features, input_cluster, top_n, song_idx)
            
            results = {
                'input_title': input_title,
                'euclidean': top_e,
                'cosine': top_c,
                'metrics': metrics,
                'heatmap_eucl': heatmap_e,
                'heatmap_cos': heatmap_c,
                'cluster': int(input_cluster)
            }
        except Exception as e:
            error_msg = str(e)
            
    return render_template('dashboard.html', results=results, songs=df_meta.to_dict('records'), error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)