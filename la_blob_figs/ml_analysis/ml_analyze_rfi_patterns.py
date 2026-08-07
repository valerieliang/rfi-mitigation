"""
Machine learning analysis of RFI patterns in complex SAR data.
Uses unsupervised learning to discover patterns and anomalies.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

def load_tiles(h5_path, skip_range=250):
    """Load tiles and metadata, skipping first N range samples."""
    with h5py.File(h5_path, 'r') as f:
        tiles = f['tiles'][:, :, skip_range:]
        valid = f['valid'][:, :, skip_range:]
        tile_pulse = f['tile_pulse'][:]
        tile_range = f['tile_range'][:]
    return tiles, valid, tile_pulse, tile_range

def extract_complex_features(tiles, valid):
    """
    Extract features from complex data suitable for ML.
    Returns real-valued feature matrix.
    """
    n_tiles, n_pulse, n_range = tiles.shape

    features = []
    feature_names = []

    for i in range(n_tiles):
        tile_features = []

        if not valid[i].any():
            # If no valid data, append zeros
            features.append(np.zeros(20))  # Placeholder size
            continue

        tile_data = tiles[i]
        valid_mask = valid[i]

        # 1. Magnitude statistics (in dB)
        mag = np.abs(tile_data)
        mag_db = 20 * np.log10(mag + 1e-10)
        mag_valid = mag_db[valid_mask]

        tile_features.extend([
            np.mean(mag_valid),           # mean magnitude
            np.std(mag_valid),            # std magnitude
            np.max(mag_valid),            # max magnitude
            np.min(mag_valid),            # min magnitude
            np.percentile(mag_valid, 95), # 95th percentile
        ])

        # 2. Phase statistics
        phase = np.angle(tile_data)
        phase_valid = phase[valid_mask]

        # Phase variance (circular)
        phase_var = 1 - np.abs(np.mean(np.exp(1j * phase_valid)))
        tile_features.extend([
            phase_var,                    # phase variance
            np.std(phase_valid),          # phase std
        ])

        # 3. Spatial variation
        # Range variation (horizontal)
        range_profile = np.mean(mag_db[:, valid_mask[0]], axis=0) if valid_mask[0].any() else np.zeros(1)
        range_std = np.std(range_profile) if len(range_profile) > 1 else 0

        # Azimuth variation (vertical)
        valid_ranges = np.any(valid_mask, axis=0)
        azimuth_profile = np.mean(mag_db[:, valid_ranges], axis=1) if valid_ranges.any() else np.zeros(n_pulse)
        azimuth_std = np.std(azimuth_profile)
        azimuth_peak_to_peak = np.max(azimuth_profile) - np.min(azimuth_profile)

        tile_features.extend([
            range_std,
            azimuth_std,
            azimuth_peak_to_peak,
        ])

        # 4. Periodicity indicators (FFT based)
        # Azimuth FFT to detect pulse-to-pulse periodicity
        if valid_ranges.any():
            azimuth_fft = np.fft.fft(azimuth_profile)
            azimuth_power = np.abs(azimuth_fft[:len(azimuth_fft)//2])**2

            # Find dominant frequency (excluding DC)
            if len(azimuth_power) > 1:
                dominant_freq_idx = np.argmax(azimuth_power[1:]) + 1
                dominant_freq_power = azimuth_power[dominant_freq_idx]
                dc_power = azimuth_power[0]
                periodicity_ratio = dominant_freq_power / (dc_power + 1e-10)
            else:
                dominant_freq_idx = 0
                periodicity_ratio = 0
        else:
            dominant_freq_idx = 0
            periodicity_ratio = 0

        tile_features.extend([
            periodicity_ratio,
            dominant_freq_idx,
        ])

        # 5. Real/Imaginary balance
        real_part = np.real(tile_data[valid_mask])
        imag_part = np.imag(tile_data[valid_mask])

        real_imag_ratio = np.std(real_part) / (np.std(imag_part) + 1e-10)
        real_imag_corr = np.corrcoef(real_part, imag_part)[0, 1]

        tile_features.extend([
            real_imag_ratio,
            real_imag_corr,
        ])

        # 6. Kurtosis (measure of outliers/spikiness)
        mag_kurtosis = np.mean((mag_valid - np.mean(mag_valid))**4) / (np.std(mag_valid)**4 + 1e-10)

        tile_features.append(mag_kurtosis)

        # 7. Invalid data fraction
        invalid_frac = 1.0 - np.mean(valid_mask)
        tile_features.append(invalid_frac)

        # 8. Temporal gradient (change along azimuth)
        azimuth_gradient = np.mean(np.abs(np.diff(azimuth_profile)))
        tile_features.append(azimuth_gradient)

        # 9. Peak count in azimuth profile (spikiness)
        # Count peaks above mean + 1 std
        threshold = np.mean(azimuth_profile) + np.std(azimuth_profile)
        peak_count = np.sum(azimuth_profile > threshold)
        tile_features.append(peak_count)

        features.append(tile_features)

    feature_names = [
        'mag_mean', 'mag_std', 'mag_max', 'mag_min', 'mag_p95',
        'phase_var', 'phase_std',
        'range_std', 'azimuth_std', 'azimuth_p2p',
        'periodicity_ratio', 'dominant_freq_idx',
        'real_imag_ratio', 'real_imag_corr',
        'mag_kurtosis', 'invalid_frac',
        'azimuth_gradient', 'peak_count'
    ]

    return np.array(features), feature_names

def apply_pca(features, n_components=5):
    """Apply PCA for dimensionality reduction and pattern discovery."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, pca, scaler

def apply_ica(features, n_components=5):
    """Apply ICA to find independent components (different RFI sources)."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    features_ica = ica.fit_transform(features_scaled)

    return features_ica, ica

def cluster_tiles(features, method='kmeans', n_clusters=4):
    """Cluster tiles based on features."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(features_scaled)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=3)
        labels = clusterer.fit_predict(features_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, clusterer

def detect_anomalies(features):
    """Detect anomalous tiles using Isolation Forest."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features_scaled)
    anomaly_scores = iso_forest.score_samples(features_scaled)

    # -1 = anomaly, 1 = normal
    return anomaly_labels, anomaly_scores, iso_forest

def plot_ml_analysis(pol, features, feature_names, tile_pulse, tile_range, output_dir):
    """Create comprehensive ML analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n=== ML Analysis for {pol} ===")
    print(f"Feature matrix shape: {features.shape}")

    # 1. PCA Analysis
    print("Running PCA...")
    features_pca, pca, scaler = apply_pca(features, n_components=5)

    # 2. ICA Analysis
    print("Running ICA...")
    features_ica, ica = apply_ica(features, n_components=5)

    # 3. K-means clustering
    print("Running K-means clustering...")
    labels_kmeans, _ = cluster_tiles(features, method='kmeans', n_clusters=4)

    # 4. Anomaly detection
    print("Running anomaly detection...")
    anomaly_labels, anomaly_scores, _ = detect_anomalies(features)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))

    # Plot 1: PCA - First 2 components
    ax1 = plt.subplot(3, 4, 1)
    scatter1 = ax1.scatter(features_pca[:, 0], features_pca[:, 1],
                           c=range(len(features_pca)), cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title(f'{pol} PCA Projection (colored by tile index)')
    plt.colorbar(scatter1, ax=ax1, label='Tile Index')
    ax1.grid(True, alpha=0.3)

    # Plot 2: PCA explained variance
    ax2 = plt.subplot(3, 4, 2)
    ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, alpha=0.7)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title(f'{pol} PCA Explained Variance')
    ax2.grid(True, alpha=0.3)

    # Plot 3: PCA - PC1 vs PC3
    ax3 = plt.subplot(3, 4, 3)
    scatter3 = ax3.scatter(features_pca[:, 0], features_pca[:, 2],
                           c=range(len(features_pca)), cmap='viridis', s=50, alpha=0.7)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
    ax3.set_title(f'{pol} PCA: PC1 vs PC3')
    plt.colorbar(scatter3, ax=ax3, label='Tile Index')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature importance (PCA loadings for PC1)
    ax4 = plt.subplot(3, 4, 4)
    pc1_loadings = np.abs(pca.components_[0])
    sorted_idx = np.argsort(pc1_loadings)[::-1][:10]  # Top 10
    ax4.barh(range(len(sorted_idx)), pc1_loadings[sorted_idx], alpha=0.7)
    ax4.set_yticks(range(len(sorted_idx)))
    ax4.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax4.set_xlabel('Absolute Loading')
    ax4.set_title(f'{pol} PC1 Feature Importance')
    ax4.grid(True, alpha=0.3)

    # Plot 5: ICA components 1 vs 2
    ax5 = plt.subplot(3, 4, 5)
    scatter5 = ax5.scatter(features_ica[:, 0], features_ica[:, 1],
                           c=range(len(features_ica)), cmap='plasma', s=50, alpha=0.7)
    ax5.set_xlabel('IC1')
    ax5.set_ylabel('IC2')
    ax5.set_title(f'{pol} ICA Projection (Independent Sources)')
    plt.colorbar(scatter5, ax=ax5, label='Tile Index')
    ax5.grid(True, alpha=0.3)

    # Plot 6: ICA component evolution
    ax6 = plt.subplot(3, 4, 6)
    for i in range(min(3, features_ica.shape[1])):
        ax6.plot(features_ica[:, i], label=f'IC{i+1}', alpha=0.7)
    ax6.set_xlabel('Tile Index')
    ax6.set_ylabel('IC Value')
    ax6.set_title(f'{pol} ICA Component Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: K-means clustering (on PCA space)
    ax7 = plt.subplot(3, 4, 7)
    scatter7 = ax7.scatter(features_pca[:, 0], features_pca[:, 1],
                           c=labels_kmeans, cmap='tab10', s=50, alpha=0.7)
    ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax7.set_title(f'{pol} K-means Clusters (k=4)')
    plt.colorbar(scatter7, ax=ax7, label='Cluster')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Cluster distribution over tiles
    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(range(len(labels_kmeans)), labels_kmeans, c=labels_kmeans,
                cmap='tab10', s=50, alpha=0.7)
    ax8.set_xlabel('Tile Index')
    ax8.set_ylabel('Cluster ID')
    ax8.set_title(f'{pol} Cluster Assignment by Tile')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Anomaly detection
    ax9 = plt.subplot(3, 4, 9)
    colors = ['red' if label == -1 else 'blue' for label in anomaly_labels]
    ax9.scatter(features_pca[:, 0], features_pca[:, 1],
                c=colors, s=50, alpha=0.7)
    ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax9.set_title(f'{pol} Anomaly Detection (red=anomaly)')
    ax9.grid(True, alpha=0.3)

    # Plot 10: Anomaly scores
    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(anomaly_scores, 'o-', markersize=4, alpha=0.7)
    anomaly_threshold = np.percentile(anomaly_scores, 10)
    ax10.axhline(anomaly_threshold, color='r', linestyle='--', alpha=0.5, label='10th percentile')
    ax10.set_xlabel('Tile Index')
    ax10.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax10.set_title(f'{pol} Anomaly Scores')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Plot 11: Spatial distribution of clusters
    ax11 = plt.subplot(3, 4, 11)
    scatter11 = ax11.scatter(tile_range, tile_pulse, c=labels_kmeans,
                             cmap='tab10', s=100, alpha=0.7)
    ax11.set_xlabel('Range Start')
    ax11.set_ylabel('Pulse Start')
    ax11.set_title(f'{pol} Spatial Cluster Distribution')
    plt.colorbar(scatter11, ax=ax11, label='Cluster')
    ax11.grid(True, alpha=0.3)

    # Plot 12: Spatial distribution of anomalies
    ax12 = plt.subplot(3, 4, 12)
    colors_spatial = ['red' if label == -1 else 'blue' for label in anomaly_labels]
    ax12.scatter(tile_range, tile_pulse, c=colors_spatial, s=100, alpha=0.7)
    ax12.set_xlabel('Range Start')
    ax12.set_ylabel('Pulse Start')
    ax12.set_title(f'{pol} Spatial Anomaly Distribution')
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'ml_analysis_{pol}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary statistics
    print(f"\n{pol} ML Summary:")
    print(f"  PCA total explained variance (5 PCs): {np.sum(pca.explained_variance_ratio_):.1%}")
    print(f"  Number of clusters: {len(np.unique(labels_kmeans))}")
    print(f"  Cluster distribution: {np.bincount(labels_kmeans)}")
    print(f"  Number of anomalies: {np.sum(anomaly_labels == -1)} / {len(anomaly_labels)}")
    print(f"  Anomaly percentage: {100 * np.sum(anomaly_labels == -1) / len(anomaly_labels):.1f}%")

    return {
        'features_pca': features_pca,
        'features_ica': features_ica,
        'pca': pca,
        'ica': ica,
        'scaler': scaler,
        'labels_kmeans': labels_kmeans,
        'anomaly_labels': anomaly_labels,
        'anomaly_scores': anomaly_scores
    }

def main():
    hh_path = Path('la_blob_figs/raw_tiles_A_HH.h5')
    hv_path = Path('la_blob_figs/raw_tiles_A_HV.h5')
    output_dir = Path('la_blob_figs/ml_analysis')

    skip_range = 250

    # Analyze HH
    print(f"Loading HH polarization (skipping first {skip_range} range samples)...")
    tiles_hh, valid_hh, pulse_hh, range_hh = load_tiles(hh_path, skip_range=skip_range)
    print(f"HH tiles shape after skip: {tiles_hh.shape}")

    print("Extracting features from HH...")
    features_hh, feature_names = extract_complex_features(tiles_hh, valid_hh)

    hh_results = plot_ml_analysis('HH', features_hh, feature_names, pulse_hh, range_hh, output_dir)

    # Analyze HV
    print(f"\nLoading HV polarization (skipping first {skip_range} range samples)...")
    tiles_hv, valid_hv, pulse_hv, range_hv = load_tiles(hv_path, skip_range=skip_range)
    print(f"HV tiles shape after skip: {tiles_hv.shape}")

    print("Extracting features from HV...")
    features_hv, _ = extract_complex_features(tiles_hv, valid_hv)

    hv_results = plot_ml_analysis('HV', features_hv, feature_names, pulse_hv, range_hv, output_dir)

    # Cross-polarization ML comparison
    print("\n=== Cross-Polarization ML Comparison ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # PCA comparison
    axes[0, 0].scatter(hh_results['features_pca'][:, 0], hh_results['features_pca'][:, 1],
                       alpha=0.6, label='HH', s=50)
    axes[0, 0].scatter(hv_results['features_pca'][:, 0], hv_results['features_pca'][:, 1],
                       alpha=0.6, label='HV', s=50)
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA Comparison: HH vs HV')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ICA comparison
    axes[0, 1].scatter(hh_results['features_ica'][:, 0], hh_results['features_ica'][:, 1],
                       alpha=0.6, label='HH', s=50)
    axes[0, 1].scatter(hv_results['features_ica'][:, 0], hv_results['features_ica'][:, 1],
                       alpha=0.6, label='HV', s=50)
    axes[0, 1].set_xlabel('IC1')
    axes[0, 1].set_ylabel('IC2')
    axes[0, 1].set_title('ICA Comparison: HH vs HV')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cluster comparison
    axes[0, 2].plot(hh_results['labels_kmeans'], 'o-', alpha=0.7, label='HH', markersize=4)
    axes[0, 2].plot(hv_results['labels_kmeans'], 's-', alpha=0.7, label='HV', markersize=4)
    axes[0, 2].set_xlabel('Tile Index')
    axes[0, 2].set_ylabel('Cluster ID')
    axes[0, 2].set_title('Cluster Assignment Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Anomaly score comparison
    axes[1, 0].plot(hh_results['anomaly_scores'], alpha=0.7, label='HH')
    axes[1, 0].plot(hv_results['anomaly_scores'], alpha=0.7, label='HV')
    axes[1, 0].set_xlabel('Tile Index')
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].set_title('Anomaly Score Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Feature correlation between HH and HV
    axes[1, 1].scatter(features_hh[:, 0], features_hv[:, 0], alpha=0.6, s=50)
    axes[1, 1].set_xlabel('HH Mean Magnitude (dB)')
    axes[1, 1].set_ylabel('HV Mean Magnitude (dB)')
    axes[1, 1].set_title('Feature Correlation: Mean Magnitude')
    axes[1, 1].grid(True, alpha=0.3)

    # Periodicity comparison
    axes[1, 2].scatter(features_hh[:, 10], features_hv[:, 10], alpha=0.6, s=50)
    axes[1, 2].set_xlabel('HH Periodicity Ratio')
    axes[1, 2].set_ylabel('HV Periodicity Ratio')
    axes[1, 2].set_title('Feature Correlation: Periodicity')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'cross_pol_ml_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    print("\n=== ML Analysis Complete ===")
    print(f"\nKey feature names (total {len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")

if __name__ == '__main__':
    main()
