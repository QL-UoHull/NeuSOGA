import os
import urllib.request
import zipfile
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from skimage.measure import approximate_polygon
from skimage.feature import peak_local_max
import math
from numba import njit, prange
import torch
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. PLATONIST MATH ENGINE (System 2: Logic)
# ==========================================
@njit
def fact(n):
    res = 1
    for i in range(1, n + 1): res *= i
    return res

@njit
def comb_numba(n, k):
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    return fact(n) // (fact(k) * fact(n - k))

@njit
def FF_scalar(s, n):
    if n == 0: return 0.5 if s == 0.0 else (1.0 if s > 0.0 else 0.0)
    else: return (s / n) * FF_scalar(s, n-1) + (1.0 - s/n) * FF_scalar(s-1, n-1)

@njit
def H_scalar(s, n):
    if n == 0: return FF_scalar(s, 0)
    else: return FF_scalar(n * (s + 1.0) / 2.0, n)

@njit
def Lxy_scalar(x, y, slope, n):
    if slope * x > y: return ((slope * x - y)**(2*n)) / (fact(2*n) * (slope**n))
    return 0.0

@njit
def Lxy00_scalar(x, y, slope, n):
    AA = 0.0
    for k in range(1, n+1):
        AA += ((-1.0)**(n+k) * (x**(n-k)) * (y**(n+k))) / (fact(n-k) * fact(n+k) * (slope**k))
    return AA

@njit
def L_corner_inter_scalar(x, y, slope, n):
    slope = abs(slope)
    if n < 1: return 0.0
    if y < min(0.0, slope * x):
        return Lxy_scalar(x, y, slope, n) if x <= 0.0 else Lxy00_scalar(x, y, slope, n)
    return 0.0

@njit
def U_Angle_inter_scalar(x, y, slope, delta, n):
    FF_val = 0.0
    for k in range(0, n+1):
        FF_val += (-1.0)**k * comb_numba(n, k) * L_corner_inter_scalar(x + (n - 2*k)*delta, y, slope, n)
    return FF_val

@njit
def impAngle4BigSlope_scalar(x, y, slope, delta, n):
    GG = 0.0
    for k in range(0, n+1):
        GG += (-1.0)**k * comb_numba(n, k) * U_Angle_inter_scalar(x, y - (n - 2*k)*delta, slope, delta, n)
    return GG / ((4.0 * delta**2)**n)

@njit
def Square_Angle_inter_scalar(x, y, slope, delta, n):
    if slope > 1.0: return impAngle4BigSlope_scalar(x, y, slope, delta, n)
    else:
        return H_scalar(-x/(n*delta), n) * H_scalar(-y/(n*delta), n) - impAngle4BigSlope_scalar(y, x, 1.0/slope, delta, n)

@njit
def Point_imp_scalar(x, y, x0, y0, slope, delta, n):
    if math.isinf(slope): return 0.0
    if abs(slope) < 1e-32: return H_scalar((x0-x)/(n*delta), n) * H_scalar((y0-y)/(n*delta), n)
    if slope >= 1e-32: return Square_Angle_inter_scalar(x-x0, y-y0, slope, delta, n)
    else: return Square_Angle_inter_scalar(-(x-x0), y-y0, -slope, delta, n)

@njit
def LineSeg_imp_scalar(x, y, x0, y0, x1, y1, delta, n):
    y01, x01 = y1 - y0, x1 - x0
    if x01 == 0.0: return 0.0
    slope = y01 / x01
    if slope == 0.0:
        if x01 > 0.0: return Point_imp_scalar(x, y, x1, y1, slope, delta, n) - Point_imp_scalar(x, y, x0, y0, slope, delta, n)
        else: return Point_imp_scalar(x, y, x0, y0, slope, delta, n) - Point_imp_scalar(x, y, x1, y1, slope, delta, n)
    if y01 > 0.0: return Point_imp_scalar(x, y, x1, y1, slope, delta, n) - Point_imp_scalar(x, y, x0, y0, slope, delta, n)
    else: return Point_imp_scalar(x, y, x0, y0, slope, delta, n) - Point_imp_scalar(x, y, x1, y1, slope, delta, n)

@njit(parallel=True)
def ImpSpline2D(xx, yy, data, delta=0.01, n=2):
    rows, cols = xx.shape
    imp2DFun = np.zeros((rows, cols), dtype=np.float64)
    NumP = data.shape[0]

    for i in prange(rows):
        for j in range(cols):
            x, y = xx[i, j], yy[i, j]
            val = 0.0
            for p in range(NumP):
                x0, y0 = data[p, 0], data[p, 1]
                x1, y1 = data[(p+1)%NumP, 0], data[(p+1)%NumP, 1]
                xDir = x0 - x1
                if xDir > 0: sign_val = 1.0
                elif xDir < 0: sign_val = -1.0
                else: sign_val = 0.0
                if sign_val != 0.0:
                    val += sign_val * LineSeg_imp_scalar(x, y, x0, y0, x1, y1, delta, n)
            imp2DFun[i, j] = val
    return imp2DFun


# ==========================================
# 2. DATASET DOWNLOADER & SAM INITIALIZATION
# ==========================================
def download_modelnet40():
    url = "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"
    zip_path = "modelnet40.zip"
    extract_folder = "modelnet40_data"
    if not os.path.exists(extract_folder):
        print("Downloading ModelNet40...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Extracting ModelNet40...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)
    return os.path.join(extract_folder, "modelnet40_ply_hdf5_2048")

def initialize_sam():
    print("Loading Meta Segment Anything Model (SAM)...")
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


# ==========================================
# 3. GEOMETRY UTILS & HYBRID EXTRACTION
# ==========================================
def project_to_arbitrary_plane(pc_3d, direction):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    z_vec = direction
    if abs(z_vec[1]) > 0.99: up = np.array([1.0, 0.0, 0.0])
    else: up = np.array([0.0, 1.0, 0.0])

    x_vec = np.cross(up, z_vec)
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(z_vec, x_vec)
    y_vec /= np.linalg.norm(y_vec)

    proj_matrix = np.column_stack((x_vec, y_vec))
    pc_2d = np.dot(pc_3d, proj_matrix)
    return pc_2d

def process_contour_hybrid(c, x_min, x_max, y_min, y_max, img_size, is_hole=False):
    coarse_x = gaussian_filter(c[:, 0].astype(float), sigma=6.0, mode='wrap')
    coarse_y = gaussian_filter(c[:, 1].astype(float), sigma=6.0, mode='wrap')
    curve_coarse = np.column_stack((coarse_x, coarse_y))

    fine_x = gaussian_filter(c[:, 0].astype(float), sigma=1.5, mode='wrap')
    fine_y = gaussian_filter(c[:, 1].astype(float), sigma=1.5, mode='wrap')
    curve_fine = np.column_stack((fine_x, fine_y))

    deviation = np.linalg.norm(curve_coarse - curve_fine, axis=1)
    missed_mask = (deviation > 2.0).astype(float)
    blend_weight = gaussian_filter(missed_mask, sigma=3.0, mode='wrap')
    blend_weight = np.clip(blend_weight, 0, 1).reshape(-1, 1)

    hybrid_curve = curve_coarse * (1.0 - blend_weight) + curve_fine * blend_weight
    sampled = approximate_polygon(hybrid_curve, tolerance=2.5)

    if np.linalg.norm(sampled[0] - sampled[-1]) < 1e-5:
        sampled = sampled[:-1]

    def to_world(pts):
        x = x_min + (pts[:, 0] / (img_size - 1)) * (x_max - x_min)
        y = y_min + ((img_size - 1 - pts[:, 1]) / (img_size - 1)) * (y_max - y_min)
        return np.column_stack((x, y))

    poly = to_world(sampled)
    hc_world = to_world(hybrid_curve)

    # SHOELACE FORMULA: Determine orientation
    x_pts, y_pts = poly[:, 0], poly[:, 1]
    signed_area = np.sum(x_pts[:-1] * y_pts[1:] - x_pts[1:] * y_pts[:-1]) + (x_pts[-1] * y_pts[0] - x_pts[0] * y_pts[-1])

    # EXTERNAL boundaries must be CCW (Additive, signed_area > 0)
    # INTERNAL holes must be CW (Subtractive, signed_area < 0)
    if not is_hole and signed_area < 0:
        poly = poly[::-1]
        hc_world = hc_world[::-1]
    elif is_hole and signed_area > 0:
        poly = poly[::-1]
        hc_world = hc_world[::-1]

    return poly, hc_world

def holistic_hybrid_extraction(pc_2d, predictor, img_size=512, margin=0.15):
    x_min, x_max = pc_2d[:, 0].min() - margin, pc_2d[:, 0].max() + margin
    y_min, y_max = pc_2d[:, 1].min() - margin, pc_2d[:, 1].max() + margin

    # 1. NEW HOLE-PRESERVING MASKING
    img_density = np.zeros((img_size, img_size), dtype=np.float32)
    for p in pc_2d:
        px = int(((p[0] - x_min) / (x_max - x_min)) * (img_size - 1))
        py = int(((p[1] - y_min) / (y_max - y_min)) * (img_size - 1))
        cv2.circle(img_density, (px, py), radius=3, color=255.0, thickness=-1)

    # Blur to bridge sparse gaps, but don't over-blur macro-holes
    blurred_density = cv2.GaussianBlur(img_density, (15, 15), sigmaX=4, sigmaY=4)
    _, mask_thresh = cv2.threshold(blurred_density, 20, 255, cv2.THRESH_BINARY)
    mask_thresh = mask_thresh.astype(np.uint8)

    # MORPH_CLOSE: Closes micro-gaps (15px) but leaves structural holes completely intact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_solid = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel)

    mask_flipped = cv2.flip(mask_solid, 0)
    img_rgb = cv2.cvtColor(mask_flipped, cv2.COLOR_GRAY2RGB)
    predictor.set_image(img_rgb)

    # 2. DISTANCE TRANSFORM & NODE DISCOVERY
    dist_transform = cv2.distanceTransform(mask_flipped, cv2.DIST_L2, 5)
    dist_smooth = gaussian_filter(dist_transform, sigma=3.0)

    local_max_coords = peak_local_max(
        dist_smooth,
        min_distance=40,
        threshold_abs=0.2 * dist_transform.max()
    )

    if len(local_max_coords) == 0:
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        prompts = np.array([[max_loc[0], max_loc[1]]])
    else:
        prompts = np.array([[c[1], c[0]] for c in local_max_coords])

    labels = np.ones(len(prompts))
    masks, _, _ = predictor.predict(point_coords=prompts, point_labels=labels, multimask_output=False)

    features = []
    colors = ['#FF9900', '#3366CC', '#109618', '#990099', '#DC3912', '#22AA99', '#994499', '#316395']
    mask_binary = np.zeros((img_size, img_size), dtype=np.uint8)

    if len(masks) > 0:
        mask_binary = (masks[0] * 255).astype(np.uint8)

        # RETR_CCOMP: Retrieves both outer boundaries AND inner holes
        contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is not None:
            for idx, c in enumerate(contours):
                if cv2.contourArea(c) < 150:
                    continue

                # Check hierarchy: If parent != -1, this contour is an internal hole
                is_hole = hierarchy[0][idx][3] != -1

                c = c.reshape(-1, 2)
                diffs = np.sum(np.abs(np.diff(c, axis=0)), axis=1)
                c = c[np.insert(diffs > 0, 0, True)]

                if len(c) >= 10:
                    poly, hc_world = process_contour_hybrid(c, x_min, x_max, y_min, y_max, img_size, is_hole)

                    # Color holes gray for visual distinction
                    color = '#666666' if is_hole else colors[idx % len(colors)]
                    features.append({
                        'polygon': poly,
                        'hybrid_curve': hc_world,
                        'color': color,
                        'name': f'Structure_{idx}',
                        'is_hole': is_hole
                    })

    vis_data = {
        'dist_transform': dist_transform,
        'nodes': prompts,
        'mask_binary': mask_binary
    }

    return features, (x_min, x_max, y_min, y_max), vis_data


# ==========================================
# 4. ORCHESTRATOR: BATCH PROCESSING 40 CLASSES
# ==========================================
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair",
    "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box",
    "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand",
    "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs",
    "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]

def run_neuro_symbolic_pipeline():
    print("Pre-compiling Platonist Math Engine...")
    _ = ImpSpline2D(np.zeros((2,2)), np.zeros((2,2)), np.array([[0,0],[1,0],[0,1]]), delta=0.01)

    dataset_folder = download_modelnet40()
    h5_file = os.path.join(dataset_folder, "ply_data_train0.h5")
    predictor = initialize_sam()

    print(f"\nOpening {h5_file}...")
    with h5py.File(h5_file, 'r') as f:
        point_clouds = f['data'][:]
        labels = f['label'][:]

    test_objects = []
    for i in range(40):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) > 0:
            test_objects.append((MODELNET40_CLASSES[i].capitalize(), class_indices[0]))

    output_dir = "robustness_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFound {len(test_objects)} unique classes. Results will be saved to '{output_dir}/'")

    proj_direction = [1, 1, 1]

    dir_norm = np.array(proj_direction) / np.linalg.norm(proj_direction)
    elev_angle = np.degrees(np.arcsin(dir_norm[2]))
    azim_angle = np.degrees(np.arctan2(dir_norm[1], dir_norm[0]))

    for idx, (obj_name, target_idx) in enumerate(test_objects):
        print(f"\n[{idx+1}/{len(test_objects)}] PROCESSING OBJECT: {obj_name}")

        pc_3d = point_clouds[target_idx]
        pc_3d = pc_3d - np.mean(pc_3d, axis=0)
        pc_3d = pc_3d / np.max(np.linalg.norm(pc_3d, axis=1))

        pc_2d = project_to_arbitrary_plane(pc_3d, proj_direction)

        features, bounds, vis_data = holistic_hybrid_extraction(pc_2d, predictor)

        b_min, b_max = -1.2, 1.2
        xx, yy = np.meshgrid(np.linspace(b_min, b_max, 200), np.linspace(b_min, b_max, 200))
        full_field = np.zeros_like(xx)

        for feature in features:
            spline = ImpSpline2D(xx, yy, feature['polygon'], delta=0.005, n=2)
            full_field += spline

        # ==========================================
        # 8-STEP VISUALIZATION GRID (2x4)
        # ==========================================
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle(f"Neuro-Symbolic Pipeline: {obj_name} (Projection Direction: [1, 1, 1])", fontsize=18, fontweight='bold')
        plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9)

        # 1. Point Cloud
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        ax1.scatter(pc_3d[:,0], pc_3d[:,1], pc_3d[:,2], s=2, c='gray', alpha=0.6)
        ax1.view_init(elev=elev_angle, azim=azim_angle)
        ax1.set_title("1. 3D Point Cloud (Matching View)", fontsize=14)

        # 2. Distance Transform
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.imshow(vis_data['dist_transform'], cmap='viridis')
        ax2.set_title("2. Distance Transform", fontsize=14)
        ax2.axis('off')

        # 3. Topology Nodes
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.imshow(vis_data['dist_transform'], cmap='viridis')
        nodes = vis_data['nodes']
        if len(nodes) > 0:
            ax3.scatter(nodes[:,0], nodes[:,1], c='red', s=100, marker='X', edgecolors='white')
        ax3.set_title("3. Topology Nodes (T)", fontsize=14)
        ax3.axis('off')

        # 4. SAM Segmentation
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.imshow(vis_data['mask_binary'], cmap='gray')
        ax4.set_title("4. Neural Segmentation", fontsize=14)
        ax4.axis('off')

        # 5. Hybrid Contour
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.scatter(pc_2d[:,0], pc_2d[:,1], s=1, c='lightgray', alpha=0.5)
        for feat in features:
            hc = feat['hybrid_curve']
            closed_hc = np.vstack((hc, hc[0]))
            ls = '--' if feat['is_hole'] else '-'
            ax5.plot(closed_hc[:,0], closed_hc[:,1], c=feat['color'], lw=2, linestyle=ls)
        ax5.set_title("5. Scale-Space Contour", fontsize=14)
        ax5.axis('equal'); ax5.grid(True, linestyle=':')
        ax5.set_xlim([b_min, b_max]); ax5.set_ylim([b_min, b_max])

        # 6. Sparse Polygon
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.scatter(pc_2d[:,0], pc_2d[:,1], s=1, c='lightgray', alpha=0.5)
        for feat in features:
            poly = feat['polygon']
            is_hole = feat['is_hole']
            closed_poly = np.vstack((poly, poly[0]))

            # Plot boundary
            ls = '--' if is_hole else '-'
            ax6.plot(closed_poly[:,0], closed_poly[:,1], c=feat['color'], lw=2, linestyle=ls, marker='o', markersize=4, markerfacecolor='black')

            # Visually subtract hole using white fill
            fill_color = 'white' if is_hole else feat['color']
            alpha_val = 1.0 if is_hole else 0.3
            ax6.fill(closed_poly[:,0], closed_poly[:,1], color=fill_color, alpha=alpha_val)

        ax6.set_title("6. Control Polygon (G)", fontsize=14)
        ax6.axis('equal'); ax6.grid(True, linestyle=':')
        ax6.set_xlim([b_min, b_max]); ax6.set_ylim([b_min, b_max])

        # 7. Area Spline Field
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.imshow(full_field, extent=[b_min, b_max, b_min, b_max], origin='lower', cmap='Purples', alpha=0.9)
        ax7.set_title("7. Area Spline Field", fontsize=14)
        ax7.axis('equal'); ax7.grid(True, linestyle=':')
        ax7.set_xlim([b_min, b_max]); ax7.set_ylim([b_min, b_max])

        # 8. F(x,y)=0
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.scatter(pc_2d[:,0], pc_2d[:,1], s=1, c='lightgray', alpha=0.5)
        ax8.contour(xx, yy, full_field, levels=[0.5], colors='purple', linewidths=2.5)
        ax8.set_title("8. F(x,y)=0 Boundary (S)", fontsize=14)
        ax8.axis('equal'); ax8.grid(True, linestyle=':')
        ax8.set_xlim([b_min, b_max]); ax8.set_ylim([b_min, b_max])

        save_path = os.path.join(output_dir, f"{obj_name}_111_view.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    --> Saved robust visualization to: {save_path}")

    print("\nRobustness batch processing complete!")

if __name__ == "__main__":
    run_neuro_symbolic_pipeline()
