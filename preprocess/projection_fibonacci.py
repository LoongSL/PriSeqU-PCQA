import numpy as np
import time
import open3d as o3d
import os
import json
import math
from PIL import Image
import cv2
import argparse
from scipy.spatial import ConvexHull

DEG_TO_PIXEL = 5.82


def fibonacci_sphere(n):
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        theta = np.arccos(1 - 2 * (i + 0.5) / n)
        phi = 2 * np.pi * i / golden_ratio
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)
        z = np.sin(theta) * np.sin(phi)
        points.append([x, y, z])
    return np.array(points)


def enforce_front_view(points):
    front = np.array([0.0, 0.0, 1.0])
    dots = points @ front
    closest_idx = np.argmax(dots)
    points[closest_idx] = front
    if closest_idx != 0:
        points[[0, closest_idx]] = points[[closest_idx, 0]]
    return points


def compute_up_vector(front_vec):
    default_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(front_vec, default_up)) > 0.99:
        return np.array([0.0, 0.0, -1.0])
    return default_up


def compute_adjacency(points):
    hull = ConvexHull(points)
    edges = set()
    for simplex in hull.simplices:
        for j in range(3):
            edge = tuple(sorted([int(simplex[j]), int(simplex[(j + 1) % 3])]))
            edges.add(edge)

    adj = {i: [] for i in range(len(points))}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for k in adj:
        adj[k].sort()

    return {
        "num_views": len(points),
        "viewpoints": points.tolist(),
        "adjacency": {str(k): v for k, v in adj.items()},
    }


def background_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_mean = np.mean(gray, axis=0)
    row_mean = np.mean(gray, axis=1)

    col_mask = col_mean < 255
    row_mask = row_mean < 255

    if not col_mask.any() or not row_mask.any():
        return img

    col_idx = np.where(col_mask)[0]
    row_idx = np.where(row_mask)[0]

    return img[row_idx[0]:row_idx[-1] + 1, col_idx[0]:col_idx[-1] + 1]


def generate_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def capture_frame(vis):
    buf = vis.capture_screen_float_buffer(True)
    img = Image.fromarray((np.asarray(buf) * 255).astype(np.uint8))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return background_crop(img)


def stitch_images(img_dir, images):
    if len(images) == 0:
        return None

    ncols = min(3, len(images))
    nrows = math.ceil(len(images) / ncols)

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    padded = list(images)
    while len(padded) < nrows * ncols:
        padded.append(np.ones((max_h, max_w, 3), dtype=np.uint8) * 255)

    resized = [cv2.resize(img, (max_w, max_h)) for img in padded]

    rows = []
    for r in range(nrows):
        rows.append(np.hstack(resized[r * ncols:(r + 1) * ncols]))

    stitched = np.vstack(rows)
    cv2.imwrite(os.path.join(img_dir, "stitched.png"), stitched)
    return stitched


SIXFACE_ROTATIONS = [
    (0, 0),
    (90 * DEG_TO_PIXEL, 0),
    (90 * DEG_TO_PIXEL, 0),
    (90 * DEG_TO_PIXEL, 0),
    (0, 90 * DEG_TO_PIXEL),
    (0, 180 * DEG_TO_PIXEL),
]


def render_6face_stitch(obj, output_dir, zoom, render_width, render_height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=render_width, height=render_height)
    vis.add_geometry(obj)

    ctrl = vis.get_view_control()
    vis.get_render_option().light_on = False

    ctrl.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()

    images = []
    for rx, ry in SIXFACE_ROTATIONS:
        ctrl.rotate(rx, ry)
        vis.poll_events()
        vis.update_renderer()
        images.append(capture_frame(vis))

    stitch_images(output_dir, images)

    vis.destroy_window()
    del ctrl, vis


def render_fibonacci_views(obj, output_dir, viewpoints, zoom,
                           render_width, render_height, output_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=render_width, height=render_height)
    vis.add_geometry(obj)

    ctrl = vis.get_view_control()
    vis.get_render_option().light_on = False

    center = np.asarray(obj.get_center())

    vis.poll_events()
    vis.update_renderer()

    for i, front_vec in enumerate(viewpoints):
        ctrl.set_front(front_vec.tolist())
        ctrl.set_lookat(center.tolist())
        ctrl.set_up(compute_up_vector(front_vec).tolist())
        ctrl.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()

        img = capture_frame(vis)
        if output_size is not None:
            img = cv2.resize(img, (output_size, output_size))
        cv2.imwrite(os.path.join(output_dir, f"{i}.png"), img)

    vis.destroy_window()
    del ctrl, vis


def project_single(obj_path, obj_type, output_dir, viewpoints,
                   zoom, render_width, render_height,
                   output_size=None):
    if obj_type == 'ply':
        obj = o3d.io.read_point_cloud(obj_path)
    elif obj_type == 'mesh':
        obj = o3d.io.read_triangle_mesh(obj_path, True)
        obj.compute_vertex_normals()
    else:
        raise ValueError(f"Unknown object type: {obj_type}")

    if len(np.asarray(obj.points)) == 0:
        print(f"  [SKIP] empty point cloud: {obj_path}")
        return

    generate_dir(output_dir)

    t0 = time.time()

    render_6face_stitch(obj, output_dir, zoom, render_width, render_height)

    render_fibonacci_views(obj, output_dir, viewpoints, zoom,
                           render_width, render_height, output_size)

    print(f"  {len(viewpoints)} fib views + 6-face stitch -> "
          f"{time.time() - t0:.2f}s")


def batch_project(obj_type, input_dir, output_dir, viewpoints, **kwargs):
    ext = '.ply' if obj_type == 'ply' else '.obj'
    num_views = len(viewpoints)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(ext)])
    print(f"Found {len(files)} {ext} files in {input_dir}")

    done, skipped, failed = 0, 0, 0
    for f in files:
        out_dir = os.path.join(output_dir, f)
        last_view = os.path.join(out_dir, f"{num_views - 1}.png")
        stitch_file = os.path.join(out_dir, "stitched.png")
        if os.path.exists(last_view) and os.path.exists(stitch_file):
            skipped += 1
            continue

        obj_path = os.path.join(input_dir, f)
        print(f"[{done + skipped + failed + 1}/{len(files)}] {f}")
        try:
            project_single(obj_path, obj_type, out_dir, viewpoints, **kwargs)
            done += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\nDone: {done}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Fibonacci Sphere Projection for NR-PCQA")

    parser.add_argument('--type', type=str, default='ply',
                        choices=['ply', 'mesh'])
    parser.add_argument('--path', type=str, required=True,
                        help='Input point cloud directory')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Output image directory')
    parser.add_argument('--num_views', type=int, required=True,
                        help='Number of Fibonacci sphere viewpoints')
    parser.add_argument('--zoom', type=float, required=True)
    parser.add_argument('--render_width', type=int, required=True)
    parser.add_argument('--render_height', type=int, required=True)
    parser.add_argument('--output_size', type=int, default=None,
                        help='Resize output images to NxN')

    args = parser.parse_args()

    points = fibonacci_sphere(args.num_views)
    points = enforce_front_view(points)

    generate_dir(args.img_path)
    adj = compute_adjacency(points)
    adj_path = os.path.join(args.img_path, "adjacency.json")
    with open(adj_path, 'w') as f:
        json.dump(adj, f, indent=2)
    print(f"Adjacency graph saved to {adj_path}")
    for k, v in adj["adjacency"].items():
        print(f"  view {k}: neighbors {v}")

    batch_project(
        obj_type=args.type,
        input_dir=args.path,
        output_dir=args.img_path,
        viewpoints=points,
        zoom=args.zoom,
        render_width=args.render_width,
        render_height=args.render_height,
        output_size=args.output_size,
    )


if __name__ == '__main__':
    main()
