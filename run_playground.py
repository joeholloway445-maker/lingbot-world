#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sys
import webbrowser
import numpy as np
import shutil

PORT = 8000
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Standalone implementation of wasd_ijkl_to_c2ws.py to avoid torch/ml dependencies
_ALLOWED_KEYS = frozenset("wasdijkl")

def normalize_action_string(action_string: str) -> str:
    if not action_string:
        return ""
    normalized = action_string.replace("，", ",")
    return "".join(normalized.split())

def parse_action_string_segments(action_string: str):
    s = normalize_action_string(action_string)
    if not s:
        raise ValueError("action_string is empty")
    raw_parts = s.split(",")
    parts = [p for p in raw_parts if p]
    if len(parts) != len(raw_parts):
        raise ValueError("action_string has an empty segment (check commas)")
    segments = []
    total = 0
    for part in parts:
        if "-" not in part:
            raise ValueError(f"Invalid segment {part!r}: expected form <keys>-<duration>")
        keys_part, dur_str = part.rsplit("-", 1)
        if not dur_str.isdigit():
            raise ValueError(f"Invalid duration in segment {part!r}")
        n = int(dur_str)
        if n <= 0:
            raise ValueError(f"Duration must be positive in segment {part!r}")
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys = frozenset()
        else:
            bad = [c for c in keys_lower if c not in _ALLOWED_KEYS]
            if bad:
                raise ValueError(f"Invalid key character(s) {bad!r} in segment {part!r}")
            keys = frozenset(keys_lower)
        segments.append((keys, n))
        total += n
    return segments, total

def segments_to_wasd_ijkl(segments):
    total = sum(n for _, n in segments)
    wasd = np.zeros((total, 4), dtype=np.float32)
    ijkl = np.zeros((total, 4), dtype=np.float32)
    wasd_idx = {"w": 0, "a": 1, "s": 2, "d": 3}
    ijkl_idx = {"i": 0, "j": 1, "k": 2, "l": 3}
    t = 0
    for keys, n in segments:
        for _ in range(n):
            for c in keys:
                if c in wasd_idx:
                    wasd[t, wasd_idx[c]] = 1.0
                else:
                    ijkl[t, ijkl_idx[c]] = 1.0
            t += 1
    return wasd, ijkl

def action_string_to_wasd_ijkl(action_string: str):
    segments, total = parse_action_string_segments(action_string)
    wasd, ijkl = segments_to_wasd_ijkl(segments)
    return wasd, ijkl, total

def wasd_array_to_frame_keys(wasd_array, ijkl_array=None):
    wasd_mapping = ['w', 'a', 's', 'd']
    ijkl_mapping = ['i', 'j', 'k', 'l']
    frame_keys = []
    for idx, frame in enumerate(wasd_array):
        pressed_keys = [wasd_mapping[i] for i in range(4) if frame[i] > 0.5]
        if ijkl_array is not None:
            ijkl_frame = ijkl_array[idx]
            pressed_keys += [ijkl_mapping[i] for i in range(4) if ijkl_frame[i] > 0.5]
        frame_keys.append(pressed_keys)
    return frame_keys

def get_rotation_matrix(axis, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)

def generate_and_save_trajectory(arrow_actions):
    move_speed = 0.05
    rotate_speed_rad = np.deg2rad(2.0)
    current_c2w = np.eye(4)
    current_pitch = 0.0
    pitch_limit = np.deg2rad(85)
    all_matrices = [current_c2w]
    
    for f in range(len(arrow_actions)):
        frame_keys = arrow_actions[f]
        R = current_c2w[:3, :3]
        T = current_c2w[:3, 3]

        pitch_delta = 0.0
        if 'i' in frame_keys: pitch_delta += rotate_speed_rad
        if 'k' in frame_keys: pitch_delta -= rotate_speed_rad
        
        new_pitch = current_pitch + pitch_delta
        if -pitch_limit <= new_pitch <= pitch_limit:
            current_pitch = new_pitch
        else:
            pitch_delta = 0.0
        
        R_pitch = get_rotation_matrix('x', pitch_delta)

        yaw_delta = 0.0
        if 'j' in frame_keys: yaw_delta -= rotate_speed_rad
        if 'l' in frame_keys: yaw_delta += rotate_speed_rad
        R_yaw = get_rotation_matrix('y', yaw_delta)

        R_new = R_yaw @ R @ R_pitch
        vec_right = R_new[:, 0]
        vec_forward = R_new[:, 2]

        forward_flat = np.array([vec_forward[0], 0, vec_forward[2]])
        right_flat   = np.array([vec_right[0],   0, vec_right[2]])

        f_norm = np.linalg.norm(forward_flat)
        r_norm = np.linalg.norm(right_flat)
        forward_flat = forward_flat / (f_norm + 1e-6) if f_norm > 0 else forward_flat
        right_flat   = right_flat / (r_norm + 1e-6) if r_norm > 0 else right_flat

        move_vec = np.zeros(3)
        if 'w' in frame_keys: move_vec += forward_flat * move_speed
        if 's' in frame_keys: move_vec -= forward_flat * move_speed
        if 'd' in frame_keys: move_vec += right_flat * move_speed
        if 'a' in frame_keys: move_vec -= right_flat * move_speed

        T_new = T + move_vec
        current_c2w = np.eye(4)
        current_c2w[:3, :3] = R_new
        current_c2w[:3, 3] = T_new
        all_matrices.append(current_c2w)
        
    return all_matrices

class PlaygroundRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_POST(self):
        if self.path == '/api/export':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                target_dir_rel = data.get('target_dir', 'examples/custom_trajectory').strip()
                action_string = data.get('action_string', '').strip()
                frame_num = int(data.get('frame_num', 1))

                target_dir = os.path.abspath(os.path.join(WORKSPACE_ROOT, target_dir_rel))
                if not target_dir.startswith(os.path.abspath(WORKSPACE_ROOT)):
                    raise ValueError("Target directory must be inside the project workspace.")

                os.makedirs(target_dir, exist_ok=True)

                wasd_action, ijkl_action, str_frames = action_string_to_wasd_ijkl(action_string)
                
                if str_frames < frame_num:
                    pad_len = frame_num - str_frames
                    wasd_action = np.pad(wasd_action, ((0, pad_len), (0, 0)), mode="constant")
                    ijkl_action = np.pad(ijkl_action, ((0, pad_len), (0, 0)), mode="constant")
                else:
                    wasd_action = wasd_action[:frame_num]
                    ijkl_action = ijkl_action[:frame_num]

                frame_keys = wasd_array_to_frame_keys(wasd_action, ijkl_action)
                c2ws_list = generate_and_save_trajectory(frame_keys)
                c2ws = np.array(c2ws_list, dtype=np.float32)

                c2ws_final = c2ws[:frame_num]
                intrinsics = np.zeros((frame_num, 4), dtype=np.float32)
                intrinsics[:, 0] = 480.0
                intrinsics[:, 1] = 480.0
                intrinsics[:, 2] = 416.0
                intrinsics[:, 3] = 240.0

                np.save(os.path.join(target_dir, "wasd_action.npy"), wasd_action)
                np.save(os.path.join(target_dir, "ijkl_action.npy"), ijkl_action)
                np.save(os.path.join(target_dir, "poses.npy"), c2ws_final)
                np.save(os.path.join(target_dir, "intrinsics.npy"), intrinsics)

                image_target_path = os.path.join(target_dir, "image.jpg")
                if not os.path.exists(image_target_path):
                    src_image = os.path.join(WORKSPACE_ROOT, "examples", "05", "image.jpg")
                    if os.path.exists(src_image):
                        shutil.copy(src_image, image_target_path)

                response_data = {
                    'status': 'success',
                    'output_dir': target_dir_rel,
                    'absolute_path': target_dir
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        else:
            super().do_POST()

def main():
    os.chdir(WORKSPACE_ROOT)
    handler = PlaygroundRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"=================================================================")
        print(f" LingBot-World Controller & Trajectory Playground Server Active!")
        print(f" Serving at: http://localhost:{PORT}/assets/playground.html")
        print(f"=================================================================")
        
        webbrowser.open(f"http://localhost:{PORT}/assets/playground.html")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    main()
