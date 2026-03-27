#!/usr/bin/env python3
"""
Track Post-Processing Script

Applies track interpolation to fill gaps in tracking results.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.motip.advanced_strategies import TrackInterpolation


def load_mot_results(filepath):
    """Load MOT format results"""
    tracks = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            
            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][frame_id] = np.array([x, y, w, h])
    
    return tracks


def save_mot_results(tracks, filepath, conf=1.0, cls=1, vis=1):
    """Save results in MOT format"""
    lines = []
    for track_id, track_data in tracks.items():
        for frame_id, box in sorted(track_data.items()):
            x, y, w, h = box
            line = f'{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},{cls},{vis}'
            lines.append(line)
    
    # Sort by frame, then track
    lines.sort(key=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])))
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input MOT file')
    parser.add_argument('--output', required=True, help='Output MOT file')
    parser.add_argument('--max-gap', type=int, default=10, help='Max gap to interpolate')
    parser.add_argument('--method', default='linear', choices=['linear', 'spline'])
    args = parser.parse_args()
    
    # Load tracks
    print(f'Loading tracks from {args.input}...')
    tracks = load_mot_results(args.input)
    print(f'Loaded {len(tracks)} tracks')
    
    # Apply interpolation
    print(f'Applying interpolation (max_gap={args.max_gap}, method={args.method})...')
    interpolator = TrackInterpolation(
        max_gap=args.max_gap,
        interpolation_method=args.method,
    )
    processed_tracks = interpolator.process_tracks(tracks)
    
    # Count interpolated frames
    orig_count = sum(len(t) for t in tracks.values())
    new_count = sum(len(t) for t in processed_tracks.values())
    print(f'Interpolated {new_count - orig_count} frames')
    
    # Save results
    print(f'Saving to {args.output}...')
    save_mot_results(processed_tracks, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
