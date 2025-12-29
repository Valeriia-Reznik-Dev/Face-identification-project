# Dataset Structure

This directory contains the dataset for face detection, recognition and tracking.

## Structure

- `train/` - Training images (one folder per person)
- `test/` - Test videos and annotations
  - `videos/` - Test video files (excluded from git due to size)
  - `annotations/` - Ground truth annotations
  - `images/` - Extracted frames (excluded from git due to size)
- `unknown_videos/` - Videos with unknown persons (for open-set testing)
  - `videos/` - Video files (excluded from git due to size)
  - `annotations/` - Annotations (all faces labeled as "unknown")

## Annotation Format

See main README.md for annotation format details.

## Note

Video files and training images are excluded from git due to size constraints. 
Only annotation files are included in the repository.
