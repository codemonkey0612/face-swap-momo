# Test Fixtures

Place sample face images here for testing:
- `face_with_hand.jpg` — synthetic hand-over-face occlusion
- `clear_face.jpg` — clean face, no occlusion
- `no_face.jpg` — frame with no face

These are NOT committed to git (large binaries). Generate them with:
    python scripts/generate_source_face.py --test-fixtures
