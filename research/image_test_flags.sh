#!/usr/bin/env bash
# Optional explicit test shape; sourced by both independent and native builds.
image_test_defs=()
image_ref_defs=()
if [[ -n "${CNN_TEST_IMAGE:-}" ]]; then
    [[ "$CNN_TEST_IMAGE" =~ ^([0-9]+),([0-9]+),([13])$ ]] || { echo 'CNN_TEST_IMAGE must be HEIGHT,WIDTH,CHANNELS.' >&2; return 1; }
    image_test_h=${BASH_REMATCH[1]}; image_test_w=${BASH_REMATCH[2]}; image_test_c=${BASH_REMATCH[3]}
    ((image_test_h >= 36 && image_test_h <= 1024 && image_test_w >= 36 && image_test_w <= 1024)) || return 1
    image_test_defs=(-DCNN_IMAGE_V1 -DCNN_IMAGE_HEIGHT="$image_test_h" -DCNN_IMAGE_WIDTH="$image_test_w" -DCNN_IMAGE_CHANNELS="$image_test_c")
    image_ref_defs=(-DCNN_REF_HEIGHT="$image_test_h" -DCNN_REF_WIDTH="$image_test_w" -DCNN_REF_CHANNELS="$image_test_c")
fi
