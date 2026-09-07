package com.liae.android

data class BlazeFaceResult(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,

    // BlazeFace landmarks:
    // 0 = right eye
    // 1 = left eye
    // 2 = nose
    // 3 = mouth
    // 4 = right ear
    // 5 = left ear
    val keypoints: Array<FloatArray> = emptyArray()
)
