package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128
    private const val CHANNELS = 3

    private const val IMAGE_FLOATS = SIZE * SIZE * CHANNELS
    private const val MASK_FLOATS = SIZE * SIZE

    /*
     * Bitmap -> NHWC float tensor (BGR format for DeepFaceLab)
     *
     * Tensor shape: [1, 128, 128, 3]
     * Channel order: B G R (DeepFaceLab uses OpenCV BGR)
     * Range: 0.0 .. 1.0
     */
    fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val resized = if (bitmap.width == SIZE && bitmap.height == SIZE) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        }

        try {
            val pixels = IntArray(SIZE * SIZE)
            resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)

            val tensor = FloatArray(IMAGE_FLOATS)
            var index = 0

            // NHWC: Interleaved BGR (DeepFaceLab expects BGR)
            for (pixel in pixels) {
                tensor[index++] = Color.blue(pixel) / 255.0f
                tensor[index++] = Color.green(pixel) / 255.0f
                tensor[index++] = Color.red(pixel) / 255.0f
            }

            return tensor
        } finally {
            if (resized !== bitmap && !resized.isRecycled) {
                resized.recycle()
            }
        }
    }

    /*
     * NHWC BGR tensor -> Bitmap
     *
     * Input shape: [1, 128, 128, 3]
     * Layout: BGR BGR BGR ...
     * Range: 0.0 .. 1.0
     */
    fun tensorToBitmap(tensor: FloatArray): Bitmap {
        require(tensor.size == IMAGE_FLOATS) {
            "Expected $IMAGE_FLOATS RGB values, got ${tensor.size}"
        }

        val pixels = IntArray(SIZE * SIZE)
        var index = 0

        for (i in pixels.indices) {
            // Model outputs BGR, convert back to RGB for Android Bitmap
            val b = (tensor[index++].coerceIn(0.0f, 1.0f) * 255.0f).toInt()
            val g = (tensor[index++].coerceIn(0.0f, 1.0f) * 255.0f).toInt()
            val r = (tensor[index++].coerceIn(0.0f, 1.0f) * 255.0f).toInt()
            
            pixels[i] = Color.rgb(r, g, b)
        }

        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    /*
     * LIAE mask -> 128x128 Bitmap
     *
     * Model output: [1, 128, 128, 1]
     * The mask value is stored in the Alpha channel for DflMerger.
     */
    fun maskToBitmap(mask: FloatArray): Bitmap {
        require(mask.size == MASK_FLOATS) {
            "Expected $MASK_FLOATS mask values, got ${mask.size}"
        }

        val pixels = IntArray(MASK_FLOATS)

        for (i in pixels.indices) {
            val value = (mask[i].coerceIn(0.0f, 1.0f) * 255.0f).toInt()
            
            // Store mask gradient in Alpha, keep RGB white
            pixels[i] = Color.argb(value, 255, 255, 255)
        }

        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }
}
